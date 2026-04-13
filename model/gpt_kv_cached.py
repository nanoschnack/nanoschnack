from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from gpt_base import GPTBase
    from gpt_shared import apply_rope_positions, build_causal_mask, merge_attention_heads, project_qkv
except ModuleNotFoundError:
    from .gpt_base import GPTBase
    from .gpt_shared import apply_rope_positions, build_causal_mask, merge_attention_heads, project_qkv


class GPTKVCached(GPTBase):
    """Inference-only GPT variant with explicit KV-cached decoding helpers.

    Reuses the exact NanoSchnack GPT module weights and block structure.
    Keeps the baseline full-sequence forward pass from `GPT` untouched.
    Adds chunked incremental decode helpers for parity and serving prep.
    """

    def empty_kv_cache(self):
        # Start with no layer cache entries before prompt prefill.
        return [None] * len(self.blocks)

    @staticmethod
    def cache_seq_len(layer_cache):
        if layer_cache is None:
            return 0
        return layer_cache.seq_len

    def forward_cached(self, input_ids, cache=None):
        # Run one cached decode step.
        #
        # Two common cases:
        #
        # 1. Prefill with a whole prompt:
        #      input_ids = [t0 t1 t2 t3]
        #      cache     = empty
        #
        #    We compute logits for all prompt positions and build a fresh
        #    per-layer KV cache that now covers t0..t3.
        #
        # 2. Incremental decode with one new token:
        #      input_ids = [t4]
        #      cache     = KV for t0..t3
        #
        #    We compute Q/K/V only for t4, append K/V to the cache, and let
        #    the new query attend over the full history t0..t4.
        #
        # Visually, each layer evolves like this:
        #
        #   before: K/V = [past........................]
        #   new:         [cur.]
        #   after: K/V = [past........................|cur.]
        #
        # The model math stays the same as full-sequence causal decoding.
        # We only avoid recomputing old keys and values.
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, seq_len).")
        if cache is None:
            cache = self.empty_kv_cache()
        if len(cache) != len(self.blocks):
            raise ValueError("cache length must match the number of transformer blocks.")

        _, seq_len = input_ids.shape
        past_len = self.cache_seq_len(cache[0]) if cache else 0

        # Embed just the new chunk.
        #
        # Learned positions use absolute offsets:
        #   prefill: start_pos = 0
        #   decode:  start_pos = number of cached tokens
        #
        # RoPE ignores this addition here and instead uses explicit positions
        # later in attention, but learned-position checkpoints need it.
        x = self.embed_inputs(input_ids, start_pos=past_len)

        # Walk layer-by-layer exactly like a normal GPT forward pass, but keep
        # a separate KV cache entry for each layer:
        #
        #   layer 0: x -> x'  and cache_0 -> cache_0'
        #   layer 1: x'-> x'' and cache_1 -> cache_1'
        #   ...
        #
        # The hidden state flows forward through the network, while the cache
        # grows sideways per layer.
        new_cache = []
        for block, layer_cache in zip(self.blocks, cache):
            x, next_layer_cache = self._forward_block_cached(block, x, layer_cache)
            new_cache.append(next_layer_cache)

        # Finish exactly like the plain GPT path:
        #
        #   hidden -> final layer norm -> tied LM head -> logits
        #
        # Return both:
        #   - logits for the new chunk
        #   - updated per-layer cache for the next decode step
        x = self.ln(x)
        return self.lm(x), new_cache

    def _forward_block_cached(self, block, x, layer_cache):
        # Keep the block topology identical to the plain GPT block:
        #
        #   x
        #   ├─ ln1 -> attention -> + residual
        #   └─ ln2 -> mlp       -> + residual
        #
        # The only change is that the attention branch is cache-aware.
        attn_out, next_layer_cache = self._forward_attn_cached(block.attn, block.ln1(x), layer_cache)
        x = x + attn_out
        x = x + block.mlp(block.ln2(x))
        return x, next_layer_cache

    def _forward_attn_cached(self, attn, x, layer_cache):
        # Cache-aware attention for one layer.
        #
        # Input:
        #   x           = hidden states for only the new chunk
        #   layer_cache = old K/V for this layer, or None during prefill
        #
        # Output:
        #   y           = attention output for the new chunk only
        #   next_cache  = old cache with the new K/V appended
        #
        # Shape picture:
        #
        #   q:      (B, H, new_T, D)
        #   k_new:  (B, H, new_T, D)
        #   v_new:  (B, H, new_T, D)
        #
        #   k_full: (B, H, past_T + new_T, D)
        #   v_full: (B, H, past_T + new_T, D)
        #
        # Only q/k/v for the new chunk are projected. Old k/v come from cache.
        #
        # Compare the work done here to plain full-sequence attention:
        #
        #   plain step N:
        #     x      = [h0 h1 h2 ... hN]
        #     q/k/v  = project all h0..hN again
        #
        #   cached step N:
        #     cache  = [k0..kN-1], [v0..vN-1]
        #     x      = [hN]                  or, more generally, the new chunk
        #     q/k/v  = project only hN
        #     full_k = [k0..kN-1 | kN]
        #     full_v = [v0..vN-1 | vN]
        #
        # So the saving is simple: old keys/values are reused instead of
        # reprojected.
        batch_size, seq_len, embed_size = x.shape
        q, k, v = project_qkv(attn, x)

        past_len = self.cache_seq_len(layer_cache)

        # RoPE must use absolute token positions, not "positions within the
        # current chunk". During incremental decode the new token at local index
        # 0 might really be absolute position 417.
        #
        #   local positions:     [0 1 ... new_T-1]
        #   absolute positions:  [past_T ... past_T+new_T-1]
        if attn.rope is not None:
            positions = torch.arange(past_len, past_len + seq_len, device=x.device)
            q, k = apply_rope_positions(attn.rope, q, k, positions)

        # Append the new keys/values to the cached history:
        #
        #   old: [k0 k1 k2]
        #   new: [k3]
        #   out: [k0 k1 k2 k3]
        #
        # Queries are *not* cached because we only need queries for the current
        # step. Future steps will produce their own fresh queries.
        #
        # Intuition:
        #
        #   past tokens ask old questions about old prefixes
        #   new token asks one new question about the full prefix
        #
        # Once a token's query has been used to produce its hidden state, we do
        # not need that query again. Future decode steps only need the token's
        # key/value so later queries can attend to it.
        if layer_cache is not None:
            full_k = torch.cat((layer_cache.key, k), dim=-2)
            full_v = torch.cat((layer_cache.value, v), dim=-2)
        else:
            full_k = k
            full_v = v

        # Build an explicit mask for "new queries can see all keys up to their
        # own absolute position". We pass is_causal=False because the built-in
        # triangular mask only understands a fresh square attention matrix,
        # while cached decode has a rectangular layout:
        #
        #   queries = new chunk only
        #   keys    = past chunk + new chunk
        #
        # Example with past_T=3 and new_T=2:
        #
        #   keys:    [0 1 2 3 4]
        #   query 3: [1 1 1 1 0]
        #   query 4: [1 1 1 1 1]
        #
        # Score matrix view:
        #
        #              keys / values
        #            k0  k1  k2  k3  k4
        #   q(new=3)  *   *   *   *   .
        #   q(new=4)  *   *   *   *   *
        #
        #   * = allowed attention score
        #   . = masked future position
        #
        # For the common one-token decode case (new_T=1), the matrix collapses
        # to a single row:
        #
        #            k0  k1  k2  k3
        #   q(new=3)  *   *   *   *
        #
        # That is why this method "only computes the last token": the query
        # side has one row, but that row still attends over the full prefix.
        attn_mask = build_causal_mask(seq_len, full_k.shape[-2], past_len, x.device)
        dropout_p = attn.dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q,
            full_k,
            full_v,
            attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),
            is_causal=False,
            dropout_p=dropout_p,
        )

        # Merge heads and run the output projection exactly like plain attention.
        #
        # The SDPA output still has head structure:
        #
        #   y: (B, H, new_T, D)
        #
        # We fold heads back into the model dimension and apply the same output
        # projection as the plain attention path:
        #
        #   per-head outputs -> concat heads -> linear proj -> dropout
        y = merge_attention_heads(y, batch_size, seq_len, embed_size)
        y = attn.proj(y)

        # Store detached K/V tensors so the cache is a pure inference artifact,
        # not part of the backward graph.
        #
        # After this layer, the cache timeline has advanced from:
        #
        #   before: [past................]
        #   after:  [past................|new.]
        #
        # The caller passes this updated cache into the next decode step.
        next_layer_cache = KVLayerCache(key=full_k.detach(), value=full_v.detach())
        return attn.dropout(y), next_layer_cache


@dataclass
class KVLayerCache:
    """Stores cached attention keys and values for one transformer layer.

    The cache grows monotonically with autoregressive decoding.
    Keys and values keep the standard `(B, H, T, D)` layout.
    Cache entries are immutable from the caller perspective.
    """

    key: torch.Tensor
    value: torch.Tensor

    @property
    def seq_len(self):
        return int(self.key.shape[-2])
