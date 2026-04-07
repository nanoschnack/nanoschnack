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
        # Run a prefill or incremental decode step and return updated cache.
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, seq_len).")
        if cache is None:
            cache = self.empty_kv_cache()
        if len(cache) != len(self.blocks):
            raise ValueError("cache length must match the number of transformer blocks.")

        _, seq_len = input_ids.shape
        past_len = self.cache_seq_len(cache[0]) if cache else 0
        x = self.embed_inputs(input_ids, start_pos=past_len)

        new_cache = []
        for block, layer_cache in zip(self.blocks, cache):
            x, next_layer_cache = self._forward_block_cached(block, x, layer_cache)
            new_cache.append(next_layer_cache)
        x = self.ln(x)
        return self.lm(x), new_cache

    def _forward_block_cached(self, block, x, layer_cache):
        # Preserve the baseline pre-norm block structure around cached attention.
        attn_out, next_layer_cache = self._forward_attn_cached(block.attn, block.ln1(x), layer_cache)
        x = x + attn_out
        x = x + block.mlp(block.ln2(x))
        return x, next_layer_cache

    def _forward_attn_cached(self, attn, x, layer_cache):
        # Recompute only new QKV projections and append K/V to the running cache.
        batch_size, seq_len, embed_size = x.shape
        q, k, v = project_qkv(attn, x)

        past_len = self.cache_seq_len(layer_cache)
        if attn.rope is not None:
            positions = torch.arange(past_len, past_len + seq_len, device=x.device)
            q, k = apply_rope_positions(attn.rope, q, k, positions)

        if layer_cache is not None:
            full_k = torch.cat((layer_cache.key, k), dim=-2)
            full_v = torch.cat((layer_cache.value, v), dim=-2)
        else:
            full_k = k
            full_v = v

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
        y = merge_attention_heads(y, batch_size, seq_len, embed_size)
        y = attn.proj(y)
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
