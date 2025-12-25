import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

import config
from checkpointer import (
    load_model_state_dict,
    normalize_state_dict,
    resize_vocab_state_dict,
    select_state_dict,
)
from gpt import GPT
from tokenizer import load_tokenizer

try:
    from llm_transparency_tool.models.transparent_llm import ModelInfo, TransparentLlm
except Exception:
    @dataclass
    class ModelInfo:
        name: str
        n_params_estimate: int
        n_layers: int
        n_heads: int
        d_model: int
        d_vocab: int

    class TransparentLlm(ABC):
        @abstractmethod
        def model_info(self) -> ModelInfo:
            pass

        @abstractmethod
        def run(self, sentences: List[str]) -> None:
            pass

        @abstractmethod
        def batch_size(self) -> int:
            pass

        @abstractmethod
        def tokens(self) -> torch.Tensor:
            pass

        @abstractmethod
        def tokens_to_strings(self, tokens: torch.Tensor) -> List[str]:
            pass

        @abstractmethod
        def logits(self) -> torch.Tensor:
            pass

        @abstractmethod
        def unembed(self, t: torch.Tensor, normalize: bool) -> torch.Tensor:
            pass

        @abstractmethod
        def residual_in(self, layer: int) -> torch.Tensor:
            pass

        @abstractmethod
        def residual_after_attn(self, layer: int) -> torch.Tensor:
            pass

        @abstractmethod
        def residual_out(self, layer: int) -> torch.Tensor:
            pass

        @abstractmethod
        def ffn_out(self, layer: int) -> torch.Tensor:
            pass

        @abstractmethod
        def decomposed_ffn_out(
            self, batch_i: int, layer: int, pos: int
        ) -> torch.Tensor:
            pass

        @abstractmethod
        def neuron_activations(
            self, batch_i: int, layer: int, pos: int
        ) -> torch.Tensor:
            pass

        @abstractmethod
        def neuron_output(self, layer: int, neuron: int) -> torch.Tensor:
            pass

        @abstractmethod
        def attention_matrix(
            self, batch_i: int, layer: int, head: int
        ) -> torch.Tensor:
            pass

        @abstractmethod
        def attention_output(
            self, batch_i: int, layer: int, pos: int, head: int
        ) -> torch.Tensor:
            pass

        @abstractmethod
        def decomposed_attn(self, batch_i: int, layer: int) -> torch.Tensor:
            pass


@dataclass
class _RunInfo:
    tokens: torch.Tensor
    logits: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    resid_in: List[torch.Tensor]
    resid_mid: List[torch.Tensor]
    resid_out: List[torch.Tensor]
    ffn_out: List[torch.Tensor]
    ffn_pre: List[torch.Tensor]
    ffn_post: List[torch.Tensor]
    attn_pattern: List[torch.Tensor]
    attn_v: List[torch.Tensor]
    attn_per_head_out: List[torch.Tensor]
    attn_out: List[torch.Tensor]


class NanoSchnackTransparentLlm(TransparentLlm):
    """Adapter that exposes NanoSchnack internals via TransparentLlm.

    Loads a GPT checkpoint and captures residual stream, MLP, and attention tensors.
    Uses forward hooks and recomputation to avoid modifying model modules.
    Intended for inference-only transparency workflows.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        tokenizer=None,
        prepend_bos: bool = False,
        bos_token: str = "[BOS]",
        dtype: torch.dtype = torch.float32,
    ):
        # Resolve device selection and checkpoint path.
        if device == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("Asked to run on gpu, but torch couldn't find cuda.")
            resolved_device = torch.device("cuda")
        elif device == "cpu":
            resolved_device = torch.device("cpu")
        else:
            raise RuntimeError(f"Specified device {device} is not a valid option.")

        checkpoint = None
        ckpt_vocab_size = None
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
            if isinstance(checkpoint, dict):
                ckpt_vocab_size = checkpoint.get("vocab_size")
            checkpoint = select_state_dict(checkpoint)

        # Load tokenizer and align vocabulary with config.
        self._tokenizer = tokenizer or load_tokenizer()
        vocab_size = self._tokenizer.get_vocab_size()
        if config.VOCAB_SIZE == 0:
            config.VOCAB_SIZE = vocab_size
        elif config.VOCAB_SIZE != vocab_size:
            raise ValueError(
                f"Tokenizer vocab {vocab_size} does not match config {config.VOCAB_SIZE}."
            )

        # Build the GPT model using config as the source of truth.
        self._model = GPT(
            vocab_size=config.VOCAB_SIZE,
            embed_size=config.EMBED_SIZE,
            num_layers=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            hidden_size=config.HIDDEN_SIZE,
            context_len=config.CONTEXT_LEN,
        ).to(resolved_device)
        self._model.eval()
        self._device = resolved_device
        self._dtype = dtype
        self._prepend_bos = prepend_bos
        self._bos_token = bos_token
        self._last_run: Optional[_RunInfo] = None
        self._run_exception = RuntimeError(
            "Tried to use the model output before calling the `run` method."
        )

        if checkpoint is not None:
            # Normalize and resize checkpoint weights before loading.
            state_dict = normalize_state_dict(checkpoint)
            resize_vocab_state_dict(
                self._model,
                state_dict,
                ckpt_vocab_size=ckpt_vocab_size,
            )
            load_model_state_dict(self._model, state_dict)

    def model_info(self) -> ModelInfo:
        # Report architecture metadata inferred from config and parameters.
        param_count = sum(param.numel() for param in self._model.parameters())
        if param_count > 0:
            n_params_estimate = 10 ** int(math.log10(param_count))
        else:
            n_params_estimate = 0
        return ModelInfo(
            name="nanoschnack",
            n_params_estimate=n_params_estimate,
            n_layers=config.NUM_LAYERS,
            n_heads=config.NUM_HEADS,
            d_model=config.EMBED_SIZE,
            d_vocab=config.VOCAB_SIZE,
        )

    def copy(self):
        import copy
        return copy.copy(self)

    @torch.no_grad()
    def run(self, sentences: List[str]) -> None:
        # Encode sentences and pad to a single batch tensor.
        encoded = [self._tokenizer.encode(sentence).ids for sentence in sentences]
        if self._prepend_bos:
            bos_id = self._tokenizer.token_to_id(self._bos_token)
            if bos_id is not None:
                encoded = [[bos_id] + ids for ids in encoded]

        max_len = max(len(ids) for ids in encoded) if encoded else 0
        pad_id = self._tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            pad_id = 0
        tokens = torch.full(
            (len(encoded), max_len),
            pad_id,
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.zeros(
            (len(encoded), max_len),
            dtype=torch.long,
            device=self._device,
        )
        for i, ids in enumerate(encoded):
            if not ids:
                continue
            ids_tensor = torch.tensor(ids, dtype=torch.long, device=self._device)
            tokens[i, : len(ids)] = ids_tensor
            attention_mask[i, : len(ids)] = 1

        # Capture activations via hooks to avoid changing model modules.
        cache = self._run_with_cache(tokens, attention_mask)

        self._last_run = _RunInfo(
            tokens=tokens,
            logits=cache["logits"],
            attention_mask=attention_mask,
            resid_in=cache["resid_in"],
            resid_mid=cache["resid_mid"],
            resid_out=cache["resid_out"],
            ffn_out=cache["ffn_out"],
            ffn_pre=cache["ffn_pre"],
            ffn_post=cache["ffn_post"],
            attn_pattern=cache["attn_pattern"],
            attn_v=cache["attn_v"],
            attn_per_head_out=cache["attn_per_head_out"],
            attn_out=cache["attn_out"],
        )

    def batch_size(self) -> int:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.logits.shape[0]

    def tokens(self) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.tokens

    def tokens_to_strings(self, tokens: torch.Tensor) -> List[str]:
        return [self._tokenizer.decode([int(token)]) for token in tokens]

    def logits(self) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.logits

    @torch.no_grad()
    def unembed(self, t: torch.Tensor, normalize: bool) -> torch.Tensor:
        # Project a residual vector into vocabulary space.
        tdim = t.unsqueeze(0).unsqueeze(0)
        if normalize:
            normalized = self._model.ln(tdim)
            logits = self._model.lm(normalized)
        else:
            logits = self._model.lm(tdim.to(self._dtype))
        return logits[0][0]

    def residual_in(self, layer: int) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.resid_in[layer]

    def residual_after_attn(self, layer: int) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.resid_mid[layer]

    def residual_out(self, layer: int) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.resid_out[layer]

    def ffn_out(self, layer: int) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.ffn_out[layer]

    def decomposed_ffn_out(
        self,
        batch_i: int,
        layer: int,
        pos: int,
    ) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        activations = self._last_run.ffn_post[layer][batch_i][pos]
        weight = self._model.blocks[layer].mlp.output.weight
        return activations.unsqueeze(1) * weight.t()

    def neuron_activations(
        self,
        batch_i: int,
        layer: int,
        pos: int,
    ) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.ffn_post[layer][batch_i][pos]

    def neuron_output(self, layer: int, neuron: int) -> torch.Tensor:
        weight = self._model.blocks[layer].mlp.output.weight
        return weight.t()[neuron]

    def attention_matrix(
        self, batch_i: int, layer: int, head: int
    ) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.attn_pattern[layer][batch_i][head]

    def attention_output(
        self,
        batch_i: int,
        layer: int,
        pos: int,
        head: Optional[int] = None,
    ) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        if head is None:
            return self._last_run.attn_out[layer][batch_i][pos]
        return self._last_run.attn_per_head_out[layer][batch_i][pos][head]

    def attention_output_per_head(
        self,
        batch_i: int,
        layer: int,
        pos: int,
        head: int,
    ) -> torch.Tensor:
        return self.attention_output(batch_i, layer, pos, head=head)

    @torch.no_grad()
    def decomposed_attn(self, batch_i: int, layer: int) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        pattern = self._last_run.attn_pattern[layer][batch_i]
        v = self._last_run.attn_v[layer][batch_i]
        v = v.permute(1, 0, 2)
        pattern = pattern.permute(1, 2, 0)
        z = pattern.unsqueeze(-1) * v.unsqueeze(0)
        weight = self._model.blocks[layer].attn.proj.weight
        n_heads = self._model.blocks[layer].attn.num_heads
        embed_size = weight.shape[0]
        weight = weight.view(embed_size, n_heads, -1)
        return torch.einsum("t s h d, m h d -> t s h m", z, weight)

    def _run_with_cache(self, tokens: torch.Tensor, attention_mask: torch.Tensor):
        # Install hooks to capture tensors needed for transparency methods.
        num_layers = len(self._model.blocks)
        cache = {
            "resid_in": [None] * num_layers,
            "resid_mid": [None] * num_layers,
            "resid_out": [None] * num_layers,
            "ffn_out": [None] * num_layers,
            "ffn_pre": [None] * num_layers,
            "ffn_post": [None] * num_layers,
            "attn_pattern": [None] * num_layers,
            "attn_v": [None] * num_layers,
            "attn_per_head_out": [None] * num_layers,
            "attn_out": [None] * num_layers,
        }
        handles = []

        def _mask_attention(scores):
            # Apply causal and padding masks to raw attention scores.
            seq_len = scores.shape[-1]
            causal = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            masked = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
            if attention_mask is not None:
                key_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
                masked = masked.masked_fill(key_mask, torch.finfo(scores.dtype).min)
            return masked

        def make_block_pre_hook(layer_idx):
            def _hook(module, inputs):
                cache["resid_in"][layer_idx] = inputs[0].detach()
            return _hook

        def make_block_hook(layer_idx):
            def _hook(module, inputs, output):
                cache["resid_out"][layer_idx] = output.detach()
            return _hook

        def make_attn_pre_hook(layer_idx):
            def _hook(module, inputs):
                x = inputs[0]
                qkv = module.qkv(x)
                q, k, v = qkv.chunk(3, dim=-1)
                batch, seq_len, _ = x.shape
                head_dim = module.head_dim
                n_heads = module.num_heads
                q = q.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
                k = k.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
                v = v.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)

                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
                scores = _mask_attention(scores)
                pattern = torch.softmax(scores, dim=-1)
                y = torch.matmul(pattern, v)

                embed_size = module.proj.weight.shape[0]
                weight = module.proj.weight.view(embed_size, n_heads, head_dim)
                per_head_out = torch.einsum("b h t d, m h d -> b t h m", y, weight)

                cache["attn_pattern"][layer_idx] = pattern.detach()
                cache["attn_v"][layer_idx] = v.detach()
                cache["attn_per_head_out"][layer_idx] = per_head_out.detach()
            return _hook

        def make_attn_hook(layer_idx):
            def _hook(module, inputs, output):
                cache["attn_out"][layer_idx] = output.detach()
            return _hook

        def make_mlp_pre_hook(layer_idx):
            def _hook(module, inputs):
                x = inputs[0]
                pre_act = module.input(x)
                post_act = F.gelu(pre_act)
                cache["ffn_pre"][layer_idx] = pre_act.detach()
                cache["ffn_post"][layer_idx] = post_act.detach()
            return _hook

        def make_mlp_hook(layer_idx):
            def _hook(module, inputs, output):
                cache["ffn_out"][layer_idx] = output.detach()
            return _hook

        for idx, block in enumerate(self._model.blocks):
            handles.append(block.register_forward_pre_hook(make_block_pre_hook(idx)))
            handles.append(block.register_forward_hook(make_block_hook(idx)))
            handles.append(block.attn.register_forward_pre_hook(make_attn_pre_hook(idx)))
            handles.append(block.attn.register_forward_hook(make_attn_hook(idx)))
            handles.append(block.mlp.register_forward_pre_hook(make_mlp_pre_hook(idx)))
            handles.append(block.mlp.register_forward_hook(make_mlp_hook(idx)))

        logits = self._model(tokens, attention_mask=attention_mask)

        for handle in handles:
            handle.remove()

        # Derive residual after attention using cached residuals and attention output.
        for idx in range(num_layers):
            cache["resid_mid"][idx] = cache["resid_in"][idx] + cache["attn_out"][idx]

        cache["logits"] = logits
        return cache
