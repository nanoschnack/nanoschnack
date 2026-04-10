from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from configuration_nanoschnack import NanoSchnackConfig
_REPO_ROOT = Path(__file__).resolve().parents[1]
# Add the repo root so we can import the local model code and checkpoints.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model import config as base_config
from model.checkpointer import (
    apply_checkpoint_config,
    load_model_state_dict,
    normalize_state_dict,
    resize_vocab_state_dict,
    select_state_dict,
)
from model.gpt import GPT


def _apply_hf_config_to_globals(config):
    base_config.CONTEXT_LEN = config.context_len
    base_config.EMBED_SIZE = config.embed_size
    base_config.NUM_LAYERS = config.num_layers
    base_config.NUM_HEADS = config.num_heads
    base_config.HIDDEN_SIZE = config.hidden_size
    if config.vocab_size:
        base_config.VOCAB_SIZE = config.vocab_size


def _sync_hf_config(config):
    config.context_len = base_config.CONTEXT_LEN
    config.embed_size = base_config.EMBED_SIZE
    config.num_layers = base_config.NUM_LAYERS
    config.num_heads = base_config.NUM_HEADS
    config.hidden_size = base_config.HIDDEN_SIZE
    config.vocab_size = base_config.VOCAB_SIZE


def _resolve_checkpoint_path(model_path, config):
    if config.checkpoint_path:
        path = Path(config.checkpoint_path)
        if not path.is_absolute():
            path = Path(model_path) / path
        return path
    return _REPO_ROOT / "checkpoints" / "latest.pt"


def _load_checkpoint(model, config, model_path):
    checkpoint_path = _resolve_checkpoint_path(model_path, config)
    ckpt = None
    ckpt_vocab_size = None
    if checkpoint_path and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        ckpt_config = ckpt.get("config") if isinstance(ckpt, dict) else None
        if ckpt_config:
            apply_checkpoint_config(ckpt_config)
        ckpt_vocab_size = ckpt.get("vocab_size") if isinstance(ckpt, dict) else None
        if ckpt_vocab_size:
            base_config.VOCAB_SIZE = ckpt_vocab_size
        _sync_hf_config(config)
    if ckpt is None:
        return model
    state_dict = select_state_dict(ckpt)
    if state_dict is None:
        raise RuntimeError(f"Checkpoint at {checkpoint_path} has no model weights.")
    state_dict = normalize_state_dict(state_dict)
    resize_vocab_state_dict(model.gpt, state_dict, ckpt_vocab_size=ckpt_vocab_size)
    load_model_state_dict(model.gpt, state_dict)
    return model


class NanoSchnackModel(PreTrainedModel):
    """Base NanoSchnack model wrapper used by Transformers and vLLM.

    Constructs the GPT backbone and returns hidden states.
    Keeps hyperparameters aligned with model/config.py.
    """
    config_class = NanoSchnackConfig
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        _apply_hf_config_to_globals(config)
        vocab_size = base_config.VOCAB_SIZE or config.vocab_size
        self.gpt = GPT(
            vocab_size=vocab_size,
            embed_size=base_config.EMBED_SIZE,
            num_layers=base_config.NUM_LAYERS,
            num_heads=base_config.NUM_HEADS,
            hidden_size=base_config.HIDDEN_SIZE,
            context_len=base_config.CONTEXT_LEN,
        )
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required.")
        seq_length = input_ids.size(1)
        positions = torch.arange(0, seq_length, device=input_ids.device).unsqueeze(0)
        hidden_states = self.gpt.tok(input_ids) + self.gpt.pos(positions)
        for block in self.gpt.blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        hidden_states = self.gpt.ln(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = NanoSchnackConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config)
        return _load_checkpoint(model, config, pretrained_model_name_or_path)


class NanoSchnackForCausalLM(PreTrainedModel):
    """Causal LM wrapper around the NanoSchnack base model.

    Loads repo checkpoints directly without a HF weight export step.
    Reuses GPT weights for tied embeddings and output projection.
    """
    config_class = NanoSchnackConfig
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        self.model = NanoSchnackModel(config)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required.")
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            **kwargs,
        ).last_hidden_state
        logits = self.model.gpt.lm(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = NanoSchnackConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config)
        _load_checkpoint(model.model, config, pretrained_model_name_or_path)
        return model
