from dataclasses import dataclass
from pathlib import Path

import torch

import config


@dataclass
class CheckpointMetadata:
    """Shared checkpoint metadata and normalized state for inference loaders.

    Stores the raw checkpoint payload plus the normalized model state dict.
    Carries the resolved config fragment that should drive model construction.
    Keeps legacy checkpoint defaults explicit instead of scattering them.
    """

    checkpoint_path: Path
    checkpoint: object
    config: dict
    state_dict: dict | None
    vocab_size: int | None


def apply_checkpoint_config(ckpt_config):
    # Apply checkpoint hyperparameters to global config.
    if not ckpt_config:
        return
    for name in (
        "CONTEXT_LEN",
        "VOCAB_SIZE",
        "EMBED_SIZE",
        "NUM_LAYERS",
        "NUM_HEADS",
        "HIDDEN_SIZE",
        "ROPE_BASE",
    ):
        if name in ckpt_config:
            setattr(config, name, ckpt_config[name])
    if "POST_TRAINING" in ckpt_config:
        config.POST_TRAINING = ckpt_config["POST_TRAINING"]
    if "TOKENIZER_FILENAME" in ckpt_config:
        config.TOKENIZER_FILENAME = ckpt_config["TOKENIZER_FILENAME"]
    else:
        # Older checkpoints predate tokenizer filename pinning, so keep them on
        # the legacy tokenizer instead of inheriting the newer default.
        config.TOKENIZER_FILENAME = "tokenizer.json"
    if "POS_EMBED_TYPE" in ckpt_config:
        config.POS_EMBED_TYPE = ckpt_config["POS_EMBED_TYPE"]
    else:
        config.POS_EMBED_TYPE = "learned"


def resolve_checkpoint_config(ckpt):
    # Resolve embedded config while keeping legacy checkpoint defaults explicit.
    if not isinstance(ckpt, dict):
        return {}
    ckpt_config = ckpt.get("config")
    if not isinstance(ckpt_config, dict):
        return {}
    resolved = dict(ckpt_config)
    resolved.setdefault("TOKENIZER_FILENAME", "tokenizer.json")
    resolved.setdefault("POS_EMBED_TYPE", "learned")
    return resolved


def load_checkpoint_config(checkpoint_path):
    """Load checkpoint and apply its config. Returns the checkpoint dict."""
    metadata = load_checkpoint_metadata(checkpoint_path, apply_config=True)
    if metadata is None:
        return None
    return metadata.checkpoint


def strip_state_dict_prefix(state_dict, prefix):
    # Strip a common prefix applied by wrappers like DataParallel.
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(key.startswith(prefix) for key in keys):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def normalize_state_dict(state_dict):
    # Normalize wrapper prefixes to support older checkpoint formats.
    state_dict = strip_state_dict_prefix(state_dict, "module.")
    state_dict = strip_state_dict_prefix(state_dict, "_orig_mod.")
    return strip_state_dict_prefix(state_dict, "model.")


def select_state_dict(ckpt, apply_config=True):
    # Extract the model weights from known checkpoint layouts.
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            if apply_config:
                apply_checkpoint_config(resolve_checkpoint_config(ckpt))
            return ckpt["model"]
        for key in ("model_state_dict", "state_dict"):
            if key in ckpt:
                return ckpt[key]
        return None
    return ckpt


def load_checkpoint_metadata(checkpoint_path, map_location="cpu", apply_config=False):
    # Load a checkpoint once and expose its normalized metadata for shared use.
    if checkpoint_path is None:
        return None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    resolved_config = resolve_checkpoint_config(checkpoint)
    if apply_config:
        apply_checkpoint_config(resolved_config)

    state_dict = select_state_dict(checkpoint, apply_config=False)
    if state_dict is not None:
        state_dict = normalize_state_dict(state_dict)

    vocab_size = None
    if isinstance(checkpoint, dict):
        vocab_size = checkpoint.get("vocab_size")

    return CheckpointMetadata(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        config=resolved_config,
        state_dict=state_dict,
        vocab_size=vocab_size,
    )
