import time
from pathlib import Path
import copy

import torch

import config

# Retain periodic checkpoint snapshots for easy rollback.
SNAPSHOT_INTERVALS = (
    ("10min", 10 * 60),
    ("1h", 60 * 60),
    ("2h", 2 * 60 * 60),
    ("4h", 4 * 60 * 60),
    ("8h", 8 * 60 * 60),
    ("24h", 24 * 60 * 60),
)

def apply_checkpoint_config(ckpt_config):
    # Apply checkpoint hyperparameters to global config.
    if not ckpt_config:
        return
    for name in ("CONTEXT_LEN", "VOCAB_SIZE", "EMBED_SIZE", "NUM_LAYERS", "NUM_HEADS", "HIDDEN_SIZE"):
        if name in ckpt_config:
            setattr(config, name, ckpt_config[name])


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


def select_state_dict(ckpt):
    # Extract the model weights from known checkpoint layouts.
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            apply_checkpoint_config(ckpt.get("config"))
            return ckpt["model"]
        for key in ("model_state_dict", "state_dict"):
            if key in ckpt:
                return ckpt[key]
        return None
    return ckpt


def _resolve_vocab_size(state_dict):
    # Derive vocab size from known token embedding weights.
    for key in ("tok.weight", "lm.weight"):
        tensor = state_dict.get(key)
        if tensor is not None:
            return tensor.shape[0]
    return None


def _select_state_dict_key(state_dict, key):
    # Find a matching key across compiled or distributed wrappers.
    if key in state_dict:
        return key
    prefixed = f"_orig_mod.{key}"
    if prefixed in state_dict:
        return prefixed
    module_key = f"module.{key}"
    if module_key in state_dict:
        return module_key
    module_prefixed = f"module._orig_mod.{key}"
    if module_prefixed in state_dict:
        return module_prefixed
    return None


def _resize_vocab_state_dict(model, state_dict, ckpt_vocab_size=None):
    # Expand checkpoint token weights when the current model vocab is larger.
    if not state_dict:
        return None
    model_state = model.state_dict()
    target_key = _select_state_dict_key(model_state, "tok.weight")
    if target_key is None:
        return None
    target_weight = model_state.get(target_key)
    if target_weight is None:
        return None
    target_vocab = target_weight.shape[0]
    source_vocab = ckpt_vocab_size or _resolve_vocab_size(state_dict)
    if source_vocab is None:
        return None
    if source_vocab == target_vocab:
        return None
    if source_vocab > target_vocab:
        raise RuntimeError(
            "Checkpoint vocab size exceeds current model vocab size; "
            "refusing to truncate weights."
        )
    print(
        f"Expanding vocab weights from {source_vocab} to {target_vocab} tokens "
        "with fresh initialization for new rows."
    )
    device = target_weight.device
    for key in ("tok.weight", "lm.weight"):
        if key not in state_dict:
            continue
        base_key = _select_state_dict_key(model_state, key)
        if base_key is None:
            continue
        base = model_state[base_key].detach().clone()
        if state_dict[key].shape[1] != base.shape[1]:
            raise RuntimeError(f"Embedding width mismatch for {key}.")
        base[:source_vocab].copy_(state_dict[key].to(device))
        state_dict[key] = base
    return source_vocab


def resize_vocab_state_dict(model, state_dict, ckpt_vocab_size=None):
    return _resize_vocab_state_dict(model, state_dict, ckpt_vocab_size=ckpt_vocab_size)


def _load_into_compiled_module(model, state_dict):
    # Load state into the original module when torch.compile wraps the model.
    if not hasattr(model, "_orig_mod"):
        return False
    try:
        model._orig_mod.load_state_dict(state_dict)
    except Exception:
        return False
    return True


def _prefix_state_dict(state_dict, prefix):
    # Prefix keys to match wrapper-specific state dict formats.
    if not state_dict:
        return state_dict
    return {f"{prefix}{key}": value for key, value in state_dict.items()}


def _load_into_compiled_wrapper(model, state_dict):
    # Load state into a compiled wrapper when only prefixed keys are accepted.
    if not state_dict:
        return False
    try:
        model.load_state_dict(_prefix_state_dict(state_dict, "_orig_mod."))
    except Exception:
        return False
    return True


def load_model_state_dict(model, state_dict):
    # Load model state with compiled/legacy remap fallbacks for older checkpoints.
    try:
        model.load_state_dict(state_dict)
        return None
    except Exception:
        if _load_into_compiled_module(model, state_dict):
            return "compiled"
        if _load_into_compiled_wrapper(model, state_dict):
            return "compiled"
        if hasattr(model, "module"):
            try:
                return load_model_state_dict(model.module, state_dict)
            except Exception:
                pass
        remapped = _remap_legacy_state_dict(state_dict)
        if remapped is None:
            raise
        model.load_state_dict(remapped)
        return "legacy"


def _remap_legacy_state_dict(state_dict):
    # Legacy remap for old TransformerEncoder checkpoints; remove once obsolete.
    legacy_prefix = "blocks.layers."
    if not any(key.startswith(legacy_prefix) for key in state_dict):
        return None
    remapped = {}
    for key, value in state_dict.items():
        if not key.startswith(legacy_prefix):
            remapped[key] = value
            continue
        parts = key.split(".")
        if len(parts) < 4:
            continue
        layer_index = parts[2]
        suffix = ".".join(parts[3:])
        mapping = {
            "norm1.weight": f"blocks.{layer_index}.ln1.weight",
            "norm1.bias": f"blocks.{layer_index}.ln1.bias",
            "norm2.weight": f"blocks.{layer_index}.ln2.weight",
            "norm2.bias": f"blocks.{layer_index}.ln2.bias",
            "self_attn.in_proj_weight": f"blocks.{layer_index}.attn.qkv.weight",
            "self_attn.in_proj_bias": f"blocks.{layer_index}.attn.qkv.bias",
            "self_attn.out_proj.weight": f"blocks.{layer_index}.attn.proj.weight",
            "self_attn.out_proj.bias": f"blocks.{layer_index}.attn.proj.bias",
            "linear1.weight": f"blocks.{layer_index}.mlp.input.weight",
            "linear1.bias": f"blocks.{layer_index}.mlp.input.bias",
            "linear2.weight": f"blocks.{layer_index}.mlp.output.weight",
            "linear2.bias": f"blocks.{layer_index}.mlp.output.bias",
        }
        new_key = mapping.get(suffix)
        if new_key:
            remapped[new_key] = value
    return remapped


def _load_optimizer_state(optimizer, state_dict):
    # Load optimizer state and validate tensor shapes against parameters.
    if not state_dict:
        return False
    original_param_groups = copy.deepcopy(optimizer.param_groups)
    try:
        optimizer.load_state_dict(state_dict)
    except Exception:
        optimizer.state.clear()
        optimizer.param_groups = original_param_groups
        return False
    for param, state in optimizer.state.items():
        for value in state.values():
            if torch.is_tensor(value) and value.shape != param.shape:
                optimizer.state.clear()
                optimizer.param_groups = original_param_groups
                return False
    return True


class Checkpointer:
    """Save and restore training state to a local checkpoint directory.

    Stores model, optimizer, and scheduler state in a single file.
    Intended for periodic saves during long runs and auto-resume on startup.
    Keeps only the most recent checkpoint at `latest.pt`.
    """
    def __init__(self, directory, model, optimizer, scheduler, device=None):
        # Store references and resolve the checkpoint path.
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)
        self.path = self.directory / "latest.pt"
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def _snapshot_path(self, label):
        # Build a snapshot path for a retention label.
        return self.directory / f"latest_{label}.pt"

    def _write_checkpoint(self, path, ckpt):
        # Write checkpoint atomically via a temp file.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(ckpt, tmp_path)
        tmp_path.replace(path)

    def load_latest(self):
        # Load state from disk if present, otherwise start fresh.
        if not self.path.exists():
            return 0, 0, 0, 0, 0, None

        # Read checkpoint data onto the requested device.
        print(f"Loading checkpoint from {self.path}...")
        try:
            ckpt = torch.load(self.path, map_location=self.device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint {self.path}: {exc}") from exc

        # Restore model and optimizer state for resuming training.
        print("Checkpoint loaded. Restoring model and optimizer state...")
        model_state = ckpt.get("model")
        if model_state is None:
            raise RuntimeError(f"Checkpoint missing model state at {self.path}.")
        # Normalize checkpoint prefixes before loading or remapping.
        model_state = normalize_state_dict(model_state)
        _resize_vocab_state_dict(self.model, model_state, ckpt.get("vocab_size"))
        try:
            load_result = load_model_state_dict(self.model, model_state)
        except Exception as exc:
            raise RuntimeError(f"Failed to restore model state from {self.path}: {exc}") from exc
        if load_result == "compiled":
            print(f"Loaded checkpoint weights into compiled model from {self.path}.")
        elif load_result == "legacy":
            print(f"Loaded legacy checkpoint weights from {self.path}.")

        # Restore optimizer and scheduler state, falling back to fresh state on failure.
        optimizer_loaded = _load_optimizer_state(self.optimizer, ckpt.get("optimizer", {}))
        if not optimizer_loaded:
            print("Optimizer state mismatch; continuing with fresh optimizer state.")
        else:
            # Attempt to restore scheduler state but keep training if it mismatches.
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as exc:
                print(
                    "Scheduler state mismatch; continuing with a token-aligned schedule "
                    "based on resumed token counts."
                )

        # Recover counters with safe defaults (epoch stored as 1-based).
        saved_epoch = ckpt.get("epoch", 0)
        resume_step = ckpt.get("step", 0)
        global_step = ckpt.get("global_step", resume_step)

        sample_index = ckpt.get("sample_index", 0)
        total_tokens = ckpt.get("total_tokens", 0)
        resume_state = ckpt.get("resume_state")

        # Announce resume location for visibility.
        display_epoch = saved_epoch if saved_epoch > 0 else 1
        print(
            f"Resuming from {self.path} at epoch {display_epoch}, step {resume_step}, "
            f"sample index {sample_index}."
        )
        resume_epoch = max(saved_epoch - 1, 0)
        return resume_epoch, resume_step, global_step, sample_index, total_tokens, resume_state

    def save_latest(self, epoch, step, global_step, sample_index, total_tokens, resume_state=None):
        # Persist the latest training state in the current checkpoint format.
        start_time = time.time()
        vocab_size = getattr(getattr(self.model, "tok", None), "num_embeddings", None)
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": config.snapshot(),
            "epoch": epoch + 1,
            "step": step,
            "global_step": global_step,
            "sample_index": sample_index,
            "total_tokens": total_tokens,
            "resume_state": resume_state,
        }
        if vocab_size is not None:
            ckpt["vocab_size"] = vocab_size
        # Save the rolling latest checkpoint.
        self._write_checkpoint(self.path, ckpt)

        # Save periodic snapshot copies based on the interval schedule.
        now = time.time()
        for label, interval in SNAPSHOT_INTERVALS:
            snapshot_path = self._snapshot_path(label)
            if snapshot_path.exists():
                last_saved = snapshot_path.stat().st_mtime
                if now - last_saved < interval:
                    continue
            self._write_checkpoint(snapshot_path, ckpt)
        elapsed = time.time() - start_time
        print(
            f"Saved checkpoint to {self.path} at epoch {epoch + 1}, step {step} "
            f"({elapsed:.2f}s)."
        )
