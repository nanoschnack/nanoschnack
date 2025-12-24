from pathlib import Path
import time

import torch

import config

def apply_checkpoint_config(ckpt_config):
    # Apply checkpoint hyperparameters to global config.
    if not ckpt_config:
        return
    for name in ("CONTEXT_LEN", "EMBED_SIZE", "NUM_LAYERS", "NUM_HEADS", "HIDDEN_SIZE"):
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


def load_model_state_dict(model, state_dict):
    # Load model state with legacy remap fallback for older checkpoints.
    try:
        model.load_state_dict(state_dict)
        return False
    except Exception:
        remapped = _remap_legacy_state_dict(state_dict)
        if remapped is None:
            raise
        model.load_state_dict(remapped)
        return True


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
    try:
        optimizer.load_state_dict(state_dict)
    except Exception as exc:
        print(f"Failed to load optimizer state: {exc}.")
        return False
    for param, state in optimizer.state.items():
        for value in state.values():
            if torch.is_tensor(value) and value.shape != param.shape:
                print("Optimizer state shape mismatch; resetting optimizer state.")
                optimizer.state.clear()
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

    def load_latest(self):
        # Load state from disk if present, otherwise start fresh.
        if not self.path.exists():
            return 0, 0, 0, 0, 0

        # Read checkpoint data onto the requested device.
        print(f"Loading checkpoint from {self.path}...")
        try:
            ckpt = torch.load(self.path, map_location=self.device)
        except Exception as exc:
            print(f"Failed to load checkpoint {self.path}: {exc}. Starting fresh.")
            return 0, 0, 0, 0, 0

        # Restore model and optimizer state for resuming training.
        print("Checkpoint loaded. Restoring model and optimizer state...")
        model_state = ckpt.get("model")
        if model_state is None:
            print(f"Checkpoint missing model state at {self.path}. Starting fresh.")
            return 0, 0, 0, 0, 0
        # Normalize checkpoint prefixes before loading or remapping.
        model_state = normalize_state_dict(model_state)
        try:
            remapped = load_model_state_dict(self.model, model_state)
        except Exception as exc:
            print(f"Failed to restore model state from {self.path}: {exc}. Starting fresh.")
            return 0, 0, 0, 0, 0
        if remapped:
            print(f"Loaded legacy checkpoint weights from {self.path}.")

        # Restore optimizer and scheduler state, falling back to fresh state on failure.
        optimizer_ok = _load_optimizer_state(self.optimizer, ckpt.get("optimizer", {}))
        if optimizer_ok:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as exc:
                print(f"Failed to restore scheduler from {self.path}: {exc}. Resetting scheduler.")
        else:
            print("Skipping scheduler restore because optimizer state was reset.")

        # Recover counters with safe defaults (epoch stored as 1-based).
        saved_epoch = ckpt.get("epoch", 0)
        resume_step = ckpt.get("step", 0)
        global_step = ckpt.get("global_step", resume_step)

        sample_index = ckpt.get("sample_index", 0)
        total_tokens = ckpt.get("total_tokens", 0)

        # Announce resume location for visibility.
        display_epoch = saved_epoch if saved_epoch > 0 else 1
        print(
            f"Resuming from {self.path} at epoch {display_epoch}, step {resume_step}, "
            f"sample index {sample_index}."
        )
        resume_epoch = max(saved_epoch - 1, 0)
        return resume_epoch, resume_step, global_step, sample_index, total_tokens

    def save_latest(self, epoch, step, global_step, sample_index, total_tokens):
        # Persist the latest training state in the current checkpoint format.
        start_time = time.time()
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
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save(ckpt, tmp_path)
        tmp_path.replace(self.path)
        elapsed = time.time() - start_time
        print(
            f"Saved checkpoint to {self.path} at epoch {epoch + 1}, step {step} "
            f"({elapsed:.2f}s)."
        )
