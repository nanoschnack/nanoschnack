from pathlib import Path
import time

import torch

import config

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
        try:
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as exc:
            print(f"Failed to restore checkpoint state from {self.path}: {exc}. Starting fresh.")
            return 0, 0, 0, 0, 0

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
        # Persist the latest training state to disk.
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
