from pathlib import Path

import torch


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
            return 0, 0, 0

        # Read checkpoint data onto the requested device.
        ckpt = torch.load(self.path, map_location=self.device)

        # Restore model and optimizer state for resuming training.
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])

        # Recover counters with safe defaults.
        resume_epoch = ckpt.get("epoch", 0)
        resume_step = ckpt.get("step", 0)
        global_step = ckpt.get("global_step", resume_step)

        # Announce resume location for visibility.
        print(f"Resuming from {self.path} at epoch {resume_epoch}, step {resume_step}.")
        return resume_epoch, resume_step, global_step

    def save_latest(self, epoch, step, global_step):
        # Persist the latest training state to disk.
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
        }
        torch.save(ckpt, self.path)
        print(f"Saved checkpoint to {self.path} at epoch {epoch}, step {step}.")
