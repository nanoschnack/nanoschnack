"""Helpers for debug logging and diagnostics."""

import faulthandler
import os
import tempfile

from torch.utils.data import get_worker_info


def build_dataloader_worker_init(log_prefix="nanoschnack_dataloader_worker"):
    # Keep worker stderr logs open for the life of each process.
    log_files = []

    # Capture DataLoader worker crashes to per-worker logs.
    def _worker_init_fn(_):
        info = get_worker_info()
        if info is None:
            return

        log_path = os.path.join(tempfile.gettempdir(), f"{log_prefix}_{info.id}.log")
        log_file = open(log_path, "w", buffering=1)
        log_files.append(log_file)
        faulthandler.enable(file=log_file)
        os.dup2(log_file.fileno(), 2)

    _worker_init_fn.log_files = log_files
    return _worker_init_fn
