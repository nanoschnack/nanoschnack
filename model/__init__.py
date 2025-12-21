"""Shared helpers for model scripts and notebooks."""

from pathlib import Path
import sys


def setup_paths(base_dir=None):
    """Resolve model, data, and checkpoint directories with a stable layout.

    Accepts an optional base directory for scripts; notebooks can omit it.
    Ensures the model directory is on sys.path for local imports.
    Returns (model_dir, data_dir, checkpoint_dir).
    """
    # Determine where the model directory lives based on the caller context.
    if base_dir is None:
        cwd = Path.cwd()
        model_dir = cwd / "model" if (cwd / "model").is_dir() else cwd
    else:
        model_dir = Path(base_dir)
    model_dir = model_dir.resolve()

    # Add the model directory to the import path for local modules.
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    # Map common data and checkpoint directories relative to the repo root.
    data_dir = model_dir.parent / "data"
    checkpoint_dir = model_dir.parent / "checkpoints"
    return model_dir, data_dir, checkpoint_dir
