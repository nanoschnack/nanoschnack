"""Learning-rate scheduling helpers."""

import math

import torch


def build_warmup_cosine(optimizer, total_steps, warmup_pct):
    """Create a warmup + cosine schedule for the given optimizer.

    The warmup ramps linearly to the base LR, then cosine anneals for the
    remaining steps.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if not (0.0 <= warmup_pct <= 1.0):
        raise ValueError("warmup_pct must be between 0 and 1.")

    warmup_steps = int(math.ceil(total_steps * warmup_pct))
    warmup_steps = min(warmup_steps, total_steps)
    cosine_steps = max(total_steps - warmup_steps, 1)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.0,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
