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
    base_lr = optimizer.param_groups[0]["lr"]
    min_lr = base_lr * 0.1

    if warmup_steps == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=min_lr,
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        # LinearLR requires a strictly positive start factor.
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=min_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def build_warmup_cosine_tokens(optimizer, total_tokens, warmup_pct):
    """Create a warmup + cosine schedule driven by total token count."""
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive.")
    if not (0.0 <= warmup_pct <= 1.0):
        raise ValueError("warmup_pct must be between 0 and 1.")

    warmup_tokens = int(math.ceil(total_tokens * warmup_pct))
    warmup_tokens = min(warmup_tokens, total_tokens)
    base_lr = optimizer.param_groups[0]["lr"]
    min_lr = base_lr * 0.1
    start_factor = 1e-6

    def lr_lambda(token_count):
        if token_count <= 0:
            return start_factor
        if warmup_tokens > 0 and token_count < warmup_tokens:
            return max(start_factor, token_count / warmup_tokens)
        progress = (token_count - warmup_tokens) / max(total_tokens - warmup_tokens, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        min_ratio = min_lr / base_lr
        return min_ratio + cosine * (1 - min_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
