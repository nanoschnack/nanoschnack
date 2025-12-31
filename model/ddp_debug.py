import hashlib

import torch


def build_rng_label(device):
    """Build a compact RNG fingerprint for cross-rank parity checks."""
    cpu_hash = hashlib.sha1(torch.random.get_rng_state().numpy().tobytes()).hexdigest()[:8]
    cuda_hash = None
    if device.type == "cuda":
        cuda_hash = hashlib.sha1(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()[:8]
    return f"cpu:{cpu_hash}" + (f" cuda:{cuda_hash}" if cuda_hash else "")


def log_ddp_debug(gathered_losses, stats_gathered, rng_gathered, is_master):
    """Emit per-rank debug stats for DDP loss parity checks."""
    losses = " ".join(f"{idx}={value.item():.2f}" for idx, value in enumerate(gathered_losses))
    print(f"ddp-losses: {losses}", flush=True)
    stats = " ".join(
        f"{idx}=t{value[0].item()} s{value[1].item()}"
        for idx, value in enumerate(stats_gathered)
    )
    print(f"ddp-batch: {stats}", flush=True)
    rngs = " ".join(f"{idx}={label}" for idx, label in enumerate(rng_gathered))
    print(f"ddp-rng: {rngs}", flush=True)
