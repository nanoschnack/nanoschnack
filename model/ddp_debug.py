import hashlib

import torch


def build_rng_tensor(device):
    """Build a compact RNG fingerprint tensor for cross-rank parity checks."""
    cpu_hash = hashlib.sha1(torch.random.get_rng_state().numpy().tobytes()).hexdigest()
    cpu_value = int(cpu_hash[:16], 16)
    cuda_value = 0
    if device.type == "cuda":
        cuda_hash = hashlib.sha1(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
        cuda_value = int(cuda_hash[:16], 16)
    return torch.tensor([cpu_value, cuda_value], dtype=torch.uint64, device=device)


def log_ddp_debug(gathered_losses, stats_gathered, rng_gathered, is_master):
    """Emit per-rank debug stats for DDP loss parity checks."""
    losses = " ".join(f"{idx}={value.item():.2f}" for idx, value in enumerate(gathered_losses))
    print(f"ddp-losses: {losses}", flush=True)
    stats = " ".join(
        f"{idx}=t{value[0].item()} s{value[1].item()}"
        for idx, value in enumerate(stats_gathered)
    )
    print(f"ddp-batch: {stats}", flush=True)
    rngs = " ".join(
        f"{idx}=cpu:{int(value[0].item()):016x} cuda:{int(value[1].item()):016x}"
        for idx, value in enumerate(rng_gathered)
    )
    print(f"ddp-rng: {rngs}", flush=True)
