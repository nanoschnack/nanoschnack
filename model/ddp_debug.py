import hashlib

import torch
import torch.distributed as dist


def log_ddp_debug(world_size, micro_loss, micro_tokens, micro_samples, device, is_master):
    """Collect and emit per-rank debug stats for DDP loss parity checks."""
    # Gather per-rank loss values for parity checks.
    loss_tensor = torch.tensor([micro_loss], device=device)
    gathered = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, loss_tensor)
    if is_master:
        losses = " ".join(f"{idx}={value.item():.2f}" for idx, value in enumerate(gathered))
        print(f"ddp-losses: {losses}", flush=True)

    # Gather per-rank token/sample counts to validate batch parity.
    stats_tensor = torch.tensor([micro_tokens, micro_samples], dtype=torch.long, device=device)
    stats_gathered = [torch.zeros_like(stats_tensor) for _ in range(world_size)]
    dist.all_gather(stats_gathered, stats_tensor)

    # Hash RNG states to spot divergence between ranks.
    cpu_hash = hashlib.sha1(torch.random.get_rng_state().numpy().tobytes()).hexdigest()[:8]
    cuda_hash = None
    if device.type == "cuda":
        cuda_hash = hashlib.sha1(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()[:8]
    rng_label = f"cpu:{cpu_hash}" + (f" cuda:{cuda_hash}" if cuda_hash else "")
    rng_gathered = [None for _ in range(world_size)]
    dist.all_gather_object(rng_gathered, rng_label)
    if is_master:
        stats = " ".join(
            f"{idx}=t{value[0].item()} s{value[1].item()}"
            for idx, value in enumerate(stats_gathered)
        )
        print(f"ddp-batch: {stats}", flush=True)
        rngs = " ".join(f"{idx}={label}" for idx, label in enumerate(rng_gathered))
        print(f"ddp-rng: {rngs}", flush=True)
