import torch


def _try_batch(model, vocab_size, seq_len, batch_size, device):
    # Run a single forward/backward step to probe memory usage.
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
    )
    loss.backward()


def find_max_batch_size(
    model,
    vocab_size,
    seq_len,
    device,
    start=4,
    max_batch=512,
    safety_factor=0.9,
):
    """Estimate the maximum batch size that fits in memory.

    Uses exponential growth to find an upper bound, then binary search.
    Returns a safe batch size reduced by the provided safety factor.
    Designed for CUDA; returns None if the start size fails.
    """
    if device.type != "cuda":
        return None

    # Grow batch size exponentially until an OOM occurs.
    batch = start
    last_good = None
    while batch <= max_batch:
        try:
            model.zero_grad(set_to_none=True)
            _try_batch(model, vocab_size, seq_len, batch, device)
            last_good = batch
            batch *= 2
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            break

    if last_good is None:
        return None


    # Binary search between last_good and the failing batch size.
    lo, hi = last_good, batch
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        try:
            model.zero_grad(set_to_none=True)
            _try_batch(model, vocab_size, seq_len, mid, device)
            lo = mid
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            hi = mid

    return max(1, int(lo * safety_factor))
