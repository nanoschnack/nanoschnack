import torch


def pick_device():
    # Prefer MPS, then CUDA, then CPU for inference and training.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
