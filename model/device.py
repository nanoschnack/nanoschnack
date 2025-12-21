import torch


def pick_device():
    # Prefer MPS, then CUDA, then CPU for inference and training.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_info(device):
    # Collect human-friendly device details for logging.
    info = {"device": str(device), "device_type": device.type}

    if device.type == "cuda":
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "cuda_device": torch.cuda.get_device_name(idx),
                "cuda_capability": f"{props.major}.{props.minor}",
                "cuda_total_memory_bytes": props.total_memory,
            }
        )
    elif device.type == "mps":
        info.update(
            {
                "mps_available": torch.backends.mps.is_available(),
                "mps_built": torch.backends.mps.is_built(),
                "mps_device": "apple.mps",
            }
        )
    else:
        info["cpu_threads"] = torch.get_num_threads()

    return info


def print_device_info(info):
    # Print the device information in a shared format.
    print(f"Using device: {info['device']}")
    if info["device_type"] == "cuda":
        print(f"cuda_device={info['cuda_device']}")
        print(f"cuda_capability={info['cuda_capability']}")
        print(f"cuda_total_memory_bytes={info['cuda_total_memory_bytes']}")
    elif info["device_type"] == "mps":
        print(f"mps_device={info['mps_device']}")
        print(f"mps_available={info['mps_available']} mps_built={info['mps_built']}")
    else:
        print(f"cpu_threads={info['cpu_threads']}")
