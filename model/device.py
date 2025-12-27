import torch


def pick_device(ddp_local_rank=None):
    # Prefer MPS, then CUDA, then CPU for inference and training.
    if ddp_local_rank is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP local rank requires CUDA, but CUDA is not available.")
        torch.cuda.set_device(ddp_local_rank)
        return torch.device("cuda", ddp_local_rank)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_info(device):
    # Collect human-friendly device details for logging.
    info = {"device": str(device), "device_type": device.type}

    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "cuda_device_count": torch.cuda.device_count(),
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
    lines = ["Device:", f"  device={info['device']}"]
    if info["device_type"] == "cuda":
        lines.append(f"  cuda_device_count={info['cuda_device_count']}")
        lines.append(f"  cuda_device={info['cuda_device']}")
        lines.append(f"  cuda_capability={info['cuda_capability']}")
        lines.append(f"  cuda_total_memory_bytes={info['cuda_total_memory_bytes']}")
    elif info["device_type"] == "mps":
        lines.append(f"  mps_device={info['mps_device']}")
        lines.append(f"  mps_available={info['mps_available']} mps_built={info['mps_built']}")
    else:
        lines.append(f"  cpu_threads={info['cpu_threads']}")
    print("\n".join(lines))


def print_ddp_info(ddp_rank, ddp_world_size, ddp_local_rank, ddp_master_addr, ddp_master_port):
    # Print distributed environment details for debugging.
    lines = ["DDP:", f"  rank={ddp_rank}", f"  world_size={ddp_world_size}",
             f"  local_rank={ddp_local_rank}",
             f"  master_addr={ddp_master_addr}",
             f"  master_port={ddp_master_port}"]
    print("\n".join(lines))


def print_sdpa_info():
    # Report SDPA kernel availability for attention debugging.
    print("Performance:")
    print(f"  Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"  Mem-efficient SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"  Math SDP enabled: {torch.backends.cuda.math_sdp_enabled()}")
    print("  SDPA kernel selection: set TORCH_LOGS=attention")
