import argparse
import os
import time

try:
    from model import setup_paths
except ModuleNotFoundError:
    from __init__ import setup_paths

_, data_dir, _ = setup_paths()

# Silence tokenizers fork warnings in multiprocessing.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import config
from loader import (
    build_interleaved_dataset,
    build_packed_dataset,
    format_dataset_for_torch,
    load_dataset_from_spec,
    parse_dataset_specs,
    resolve_dataloader_workers,
    shuffle_dataset,
)
from tokenizer import load_tokenizer
from torch.utils.data import DataLoader
import torch


def _count_tokens(attention_mask):
    # Normalize token counting across tensor or list batches.
    if torch.is_tensor(attention_mask):
        return int(attention_mask.sum().item())
    if attention_mask and isinstance(attention_mask[0], (list, tuple)):
        return int(sum(sum(row) for row in attention_mask))
    if attention_mask and torch.is_tensor(attention_mask[0]):
        return int(sum(int(mask.sum().item()) for mask in attention_mask))
    return int(sum(attention_mask))


def _count_samples(attention_mask):
    # Determine batch size from tensor or list batches.
    if torch.is_tensor(attention_mask):
        return int(attention_mask.size(0))
    return int(len(attention_mask))


def main():
    parser = argparse.ArgumentParser(description="Benchmark loader pipeline throughput.")
    parser.add_argument("--batches", type=int, default=200, help="number of batches to measure")
    parser.add_argument("--warmup", type=int, default=10, help="number of warmup batches to skip")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="batch size for DataLoader")
    parser.add_argument(
        "--pack-batch-size",
        type=int,
        default=config.PACK_BATCH_SIZE,
        help="batch size for packing token blocks",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=config.SHUFFLE_BUFFER,
        help="shuffle buffer size for streaming datasets",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.DATA_LOADER_WORKERS,
        help="DataLoader worker count",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=config.DATA_LOADER_PREFETCH_FACTOR,
        help="DataLoader prefetch factor per worker",
    )
    parser.add_argument(
        "--persistent-workers",
        type=int,
        choices=[0, 1],
        default=int(config.DATA_LOADER_PERSISTENT),
        help="keep DataLoader workers alive between iterations (0/1)",
    )
    parser.add_argument(
        "--pin-memory",
        type=int,
        choices=[0, 1],
        default=int(config.DATA_LOADER_PIN_MEMORY),
        help="enable pinned memory for faster host-to-device copies (0/1)",
    )
    parser.add_argument(
        "--dataset-specs",
        type=str,
        default=config.DATASET_SPECS,
        help="comma-separated dataset specs (overrides DATASET_SPECS)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(data_dir),
        help="cache directory for dataset metadata (empty disables caching)",
    )
    args = parser.parse_args()

    # Build the streaming dataset pipeline.
    cache_dir = args.cache_dir or None
    tokenizer = load_tokenizer()
    dataset_specs = parse_dataset_specs(args.dataset_specs)
    packed_datasets = []
    for dataset_index, spec in enumerate(dataset_specs):
        raw_streaming = load_dataset_from_spec(spec, cache_dir=cache_dir, streaming=True)
        packed = build_packed_dataset(
            raw_streaming,
            tokenizer=tokenizer,
            block_size=config.CONTEXT_LEN,
            text_key=spec["text_key"],
            pack_batch_size=args.pack_batch_size,
            source_id=dataset_index,
        )
        packed_datasets.append(packed)
    base_dataset = build_interleaved_dataset(packed_datasets, seed=42)
    dataset_epoch = shuffle_dataset(base_dataset, args.shuffle_buffer, 42)
    dataset_epoch = format_dataset_for_torch(dataset_epoch)
    worker_count = resolve_dataloader_workers(dataset_epoch, args.workers)
    if worker_count != args.workers:
        print(
            f"Benchmark: forcing workers=0 (requested {args.workers}) "
            "for the fast dataset pipeline.",
            flush=True,
        )
    # Apply DataLoader overlap tuning when workers are enabled.
    if worker_count > 0:
        loader = DataLoader(
            dataset_epoch,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=worker_count,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=bool(args.persistent_workers),
            pin_memory=bool(args.pin_memory),
        )
    else:
        loader = DataLoader(
            dataset_epoch,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

    # Warm up to get past initial streaming latency.
    iterator = iter(loader)
    for _ in range(args.warmup):
        next(iterator)

    # Measure tokenization throughput over the requested batches.
    token_count = 0
    sample_count = 0
    start = time.time()
    for _ in range(args.batches):
        batch = next(iterator)
        attention_mask = batch["attention_mask"]
        token_count += _count_tokens(attention_mask)
        sample_count += _count_samples(attention_mask)
    elapsed = time.time() - start
    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0.0
    samples_per_sec = sample_count / elapsed if elapsed > 0 else 0.0
    print(
        f"tokens={token_count} samples={sample_count} "
        f"duration={elapsed:.2f}s "
        f"tokens/s={tokens_per_sec:.1f} samples/s={samples_per_sec:.2f}"
    )


if __name__ == "__main__":
    main()
