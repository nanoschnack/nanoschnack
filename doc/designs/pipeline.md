# Data Pipeline Overview

This document captures the current training data pipeline and threading setup.
It reflects the behavior in `model/loader.py` and `model/training.py`.

## Pipeline Stages

1) Dataset specs
- Parse `DATASET_SPECS` into HF or TXT specs.
- Resolve per-spec metadata (split, text_key, cache).

2) Resume offsets
- Determine resume row offsets from checkpoints.
- Skip rows or shard-local offsets before tokenization.
- Cap remaining rows for exhausted datasets.

3) Streaming datasets
- Load datasets in streaming mode when available.
- Optionally interleave sources (token-count-aware interleave for resume).

4) Tokenization and packing
- Add `row_count` to preserve resume accounting.
- Tokenize with `tokenizer.encode_batch`.
- Pack into fixed-length blocks (`CONTEXT_LEN`).
- Drop empty packed samples and unused columns.

5) Batching
- Wrap the packed stream in a `DataLoader`.
- Yield batches to the training loop.

## Threading and Parallelism

The pipeline supports three distinct concurrency layers:

1) DataLoader workers (processes)
- Config: `DATA_LOADER_WORKERS` (default 0).
- When > 0, the DataLoader spawns worker processes to parallelize iteration.
- This is the primary multiprocessing path for the pipeline.

2) Tokenizer threads (within a process)
- Config: `TOKENIZER_WORKERS` (default 4).
- A `ThreadPoolExecutor` is used to split a batch into chunks and
  parallelize `tokenizer.encode_batch`.
- Intended for `DATA_LOADER_WORKERS=0`.

3) Prefetch thread (main process)
- Config: `PREFETCH_BATCHES` (default 2).
- A background thread prefetches iterator items into a small queue.
- This overlaps iterator work with the training loop.

## Safety Rule

The training loop enforces:

- `DATA_LOADER_WORKERS > 0` and `TOKENIZER_WORKERS > 0` is not allowed.

Rationale: nested process + thread pools with HF streaming can lead to
oversubscription or stalls. Choose one:

- `DATA_LOADER_WORKERS > 0`, `TOKENIZER_WORKERS = 0`
- `DATA_LOADER_WORKERS = 0`, `TOKENIZER_WORKERS > 0`

Prefetching (`PREFETCH_BATCHES`) can be used in both cases.

## Iterator Flow (Sequence)

```
dataset specs
  -> load_dataset_from_spec (streaming)
    -> resume skip / cap
      -> (optional) interleave datasets
        -> map: add row_count
          -> map: tokenize batch
            -> map: pack tokens
              -> filter: drop empty
                -> DataLoader
                  -> prefetch thread (optional)
                    -> training loop
```

## Known Bottlenecks

- Shard boundaries can cause multi-second stalls if a new shard is downloaded
  or decoded.
- Tokenization can be a bursty CPU cost for large batch sizes.

## Failure Modes and Mitigations

1) Multiprocessing pickling errors
- Symptom: `AttributeError: Can't pickle ...` when workers start.
- Cause: local functions/closures used in dataset map/filter or generators.
- Mitigation: keep map/filter callbacks at module scope and avoid lambdas.

2) Nested process + thread contention
- Symptom: worker stalls or long first-batch waits.
- Cause: `DATA_LOADER_WORKERS > 0` with tokenizer thread pool enabled.
- Mitigation: choose one form of parallelism and rely on prefetching.

3) Shard boundary stalls
- Symptom: periodic multi-second IO waits (often every few macro steps).
- Cause: shard download, cache miss, or parquet decode.
- Mitigation: increase prefetch, warm the cache, or avoid aggressive cleanup.

## Places to Inspect

- `model/training.py`: dataset setup and `DataLoader` usage.
- `model/loader.py`: dataset transforms, tokenization, packing, prefetch.
