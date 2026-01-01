# HF Shard-Local Loading with Prefetch

## Summary

This design adds a Hugging Face dataset loader that downloads parquet shards
locally, one shard at a time, and prefetches the next shard in the background.
It targets fast resume and skip behavior by using non-streaming per-shard loads
while avoiding upfront download of the entire split.

## Goals

- Download only the current shard and prefetch the next shard.
- Keep resume/skip fast by reading from local shards.
- Preserve existing dataset spec parsing and resume logic.
- Scope the behavior to HF datasets only.
- Keep the API surface similar to the current loader.

## Non-Goals

- Full-dataset download or materialization.
- Rewriting Hugging Face internals.
- Transparent support for non-parquet formats.

## Current Behavior

- HF datasets use `streaming=True`, which fetches byte ranges over HTTP.
- Resume and skip are linear and can be slow even with range caching.
- Shard selection is handled by `data_files` and resume mapping.

## Proposed Design

### New Loader: `hf_load_dataset_sharded`

Add a helper in `model/loader.py` that:

- Enumerates parquet shard filenames (using existing HF parquet index helpers).
- Downloads shard files locally via `hf_hub_download`.
- Loads one shard at a time via `load_dataset("parquet", data_files=..., streaming=False)`.
- Prefetches the next shard in a background thread.

### Cache Layout

Shard cache lives under:

```
data/hf_shards/<repo_id>/<split>/<filename>
```

This keeps shards grouped by source and avoids name collisions.

### Resume Integration

The existing resume logic computes:

```
data_files, in_shard_offset, shard_label
```

The new loader should:

1. Start from the shard returned by `resolve_resume_plan`.
2. Apply `skip(in_shard_offset)` on the local shard dataset.
3. Continue with subsequent shards in order.

### Prefetcher Behavior

- Maintain a one-shard lookahead per dataset.
- Start downloading shard N+1 as soon as shard N is opened.
- Wait for the prefetched shard before switching to it.

### Cleanup Policy

Add a config option in `model/config.py` to control shard cleanup, for example:

```
HF_SHARD_CACHE_CLEANUP = _env_str("HF_SHARD_CACHE_CLEANUP", "auto")
```

Proposed values:

- `auto`: delete shards older than the current and next shard for each dataset.
- `keep`: never remove cached shards.

Cleanup should only touch the `data/hf_shards/<repo_id>/<split>` subtree.

### Integration Points

- Replace `load_dataset_source(...)` for HF specs with `hf_load_dataset_sharded(...)`.
- Preserve the `txt:` path behavior unchanged.
- Keep existing `data_files` usage for resume mapping.

## Implementation Sketch

```
def hf_load_dataset_sharded(...):
    shards = resolve_shards(...)
    prefetch = ShardPrefetcher(shards, cache_root)
    prefetch.start()

    for shard in shards:
        path = prefetch.wait_for(shard)
        dataset = load_dataset("parquet", data_files=path, streaming=False)
        for row in dataset:
            yield row
```

`ShardPrefetcher`:
- downloads shard N+1 via `hf_hub_download`,
- stores under `data/hf_shards/<repo_id>/<split>/`,
- optionally deletes old shards if cleanup is enabled.

## Risks and Tradeoffs

- Non-streaming per shard uses more memory than pure streaming.
- Prefetch thread needs robust error handling and retries.
- Cleanup must avoid deleting shards still in use.

## Alternatives Considered

- Continue streaming with range cache: resume remains slow.
- Full non-streaming download: violates the "no full dataset download" goal.
