# Streaming Resume by Per-Source Row Offsets

## Summary

This design replaces the current resume mechanism (packed-sample index) with a
per-source row offset resume that works for HF parquet datasets and local text
files. The goal is to avoid repeating already-seen rows while allowing future
order changes. The behavior stays transparent to the user and keeps existing
training flow, packing, and interleaving intact.

## Goals

- Resume without repeating already-seen rows.
- Keep streaming datasets and on-demand shard access.
- Stay slim: minimal new logic, no new user-facing knobs.
- Work for HF hub parquet datasets and local text files.
- Preserve backward compatibility with existing checkpoints.

## Non-Goals

- Exact deterministic resume of packed samples.
- O(1) resume for streaming iterables.
- Rewriting or forking Hugging Face dataset internals.

## Current Behavior

- Training stores `sample_index` in the checkpoint.
- On resume, `dataset_epoch = dataset_epoch.skip(sample_index)`.
- `sample_index` counts packed samples, not raw rows.
- For streaming datasets, skip is linear and grows over time.

## Proposed Resume State

Store per-source row offsets instead of relying on packed-sample position.
The state is saved in the checkpoint alongside existing fields.

Example checkpoint payload:

```
resume_state:
  datasets:
    - spec: hf:arnomatic/german-wikipedia-clean-no-lists:train:text
      shard: train-00087-of-01024.parquet
      row_offset: 128400
    - spec: hf:PatrickHaller/fineweb-2-de-1B:train:text
      shard: train-00012-of-00480.parquet
      row_offset: 51200
```

For local text files, the shard is the file path:

```
resume_state:
  datasets:
    - spec: txt:/data/corpus/shard_001.txt:text
      shard: /data/corpus/shard_001.txt
      row_offset: 910250
```

## How It Works

### Row Counting

During training, increment per-source row counters before packing. The counters
track raw input rows consumed per dataset spec.

### Shard Mapping

At resume time, map each row offset to a shard and an in-shard row offset.
This requires a shard index for HF parquet datasets and for local text files.

- HF parquet: list parquet shard files and read their footer metadata
  (`num_rows`) to build prefix sums.
- Local text: compute row counts per file once (line counts).

### Resume Dataset Construction

For each dataset spec:

1. Use the shard index to locate the starting shard and the row offset inside
   that shard.
2. Rebuild the streaming dataset with `data_files` filtered to the shard and
   later shards.
3. Apply `skip(row_offset)` within the first shard only.

Then continue with existing packing, interleaving, and shuffling.

### Packing Behavior

Packing starts from an empty buffer after resume. This may drop a partial block
at the resume boundary but guarantees no repeated rows. Future order changes are
accepted.

## Transparency

Training logs a single resume line per dataset spec:

```
resume: spec=... sample_row=... shard=... offset=...
```

No new CLI flags or config values are added.

## Backward Compatibility

If `resume_state` is missing (older checkpoints), fall back to the current
`sample_index`-based skip. This preserves existing checkpoint behavior.

## Storage

Shard indices are cached under `data/` as small JSON files keyed by
`repo_id/split/revision` or by local file list signature.

## Risks and Tradeoffs

- Future order changes occur after resume due to packing reset.
- Resume cost is bounded by one shard, not O(1).
- Requires reading parquet footers on first index build.

## Alternatives Considered

- Non-streaming datasets for O(1) resume: rejected due to disk and materialize
  costs.
- Full packer-state checkpointing: rejected as more complex than needed.
