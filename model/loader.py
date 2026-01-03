"""Helpers for building streaming datasets with packing."""

import json
import math
import os
import random
import shutil
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

import torch
from datasets import IterableDataset, interleave_datasets, load_dataset

import time

import config

from tokenizer import DATASET_EOS_TOKEN

_TOKENIZER_POOL = None
_TOKENIZER_POOL_WORKERS = None


def _get_tokenizer_pool():
    # Lazily create a tokenizer worker pool when enabled.
    global _TOKENIZER_POOL, _TOKENIZER_POOL_WORKERS
    workers = int(config.TOKENIZER_WORKERS)
    if workers <= 1:
        return None
    if _TOKENIZER_POOL is None or _TOKENIZER_POOL_WORKERS != workers:
        if _TOKENIZER_POOL is not None:
            _TOKENIZER_POOL.shutdown(wait=False)
        _TOKENIZER_POOL = ThreadPoolExecutor(max_workers=workers)
        _TOKENIZER_POOL_WORKERS = workers
    return _TOKENIZER_POOL


def load_dataset_source(repo_id, split="train", data_files=None, cache_dir=None, streaming=True, name=None):
    # Build a dataset from hub or local parquet files.
    kwargs = {"split": split, "streaming": streaming, "cache_dir": cache_dir}
    if name:
        kwargs["name"] = name
    if data_files is not None:
        if repo_id:
            return load_dataset(
                repo_id,
                data_files=data_files,
                **kwargs,
            )
        return load_dataset(
            "parquet",
            data_files=data_files,
            **kwargs,
        )
    return load_dataset(
        repo_id,
        **kwargs,
    )


def _hf_shard_cache_root(cache_dir, repo_id, split):
    # Resolve the local cache root for HF parquet shards.
    root = Path(cache_dir) if cache_dir is not None else Path.cwd() / "data"
    return root / "hf_shards" / repo_id / split

def _hf_local_shard_path(cache_dir, repo_id, split, rel_path):
    # Map a repo-relative shard path to a local cached filename.
    filename = Path(rel_path).name
    return _hf_shard_cache_root(cache_dir, repo_id, split) / filename

def _cleanup_shard_cache(cache_root, keep_paths):
    # Remove cached shard files not in the keep list.
    keep = {Path(path).resolve() for path in keep_paths if path is not None}
    if not cache_root.exists():
        return

    # Avoid deleting active downloads stored as temp files.
    for path in cache_root.rglob("*"):
        if path.is_file() and path.suffix == ".tmp":
            continue
        if path.is_file() and path.resolve() not in keep:
            path.unlink()
    for path in sorted(cache_root.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                continue

def _ensure_local_shard(repo_id, rel_path, cache_dir, split, local_files_only=False):
    # Download a shard to the local cache if missing.
    from huggingface_hub import hf_hub_download

    cache_root = _hf_shard_cache_root(cache_dir, repo_id, split)
    cache_root.mkdir(parents=True, exist_ok=True)
    local_path = _hf_local_shard_path(cache_dir, repo_id, split, rel_path)
    if local_path.exists():
        return local_path

    # Use the HF cache as the source, then copy into the local shard cache.
    downloaded = hf_hub_download(
        repo_id,
        filename=rel_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        repo_type="dataset",
    )
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    shutil.copy2(downloaded, tmp_path)
    tmp_path.replace(local_path)
    return local_path


class ShardPrefetcher:
    """Prefetch HF shards into the local cache.

    Downloads the next shard in a background worker to hide network latency.
    Tracks the current and next shard for optional cache cleanup.
    """
    def __init__(self, repo_id, split, cache_dir, rel_files, cleanup_mode):
        self._repo_id = repo_id
        self._split = split
        self._cache_dir = cache_dir
        self._rel_files = rel_files
        self._cleanup_mode = cleanup_mode
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._future_index = None
        self._cache_root = _hf_shard_cache_root(cache_dir, repo_id, split)

    def _download(self, index):
        rel_path = self._rel_files[index]
        return _ensure_local_shard(self._repo_id, rel_path, self._cache_dir, self._split)

    def prefetch(self, index):
        if index is None or index >= len(self._rel_files):
            return
        if self._future is not None and not self._future.done():
            return
        self._future_index = index
        self._future = self._executor.submit(self._download, index)

    def get(self, index):
        if self._future_index == index and self._future is not None:
            path = self._future.result()
            self._future = None
            self._future_index = None
            return path
        return self._download(index)

    def cleanup(self, keep_indices):
        if self._cleanup_mode != "auto":
            return
        keep_paths = [
            _hf_local_shard_path(self._cache_dir, self._repo_id, self._split, self._rel_files[idx])
            for idx in keep_indices
            if idx is not None and idx < len(self._rel_files)
        ]
        # Preserve temp files for in-flight downloads to avoid cleanup races.
        temp_paths = [path.with_suffix(path.suffix + ".tmp") for path in keep_paths]
        keep_paths.extend(temp_paths)
        _cleanup_shard_cache(self._cache_root, keep_paths)

    def close(self):
        self._executor.shutdown(wait=False)


def hf_load_dataset_sharded(repo_id, split="train", name=None, cache_dir=None, data_files=None, prefetch=1):
    # Load an HF dataset shard-by-shard with local caching.
    rel_files = data_files or _hf_parquet_files(repo_id, split, name=name)
    if not rel_files:
        return None
    cleanup_mode = config.HF_SHARD_CACHE_CLEANUP

    def _iter_shards():
        prefetcher = ShardPrefetcher(repo_id, split, cache_dir, rel_files, cleanup_mode)
        try:
            for index, _ in enumerate(rel_files):
                local_path = prefetcher.get(index)
                if prefetch:
                    prefetcher.prefetch(index + 1)
                dataset = load_dataset("parquet", data_files=str(local_path), split="train", streaming=False)
                for row in dataset:
                    yield row
                keep_indices = [index, index + 1] if prefetch else [index]
                prefetcher.cleanup(keep_indices)
        finally:
            prefetcher.close()

    return IterableDataset.from_generator(_iter_shards)


def cap_streaming_rows(dataset, remaining_rows):
    # Limit a streaming dataset to a fixed number of rows.
    if remaining_rows is None:
        return dataset
    if remaining_rows <= 0:
        return dataset.take(0)
    return dataset.take(remaining_rows)


def with_initial_log(message, iterable, start_time=None):
    # Log once when the iterable yields its first element.
    logged = {"done": False}
    started = start_time or time.time()

    def _emit():
        if not logged["done"]:
            duration_s = time.time() - started
            print(message.format(duration_s=duration_s))
            logged["done"] = True

    if hasattr(iterable, "map"):
        def _log_once(example):
            _emit()
            return example
        return iterable.map(_log_once)

    def _wrapped():
        for item in iterable:
            _emit()
            yield item
    return _wrapped()


def parse_dataset_specs(specs_str):
    # Parse a comma-separated list of dataset specs into dictionaries.
    if specs_str is None:
        raise ValueError("DATASET_SPECS must be set to a non-empty string.")
    specs = []
    # Split on commas and normalize each spec string.
    for raw_spec in specs_str.split(","):
        spec = raw_spec.strip()
        if not spec:
            continue
        parts = spec.split(":")
        kind = parts[0].lower()
        if kind == "hf":
            # Parse hf:<repo_id>[:split][:text_key] or hf:<repo_id>:<config>:<split>[:text_key].
            if len(parts) < 2 or not parts[1]:
                raise ValueError(f"Invalid HF spec: {spec}")
            if len(parts) > 5:
                raise ValueError(f"Invalid HF spec: {spec}")
            name = None
            split = "train"
            text_key = "text"
            if len(parts) == 3:
                split = parts[2] or "train"
            elif len(parts) == 4:
                split = parts[2] or "train"
                text_key = parts[3] or "text"
            elif len(parts) == 5:
                name = parts[2] or None
                split = parts[3] or "train"
                text_key = parts[4] or "text"
            specs.append(
                {
                    "kind": "hf",
                    "repo_id": parts[1],
                    "name": name,
                    "split": split,
                    "text_key": text_key,
                    "spec": spec,
                }
            )
        elif kind == "txt":
            # Parse txt:<path>[:text_key].
            if len(parts) < 2 or not parts[1]:
                raise ValueError(f"Invalid TXT spec: {spec}")
            text_key = parts[2] if len(parts) > 2 and parts[2] else "text"
            specs.append(
                {
                    "kind": "txt",
                    "path": parts[1],
                    "split": "train",
                    "text_key": text_key,
                    "spec": spec,
                }
            )
        else:
            raise ValueError(f"Unknown dataset spec type: {spec}")
    if not specs:
        raise ValueError("No dataset specs found in DATASET_SPECS.")
    return specs

def _resume_cache_path(cache_dir, name):
    # Resolve a cache path for resume metadata.
    if cache_dir is None:
        return None
    cache_root = Path(cache_dir) / "resume"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / name

def _read_resume_cache(cache_path):
    # Load cached resume metadata if available.
    if cache_path is None or not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def _write_resume_cache(cache_path, payload):
    # Persist resume metadata for later runs.
    if cache_path is None:
        return
    # Avoid multi-rank races by only letting rank 0 write the cache.
    if int(os.getenv("RANK", "0")) != 0:
        return
    # Ensure the cache directory exists before writing.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    tmp_path.replace(cache_path)

def _normalize_label(value):
    # Normalize labels to compare dataset names across filesystem layouts.
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _extract_named_dir(entries, key, target):
    # Pick a directory with a matching name component like subset= or source=.
    token = f"{key}="
    normalized_target = _normalize_label(target)
    for entry in entries:
        if token not in entry:
            continue
        name = entry.split(token, 1)[1].split("/", 1)[0]
        if _normalize_label(name) == normalized_target:
            return entry
    return None


def _hf_parquet_files(repo_id, split, name=None):
    # Enumerate parquet shard files for an HF dataset split.
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    data_root = f"datasets/{repo_id}/data"
    try:
        entries = fs.ls(data_root, detail=False)
    except Exception:
        entries = []
    split_prefix = f"{split}-"
    rel_files = []
    for path in entries:
        filename = path.split("/")[-1]
        if filename.startswith(split_prefix) and filename.endswith(".parquet"):
            rel_files.append(path.replace(f"datasets/{repo_id}/", ""))
    if rel_files or not name:
        return sorted(rel_files)

    base_root = f"datasets/{repo_id}"
    try:
        base_entries = fs.ls(base_root, detail=False)
    except Exception:
        return []
    subset_dirs = [entry for entry in base_entries if "/subset=" in entry]
    subset_dir = _extract_named_dir(subset_dirs, "subset", name) if subset_dirs else None
    if not subset_dir:
        return []
    try:
        subset_entries = fs.ls(subset_dir, detail=False)
    except Exception:
        return []
    source_dirs = [entry for entry in subset_entries if "/source=" in entry]
    source_dir = _extract_named_dir(source_dirs, "source", split) if source_dirs else None
    if source_dir:
        source_entries = fs.ls(source_dir, detail=False)
        return sorted(
            entry.replace(f"datasets/{repo_id}/", "")
            for entry in source_entries
            if entry.endswith(".parquet")
        )
    return sorted(
        entry.replace(f"datasets/{repo_id}/", "")
        for entry in subset_entries
        if entry.endswith(".parquet")
    )


def _select_hf_data_files(spec, cache_dir):
    # Prefer parquet files for german-commons to unify normal and resume loads.
    if spec.get("repo_id") != "coral-nlp/german-commons":
        return None
    files, _ = _load_hf_parquet_index(
        spec["repo_id"],
        spec.get("split", "train"),
        cache_dir,
        name=spec.get("name"),
    )
    return files or None

def _hf_parquet_row_counts(repo_id, rel_files):
    # Read parquet metadata to get row counts per shard.
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    counts = []
    for rel_path in rel_files:
        hf_path = f"datasets/{repo_id}/{rel_path}"
        with fs.open(hf_path, "rb") as handle:
            counts.append(pq.ParquetFile(handle).metadata.num_rows)
    return counts

def _load_hf_parquet_index(repo_id, split, cache_dir, name=None):
    # Load or build the shard index for an HF parquet dataset.
    cache_name = f"hf-{repo_id.replace('/', '--')}"
    if name:
        cache_name = f"{cache_name}-{name}"
    cache_name = f"{cache_name}-{split}.json"
    cache_path = _resume_cache_path(cache_dir, cache_name)
    cached = _read_resume_cache(cache_path)
    if (
        cached
        and cached.get("repo_id") == repo_id
        and cached.get("split") == split
        and cached.get("name") == name
    ):
        return cached.get("files", []), cached.get("row_counts", [])

    files = _hf_parquet_files(repo_id, split, name=name)
    if not files:
        return [], []

    row_counts = _hf_parquet_row_counts(repo_id, files)
    payload = {
        "repo_id": repo_id,
        "name": name,
        "split": split,
        "files": files,
        "row_counts": row_counts,
    }
    _write_resume_cache(cache_path, payload)
    return files, row_counts

def _resolve_text_files(path):
    # Expand a txt spec path into a sorted list of files.
    path_obj = Path(path)
    if any(char in path for char in "*?[]"):
        return sorted(str(entry) for entry in path_obj.parent.glob(path_obj.name))
    return [str(path_obj)]

def _map_row_offset(row_offset, row_counts):
    # Map a global row offset to a shard index and in-shard offset.
    if not row_counts:
        return 0, row_offset
    total_rows = sum(row_counts)
    if row_offset >= total_rows:
        return len(row_counts) - 1, row_counts[-1]

    prefix = []
    running = 0
    for count in row_counts:
        running += count
        prefix.append(running)
    shard_idx = bisect_right(prefix, row_offset)
    prior = prefix[shard_idx - 1] if shard_idx > 0 else 0
    return shard_idx, row_offset - prior

def resolve_resume_plan(spec, row_offset, cache_dir=None):
    # Resolve shard-aware resume settings for a dataset spec.
    if not row_offset or row_offset <= 0:
        return None, 0, None
    if spec["kind"] == "hf":
        files, row_counts = _load_hf_parquet_index(
            spec["repo_id"],
            spec.get("split", "train"),
            cache_dir,
            name=spec.get("name"),
        )
        if not files or not row_counts:
            return None, row_offset, None
        shard_idx, in_shard_offset = _map_row_offset(row_offset, row_counts)
        shard_label = Path(files[shard_idx]).name
        return files[shard_idx:], in_shard_offset, shard_label
    if spec["kind"] == "txt":
        files = _resolve_text_files(spec["path"])
        if not files:
            return None, row_offset, None
        row_counts = [_count_text_rows(path) for path in files]
        shard_idx, in_shard_offset = _map_row_offset(row_offset, row_counts)
        shard_label = files[shard_idx]
        return files[shard_idx:], in_shard_offset, shard_label
    return None, row_offset, None

def _count_text_rows(path):
    # Count newline-delimited rows in a local text file.
    total = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total

def _rename_text_column(dataset, text_key):
    # Rename the default text column when a custom key is requested.
    if text_key == "text":
        return dataset
    if "text" not in dataset.column_names:
        return dataset
    return dataset.rename_column("text", text_key)

def _resolve_column_names(dataset, fallback=None):
    # Resolve dataset column names from schema or a single sample fallback.
    column_names = dataset.column_names
    if column_names:
        return list(column_names)
    features = getattr(dataset, "features", None)
    if features:
        return list(features.keys())
    take = getattr(dataset, "take", None)
    if take is None:
        return list(fallback) if fallback is not None else []
    try:
        sample = next(iter(take(1)))
    except Exception:
        return list(fallback) if fallback is not None else []
    if isinstance(sample, dict):
        return list(sample.keys())
    return list(fallback) if fallback is not None else []

def load_dataset_from_spec(spec, cache_dir=None, streaming=True, data_files=None):
    # Load a dataset based on a parsed spec dictionary.
    if spec["kind"] == "hf":
        # HF datasets are loaded directly from the hub.
        split = spec.get("split", "train")
        rel_files = data_files or _select_hf_data_files(spec, cache_dir)
        sharded = hf_load_dataset_sharded(
            spec["repo_id"],
            split=split,
            name=spec.get("name"),
            cache_dir=cache_dir,
            data_files=rel_files,
        )
        if sharded is not None:
            return sharded
        if rel_files is not None:
            split = "train"
        return load_dataset_source(
            spec["repo_id"],
            split=split,
            data_files=rel_files,
            cache_dir=cache_dir,
            streaming=streaming,
            name=spec.get("name"),
        )
    if spec["kind"] == "txt":
        # TXT datasets load line-delimited text from local files.
        dataset = load_dataset(
            "text",
            data_files=data_files or spec["path"],
            split=spec.get("split", "train"),
            streaming=streaming,
            cache_dir=cache_dir,
        )
        return _rename_text_column(dataset, spec.get("text_key", "text"))
    raise ValueError(f"Unsupported dataset spec kind: {spec['kind']}")

def resolve_total_rows(dataset, spec, cache_dir=None):
    # Resolve total rows for resume offsets and progress logging.
    if spec["kind"] == "hf":
        # Prefer parquet metadata for accurate row counts on streaming datasets.
        files, row_counts = _load_hf_parquet_index(
            spec["repo_id"],
            spec.get("split", "train"),
            cache_dir=cache_dir,
            name=spec.get("name"),
        )
        if row_counts:
            return sum(row_counts)
    if spec["kind"] == "txt":
        # Text files have no metadata, so count lines on disk.
        return sum(_count_text_rows(path) for path in _resolve_text_files(spec["path"]))
    if dataset.info and dataset.info.splits:
        # HF dataset info provides split sizes when available.
        split_info = dataset.info.splits.get(spec.get("split", "train"))
        if split_info is not None:
            return split_info.num_examples
    return None

def dataset_label(spec):
    # Produce a human-readable label for dataset logging.
    if spec["kind"] == "hf":
        name = spec.get("name")
        split = spec.get("split", "train")
        if name:
            return f"{spec['repo_id']}:{name}/{split}"
        return f"{spec['repo_id']}:{split}"
    if spec["kind"] == "txt":
        return spec["path"]
    return spec.get("repo_id") or spec.get("path") or "unknown"


def build_tokenizer(tokenizer, text_key="text"):
    # Wrap tokenizer to return token ids for datasets.map.
    def tokenizer_batch(batch):
        texts = batch[text_key]
        pool = _get_tokenizer_pool()
        if pool is None or len(texts) <= 1:
            token_batch = tokenizer.encode_batch(texts)
        else:
            # Split batches for parallel encoding without reordering.
            workers = min(int(config.TOKENIZER_WORKERS), len(texts))
            chunk_size = math.ceil(len(texts) / workers)
            futures = [
                pool.submit(tokenizer.encode_batch, texts[start:start + chunk_size])
                for start in range(0, len(texts), chunk_size)
            ]
            token_batch = []
            for future in futures:
                token_batch.extend(future.result())
        eos_token_id = tokenizer.token_to_id(DATASET_EOS_TOKEN)
        input_ids = []
        for encoding in token_batch:
            ids = list(encoding.ids)
            if eos_token_id is not None:
                ids.append(eos_token_id)
            input_ids.append(ids)
        row_count = batch.get("row_count")
        if row_count is not None and len(row_count) != len(input_ids):
            raise RuntimeError(
                "Tokenizer row_count mismatch: "
                f"text_rows={len(input_ids)} row_count_rows={len(row_count)}."
            )
        if row_count is not None:
            return {"input_ids": input_ids, "row_count": row_count}
        return {"input_ids": input_ids}

    return tokenizer_batch


def pack_tokens(batch, block_size, source_id=None):
    # Pack token lists into fixed-size blocks, dropping remainder tokens.
    concatenated = []
    for ids in batch["input_ids"]:
        concatenated.extend(ids)

    total_length = (len(concatenated) // block_size) * block_size
    if total_length == 0:
        return {
            "input_ids": torch.empty((0, block_size), dtype=torch.long),
            "attention_mask": torch.empty((0, block_size), dtype=torch.long),
            "row_count": torch.empty((0,), dtype=torch.long),
            "source_id": torch.empty((0,), dtype=torch.long),
        }

    batch_row_counts = batch.get("row_count")
    if batch_row_counts is not None and len(batch_row_counts) != len(batch["input_ids"]):
        raise RuntimeError(
            "Pack row_count mismatch: "
            f"input_rows={len(batch['input_ids'])} "
            f"row_count_rows={len(batch_row_counts)}."
        )
    input_ids = [
        concatenated[i:i + block_size]
        for i in range(0, total_length, block_size)
    ]
    attention_mask = [[1] * block_size for _ in input_ids]

    # Record raw row consumption on the first packed block only.
    row_counts = [0] * len(input_ids)
    if batch_row_counts is None:
        row_counts[0] = len(batch["input_ids"])
    else:
        row_counts[0] = int(sum(batch_row_counts))

    # Tag each packed sample with its source id.
    source_ids = [
        source_id if source_id is not None else -1
        for _ in input_ids
    ]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "row_count": torch.tensor(row_counts, dtype=torch.long),
        "source_id": torch.tensor(source_ids, dtype=torch.long),
    }


def build_packed_dataset(
    dataset,
    tokenizer,
    block_size,
    text_key="text",
    pack_batch_size=1000,
    source_id=None,
):
    # Tokenize and pack a dataset into fixed-length blocks.

    # Drop unused columns early to reduce dataset formatting overhead.
    column_names = _resolve_column_names(dataset, fallback=[text_key])
    if column_names:
        drop_columns = [col for col in column_names if col != text_key]
        if drop_columns:
            dataset = dataset.remove_columns(drop_columns)

    # Track raw rows so resume counts stay aligned with source rows.
    def _add_row_count(batch):
        return {"row_count": [1] * len(batch[text_key])}

    dataset = dataset.map(_add_row_count, batched=True)
    tokenizer_batch = build_tokenizer(
        tokenizer,
        text_key=text_key,
    )
    tokenized = dataset.map(
        tokenizer_batch,
        batched=True,
        remove_columns=[text_key, "row_count"],
    )
    packed_drop_columns = _resolve_column_names(tokenized)
    packed = tokenized.map(
        lambda batch: pack_tokens(batch, block_size, source_id=source_id),
        batched=True,
        batch_size=pack_batch_size,
        remove_columns=packed_drop_columns,
    )
    # Drop empty packed samples to avoid zero-length batches in DDP.
    packed = packed.filter(lambda sample: len(sample["input_ids"]) > 0)
    packed_column_names = _resolve_column_names(packed)
    if packed_column_names:
        keep_columns = {"input_ids", "attention_mask", "row_count", "source_id"}
        drop_columns = [col for col in packed_column_names if col not in keep_columns]
        if drop_columns:
            packed = packed.remove_columns(drop_columns)
    return packed


def normalize_interleave_probabilities(weights):
    # Normalize interleave weights into a probability vector.
    if weights is None:
        return None
    if not weights:
        raise ValueError("Interleave weights must not be empty.")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Interleave weights must sum to a positive value.")
    return [float(weight) / total for weight in weights]


def _least_tokens_interleave_generator(datasets, seed, token_counts):
    # Interleave datasets by always sampling from the least-consumed source.
    rng = random.Random(seed)
    token_counts = list(token_counts)
    iterators = [iter(dataset) for dataset in datasets]
    active_indices = list(range(len(iterators)))
    debug_every = int(os.getenv("DEBUG_INTERLEAVE_EVERY", "0"))
    debug_step = 0
    while active_indices:
        min_count = min(token_counts[idx] for idx in active_indices)
        candidates = [idx for idx in active_indices if token_counts[idx] == min_count]
        choice = rng.choice(candidates) if len(candidates) > 1 else candidates[0]
        # Emit periodic interleave picks for debugging local token counts.
        if debug_every > 0 and debug_step % debug_every == 0:
            print(
                "Interleave pick "
                f"step={debug_step} choice={choice} "
                f"min_tokens={min_count} counts={token_counts}",
                flush=True,
            )
        debug_step += 1
        try:
            sample = next(iterators[choice])
        except StopIteration:
            active_indices.remove(choice)
            continue
        token_count = _sample_token_count(sample)
        token_counts[choice] += token_count
        yield sample


def build_interleaved_dataset(datasets, seed=42, probabilities=None, weights_provider=None, token_counts=None):
    # Interleave datasets with normalized sampling probabilities.
    if token_counts is not None:
        return IterableDataset.from_generator(
            _least_tokens_interleave_generator,
            gen_kwargs={
                "datasets": tuple(datasets),
                "seed": seed,
                "token_counts": tuple(token_counts),
            },
        )
    if probabilities is None:
        probabilities = [1 / len(datasets)] * len(datasets)
    probabilities = normalize_interleave_probabilities(probabilities)
    return interleave_datasets(
        datasets,
        seed=seed,
        probabilities=probabilities,
        stopping_strategy="all_exhausted",
    )


def _sample_token_count(sample):
    # Count tokens from a packed sample using the attention mask.
    attention_mask = sample.get("attention_mask")
    if attention_mask is None:
        return 0
    if hasattr(attention_mask, "sum"):
        return int(attention_mask.sum())
    return int(sum(attention_mask))


# Emit a first-batch timing log while streaming batches.
def time_until_first_batch(loader, is_master):
    start = time.time()
    if is_master:
        print("Waiting for first batch...")
        loader = with_initial_log("First batch after {duration_s:.1f}s", loader, start_time=start)
    iterator = iter(loader)
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            return
        except Exception:
            if is_master:
                dataset = getattr(loader, "dataset", None)
                print("Failed while fetching a batch from the data loader.")
                if dataset is not None:
                    print(f"Dataset type: {type(dataset)}")
                    column_names = getattr(dataset, "column_names", None)
                    if column_names:
                        print(f"Dataset columns: {column_names}")
            raise
        yield batch
