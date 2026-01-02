"""Helpers for building streaming datasets with packing."""

import json
import os
import random
import shutil
import functools
import time
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

from datasets import IterableDataset as HfIterableDataset, interleave_datasets, load_dataset
from torch.utils.data import IterableDataset as TorchIterableDataset

import config

from tokenizer import DATASET_EOS_TOKEN


class _LoaderProfiler:
    """Track loader stage timings for pipeline profiling.
    Aggregates per-event counts, durations, and token totals.
    Emits periodic summaries for long-running streams.
    """
    def __init__(self, every):
        self.every = max(int(every), 1)
        self.stats = {}

    def record(self, event, duration_s, rows=0, tokens=0, blocks=0, shards=0):
        stat = self.stats.setdefault(
            event,
            {"count": 0, "duration": 0.0, "rows": 0, "tokens": 0, "blocks": 0, "shards": 0},
        )
        stat["count"] += 1
        stat["duration"] += duration_s
        stat["rows"] += rows
        stat["tokens"] += tokens
        stat["blocks"] += blocks
        stat["shards"] += shards
        if stat["count"] % self.every == 0:
            self._log(event, stat)

    def _log(self, event, stat):
        # Emit a periodic profile summary for a loader stage.
        count = stat["count"]
        duration = stat["duration"]
        rows = stat["rows"]
        tokens = stat["tokens"]
        blocks = stat["blocks"]
        shards = stat["shards"]
        rows_per_s = rows / duration if duration > 0 else 0.0
        tokens_per_s = tokens / duration if duration > 0 else 0.0
        print(
            f"Loader profile: {event} count={count} "
            f"rows={rows} tokens={tokens} blocks={blocks} shards={shards} "
            f"duration={duration:.2f}s rows/s={rows_per_s:.1f} tokens/s={tokens_per_s:.1f}",
            flush=True,
        )


# Enable loader profiling when PROFILE_LOADER is truthy.
_PROFILE_LOADER = os.getenv("PROFILE_LOADER", "0") not in {"0", "", "false", "False"}
_PROFILE_LOADER_EVERY = int(os.getenv("PROFILE_LOADER_EVERY", "200"))
_LOADER_PROFILER = _LoaderProfiler(_PROFILE_LOADER_EVERY) if _PROFILE_LOADER else None
_HF_LOCAL_ONLY = config.HF_LOCAL_ONLY
_HF_PYARROW_SHARDS = config.HF_PYARROW_SHARDS
_DATASET_FAST_PIPELINE = config.DATASET_FAST_PIPELINE


def _record_loader_profile(event, duration_s, rows=0, tokens=0, blocks=0, shards=0):
    if _LOADER_PROFILER is None:
        return
    _LOADER_PROFILER.record(event, duration_s, rows=rows, tokens=tokens, blocks=blocks, shards=shards)


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
    for path in cache_root.rglob("*"):
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
    def __init__(self, repo_id, split, cache_dir, rel_files, cleanup_mode, local_files_only=False):
        self._repo_id = repo_id
        self._split = split
        self._cache_dir = cache_dir
        self._rel_files = rel_files
        self._cleanup_mode = cleanup_mode
        self._local_files_only = local_files_only
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._future_index = None
        self._cache_root = _hf_shard_cache_root(cache_dir, repo_id, split)

    def _download(self, index):
        rel_path = self._rel_files[index]
        return _ensure_local_shard(
            self._repo_id,
            rel_path,
            self._cache_dir,
            self._split,
            local_files_only=self._local_files_only,
        )

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
        _cleanup_shard_cache(self._cache_root, keep_paths)

    def close(self):
        self._executor.shutdown(wait=False)


def _iter_hf_shards(repo_id, split, name, cache_dir, rel_files, cleanup_mode, prefetch, local_files_only):
    # Stream shards sequentially with optional prefetch and cleanup.
    prefetcher = ShardPrefetcher(
        repo_id,
        split,
        cache_dir,
        rel_files,
        cleanup_mode,
        local_files_only=local_files_only,
    )
    logged_reader = False
    try:
        for index, _ in enumerate(rel_files):
            if _PROFILE_LOADER and not logged_reader:
                reader = "pyarrow" if _HF_PYARROW_SHARDS else "datasets"
                print(f"Loader profile: shard reader={reader}")
                logged_reader = True
            # Capture shard load timing for profiling.
            start = time.perf_counter()
            local_path = prefetcher.get(index)
            if prefetch:
                prefetcher.prefetch(index + 1)
            row_count = 0
            if _HF_PYARROW_SHARDS:
                for row in _iter_parquet_rows(local_path):
                    row_count += 1
                    yield row
            else:
                dataset = load_dataset("parquet", data_files=str(local_path), split="train", streaming=False)
                for row in dataset:
                    row_count += 1
                    yield row
            keep_indices = [index, index + 1] if prefetch else [index]
            prefetcher.cleanup(keep_indices)
            _record_loader_profile(
                "shard",
                time.perf_counter() - start,
                rows=row_count,
                shards=1,
            )
    finally:
        prefetcher.close()


def _iter_parquet_rows(path):
    # Stream rows from a local parquet file via PyArrow.
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches():
        columns = batch.to_pydict()
        if not columns:
            continue
        keys = list(columns.keys())
        rows = len(next(iter(columns.values())))
        for idx in range(rows):
            yield {key: columns[key][idx] for key in keys}


def hf_load_dataset_sharded(repo_id, split="train", name=None, cache_dir=None, data_files=None, prefetch=1):
    # Load an HF dataset shard-by-shard with local caching.
    rel_files = data_files or _hf_parquet_files(repo_id, split, name=name)
    if not rel_files:
        return None
    cleanup_mode = config.HF_SHARD_CACHE_CLEANUP
    return HfIterableDataset.from_generator(
        _iter_hf_shards,
        gen_kwargs={
            "repo_id": repo_id,
            "split": split,
            "name": name,
            "cache_dir": cache_dir,
            "rel_files": tuple(rel_files),
            "cleanup_mode": cleanup_mode,
            "prefetch": prefetch,
            "local_files_only": _HF_LOCAL_ONLY,
        },
    )


def cap_streaming_rows(dataset, remaining_rows):
    # Limit a streaming dataset to a fixed number of rows.
    if remaining_rows is None:
        return dataset
    if remaining_rows <= 0:
        return dataset.take(0)
    return dataset.take(remaining_rows)


class GeneratorIterableDataset(TorchIterableDataset):
    """Wrap a generator function as a torch IterableDataset.
    Keeps streaming pipelines out of datasets.map overhead.
    Stores arguments for per-worker iteration.
    """
    supports_workers = False

    def __init__(self, generator_fn, kwargs):
        self._generator_fn = generator_fn
        self._kwargs = kwargs

    def __iter__(self):
        return iter(self._generator_fn(**self._kwargs))


class PackedIterableDataset(TorchIterableDataset):
    """Stream packed token blocks from a raw row iterable.
    Batches rows, tokenizes with EOS, and packs into fixed blocks.
    Drops empty packs to keep DataLoader batches valid.
    """
    supports_workers = False

    def __init__(self, dataset, tokenizer, block_size, text_key, pack_batch_size, source_id):
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._block_size = block_size
        self._text_key = text_key
        self._pack_batch_size = pack_batch_size
        self._source_id = source_id

    def __iter__(self):
        return iter(
            _iter_packed_samples(
                self._dataset,
                self._tokenizer,
                self._block_size,
                self._text_key,
                self._pack_batch_size,
                self._source_id,
            )
        )


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


class ShuffledIterableDataset(TorchIterableDataset):
    """Shuffle an iterable dataset with a fixed-size buffer.
    Uses a deterministic RNG seed for reproducible order.
    """
    supports_workers = False

    def __init__(self, dataset, buffer_size, seed):
        self._dataset = dataset
        self._buffer_size = buffer_size
        self._seed = seed

    def __iter__(self):
        return iter(_shuffle_generator(self._dataset, self._buffer_size, self._seed))


def _shuffle_generator(dataset, buffer_size, seed):
    rng = random.Random(seed)
    buffer = []
    for item in dataset:
        if buffer_size <= 1:
            yield item
            continue
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = item
    if buffer:
        rng.shuffle(buffer)
        yield from buffer


def shuffle_dataset(dataset, buffer_size, seed):
    # Shuffle using dataset.shuffle when available, otherwise apply a buffer shuffle.
    shuffle = getattr(dataset, "shuffle", None)
    if callable(shuffle):
        return shuffle(buffer_size=buffer_size, seed=seed)
    return ShuffledIterableDataset(dataset, buffer_size, seed)


def format_dataset_for_torch(dataset):
    # Apply datasets formatting when available, otherwise return unchanged.
    with_format = getattr(dataset, "with_format", None)
    if callable(with_format):
        return with_format("torch")
    return dataset


def resolve_dataloader_workers(dataset, requested_workers):
    # Disable workers for datasets that are not safe to pickle.
    supports_workers = getattr(dataset, "supports_workers", True)
    if not supports_workers and requested_workers > 0:
        return 0
    return requested_workers


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
    # Prefer parquet files for all HF datasets to unify normal and resume loads.
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
    if _HF_LOCAL_ONLY:
        raise FileNotFoundError(
            "HF_LOCAL_ONLY is enabled but no cached parquet index was found "
            f"for {repo_id} split={split}. Run once with HF_LOCAL_ONLY=false to "
            "populate the cache."
        )

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
        if _HF_LOCAL_ONLY:
            raise FileNotFoundError(
                "HF_LOCAL_ONLY is enabled but no local shards were found for "
                f"{spec['repo_id']} split={split}."
            )
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


class _TokenizerBatch:
    """Tokenize dataset batches with stable, picklable callables.
    Carries the tokenizer and text key across DataLoader workers.
    Appends EOS tokens and preserves row counts when present.
    """
    def __init__(self, tokenizer, text_key):
        self.tokenizer = tokenizer
        self.text_key = text_key

    def __call__(self, batch):
        # Track tokenization cost when profiling is enabled.
        start = time.perf_counter() if _LOADER_PROFILER is not None else None
        token_batch = self.tokenizer.encode_batch(batch[self.text_key])
        eos_token_id = self.tokenizer.token_to_id(DATASET_EOS_TOKEN)
        input_ids = []
        for encoding in token_batch:
            ids = list(encoding.ids)
            if eos_token_id is not None:
                ids.append(eos_token_id)
            input_ids.append(ids)
        if start is not None:
            token_count = sum(len(ids) for ids in input_ids)
            _record_loader_profile(
                "tokenize",
                time.perf_counter() - start,
                rows=len(input_ids),
                tokens=token_count,
            )
        row_count = batch.get("row_count")
        if row_count is not None and len(row_count) != len(input_ids):
            raise RuntimeError(
                "Tokenizer row_count mismatch: "
                f"text_rows={len(input_ids)} row_count_rows={len(row_count)}."
            )
        if row_count is not None:
            return {"input_ids": input_ids, "row_count": row_count}
        return {"input_ids": input_ids}


def build_tokenizer(tokenizer, text_key="text"):
    # Wrap tokenizer to return token ids for datasets.map.
    return _TokenizerBatch(tokenizer, text_key)


def pack_tokens(batch, block_size, source_id=None):
    # Pack token lists into fixed-size blocks, dropping remainder tokens.
    # Capture pack timing only when profiling is enabled.
    start = time.perf_counter() if _LOADER_PROFILER is not None else None
    concatenated = []
    for ids in batch["input_ids"]:
        concatenated.extend(ids)

    total_length = (len(concatenated) // block_size) * block_size
    if total_length == 0:
        return {"input_ids": [], "attention_mask": [], "row_count": [], "source_id": []}

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
    if start is not None:
        _record_loader_profile(
            "pack",
            time.perf_counter() - start,
            rows=len(batch["input_ids"]),
            tokens=len(concatenated),
            blocks=len(input_ids),
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "row_count": row_counts,
        "source_id": source_ids,
    }


def _pack_tokens_batch(batch, block_size, source_id):
    return pack_tokens(batch, block_size, source_id=source_id)


def _non_empty_sample(sample):
    return len(sample["input_ids"]) > 0


def _add_row_count_batch(batch, text_key):
    start = time.perf_counter() if _LOADER_PROFILER is not None else None
    row_count = len(batch[text_key])
    payload = {"row_count": [1] * row_count}
    if start is not None:
        _record_loader_profile(
            "stream",
            time.perf_counter() - start,
            rows=row_count,
        )
    return payload


def _tokenize_texts(tokenizer_batch, text_key, rows):
    texts = []
    for row in rows:
        if text_key not in row:
            raise KeyError(f"Missing text_key={text_key} in row: {row}")
        texts.append(row[text_key])
    tokenized = tokenizer_batch({text_key: texts})
    return tokenized["input_ids"]


def _iter_packed_samples(dataset, tokenizer, block_size, text_key, pack_batch_size, source_id):
    if pack_batch_size <= 0:
        raise ValueError("pack_batch_size must be positive.")

    tokenizer_batch = _TokenizerBatch(tokenizer, text_key)
    batch = []
    stream_start = time.perf_counter() if _LOADER_PROFILER is not None else None
    for row in dataset:
        batch.append(row)
        if len(batch) < pack_batch_size:
            continue
        if stream_start is not None:
            _record_loader_profile(
                "stream",
                time.perf_counter() - stream_start,
                rows=len(batch),
            )
        input_ids = _tokenize_texts(tokenizer_batch, text_key, batch)
        packed = pack_tokens({"input_ids": input_ids}, block_size, source_id=source_id)
        for idx, ids in enumerate(packed["input_ids"]):
            yield {
                "input_ids": ids,
                "attention_mask": packed["attention_mask"][idx],
                "row_count": packed["row_count"][idx],
                "source_id": packed["source_id"][idx],
            }
        batch = []
        stream_start = time.perf_counter() if _LOADER_PROFILER is not None else None
    if batch:
        if stream_start is not None:
            _record_loader_profile(
                "stream",
                time.perf_counter() - stream_start,
                rows=len(batch),
            )
        input_ids = _tokenize_texts(tokenizer_batch, text_key, batch)
        packed = pack_tokens({"input_ids": input_ids}, block_size, source_id=source_id)
        for idx, ids in enumerate(packed["input_ids"]):
            yield {
                "input_ids": ids,
                "attention_mask": packed["attention_mask"][idx],
                "row_count": packed["row_count"][idx],
                "source_id": packed["source_id"][idx],
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
    if _DATASET_FAST_PIPELINE:
        return PackedIterableDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            block_size=block_size,
            text_key=text_key,
            pack_batch_size=pack_batch_size,
            source_id=source_id,
        )

    # Track raw rows so resume counts stay aligned with source rows.
    dataset = dataset.map(
        functools.partial(_add_row_count_batch, text_key=text_key),
        batched=True,
    )
    tokenizer_batch = build_tokenizer(
        tokenizer,
        text_key=text_key,
    )
    column_names = _resolve_column_names(dataset, fallback=[text_key])
    tokenized = dataset.map(
        tokenizer_batch,
        batched=True,
        remove_columns=column_names,
    )
    packed_drop_columns = _resolve_column_names(tokenized)
    packed = tokenized.map(
        functools.partial(_pack_tokens_batch, block_size=block_size, source_id=source_id),
        batched=True,
        batch_size=pack_batch_size,
        remove_columns=packed_drop_columns,
    )
    # Drop empty packed samples to avoid zero-length batches in DDP.
    packed = packed.filter(_non_empty_sample)
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


def _weighted_interleave_generator(datasets, seed, probabilities):
    # Interleave datasets by sampling from weighted probabilities.
    rng = random.Random(seed)
    iterators = [iter(dataset) for dataset in datasets]
    active_indices = list(range(len(iterators)))
    while active_indices:
        weights = [probabilities[idx] for idx in active_indices]
        choice = rng.choices(active_indices, weights=weights, k=1)[0]
        try:
            sample = next(iterators[choice])
        except StopIteration:
            active_indices.remove(choice)
            continue
        yield sample


def build_interleaved_dataset(datasets, seed=42, probabilities=None, weights_provider=None, token_counts=None):
    # Interleave datasets with normalized sampling probabilities.
    if token_counts is not None:
        if _DATASET_FAST_PIPELINE:
            return GeneratorIterableDataset(
                _least_tokens_interleave_generator,
                {
                    "datasets": tuple(datasets),
                    "seed": seed,
                    "token_counts": tuple(token_counts),
                },
            )
        return HfIterableDataset.from_generator(
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
    if _DATASET_FAST_PIPELINE:
        return GeneratorIterableDataset(
            _weighted_interleave_generator,
            {
                "datasets": tuple(datasets),
                "seed": seed,
                "probabilities": tuple(probabilities),
            },
        )
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
