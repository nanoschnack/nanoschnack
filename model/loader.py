"""Helpers for building streaming datasets with packing."""

import json
import math
import random
from bisect import bisect_right
from pathlib import Path
import re

from datasets import interleave_datasets, load_dataset

from tokenizer import DATASET_EOS_TOKEN

class TokenEstimator:
    """Estimate token counts per document without materializing datasets.

    Uses a deterministic sample of each dataset to approximate the average
    number of tokens per document. The estimator scales the sample average
    to the full dataset size to approximate total tokens.
    """

    def __init__(
        self,
        tokenizer,
        sample_size=1000,
        seed=42,
        text_key="text",
    ):
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.seed = seed
        self.text_key = text_key

    def estimate_dataset(self, dataset):
        # Sample texts and estimate tokens per document.
        sample = self._sample_texts(dataset)
        avg_tokens = self._average_tokens(sample)
        return avg_tokens, int(math.ceil(avg_tokens * len(dataset)))

    def estimate_streaming(self, dataset, total_rows, sample_size=None):
        # Estimate tokens from a streaming dataset using a fixed sample window.
        if total_rows is None or total_rows <= 0:
            raise ValueError("total_rows must be provided for streaming estimates")

        sample = self._sample_streaming_texts(dataset, sample_size=sample_size)
        avg_tokens = self._average_tokens(sample)
        return avg_tokens, int(math.ceil(avg_tokens * total_rows))

    def _sample_texts(self, dataset):
        # Return a deterministic random sample of texts.
        if len(dataset) == 0:
            return []

        # Clamp to dataset size to avoid index errors.
        sample_size = min(self.sample_size, len(dataset))
        rng = random.Random(self.seed)
        indices = [rng.randrange(len(dataset)) for _ in range(sample_size)]
        return [dataset[idx][self.text_key] for idx in indices]

    def _sample_streaming_texts(self, dataset, sample_size=None):
        # Take the first N texts from a streaming dataset.
        size = sample_size or self.sample_size
        texts = []
        for idx, sample in enumerate(dataset):
            if idx >= size:
                break
            texts.append(sample[self.text_key])
        return texts

    def _average_tokens(self, texts):
        # Average token counts across sampled texts.
        if not texts:
            return 0.0

        # Estimate tokens from tokenized lengths.
        total = 0
        for text in texts:
            count = len(self.tokenizer.encode(text).ids)
            if self.tokenizer.token_to_id(DATASET_EOS_TOKEN) is not None:
                count += 1
            total += count
        return total / len(texts)


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

def load_dataset_from_spec(spec, cache_dir=None, streaming=True, data_files=None):
    # Load a dataset based on a parsed spec dictionary.
    if spec["kind"] == "hf":
        # HF datasets are loaded directly from the hub.
        split = spec.get("split", "train")
        if data_files is not None:
            split = "train"
        dataset = load_dataset_source(
            spec["repo_id"],
            split=split,
            data_files=data_files,
            cache_dir=cache_dir,
            streaming=streaming,
            name=spec.get("name"),
        )
        return dataset
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
    # Resolve total rows for streaming token estimates.
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
        token_batch = tokenizer.encode_batch(batch[text_key])
        eos_token_id = tokenizer.token_to_id(DATASET_EOS_TOKEN)
        input_ids = []
        for encoding in token_batch:
            ids = list(encoding.ids)
            if eos_token_id is not None:
                ids.append(eos_token_id)
            input_ids.append(ids)
        return {"input_ids": input_ids}

    return tokenizer_batch


def pack_tokens(batch, block_size, source_id=None):
    # Pack token lists into fixed-size blocks, dropping remainder tokens.
    concatenated = []
    for ids in batch["input_ids"]:
        concatenated.extend(ids)

    total_length = (len(concatenated) // block_size) * block_size
    if total_length == 0:
        return {"input_ids": [], "attention_mask": [], "row_count": [], "source_id": []}

    input_ids = [
        concatenated[i:i + block_size]
        for i in range(0, total_length, block_size)
    ]
    attention_mask = [[1] * block_size for _ in input_ids]

    # Record raw row consumption on the first packed block only.
    row_counts = [0] * len(input_ids)
    row_counts[0] = len(batch["input_ids"])

    # Tag each packed sample with its source id.
    source_ids = [
        source_id if source_id is not None else -1
        for _ in input_ids
    ]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "row_count": row_counts,
        "source_id": source_ids,
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
    tokenizer_batch = build_tokenizer(
        tokenizer,
        text_key=text_key,
    )
    column_names = dataset.column_names or [text_key]
    tokenized = dataset.map(
        tokenizer_batch,
        batched=True,
        remove_columns=column_names,
    )
    packed = tokenized.map(
        lambda batch: pack_tokens(batch, block_size, source_id=source_id),
        batched=True,
        batch_size=pack_batch_size,
    )
    return packed.with_format("torch")


def build_interleaved_dataset(datasets, seed=42):
    # Interleave datasets in a round-robin fashion.
    return interleave_datasets(
        datasets,
        seed=seed,
        stopping_strategy="all_exhausted",
    )
