"""Helpers for building streaming datasets with packing."""

import math
import random
from pathlib import Path

from datasets import interleave_datasets, load_dataset


class TokenEstimator:
    """Estimate token counts per document without materializing datasets.

    Uses a deterministic sample of each dataset to approximate the average
    number of tokens per document. The estimator scales the sample average
    to the full dataset size to approximate total tokens.
    """

    def __init__(self, tokenizer, sample_size=1000, seed=42, text_key="text"):
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
            total += len(self.tokenizer.encode(text).ids)
        return total / len(texts)


def load_dataset_source(repo_id, split="train", data_files=None, cache_dir=None, streaming=True):
    # Build a dataset from hub or local parquet files.
    if data_files is not None:
        return load_dataset(
            "parquet",
            data_files=data_files,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir,
        )
    return load_dataset(
        repo_id,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
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
            # Parse hf:<repo_id>[:split][:text_key].
            if len(parts) < 2 or not parts[1]:
                raise ValueError(f"Invalid HF spec: {spec}")
            split = parts[2] if len(parts) > 2 and parts[2] else "train"
            text_key = parts[3] if len(parts) > 3 and parts[3] else "text"
            specs.append(
                {
                    "kind": "hf",
                    "repo_id": parts[1],
                    "split": split,
                    "text_key": text_key,
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
                }
            )
        else:
            raise ValueError(f"Unknown dataset spec type: {spec}")
    if not specs:
        raise ValueError("No dataset specs found in DATASET_SPECS.")
    return specs

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

def load_dataset_from_spec(spec, cache_dir=None, streaming=True):
    # Load a dataset based on a parsed spec dictionary.
    if spec["kind"] == "hf":
        # HF datasets are loaded directly from the hub.
        dataset = load_dataset_source(
            spec["repo_id"],
            split=spec.get("split", "train"),
            cache_dir=cache_dir,
            streaming=streaming,
        )
        return dataset
    if spec["kind"] == "txt":
        # TXT datasets load line-delimited text from local files.
        dataset = load_dataset(
            "text",
            data_files=spec["path"],
            split=spec.get("split", "train"),
            streaming=streaming,
            cache_dir=cache_dir,
        )
        return _rename_text_column(dataset, spec.get("text_key", "text"))
    raise ValueError(f"Unsupported dataset spec kind: {spec['kind']}")

def resolve_total_rows(dataset, spec):
    # Resolve total rows for streaming token estimates.
    if spec["kind"] == "txt":
        # Text files have no metadata, so count lines on disk.
        return _count_text_rows(spec["path"])
    if dataset.info and dataset.info.splits:
        # HF dataset info provides split sizes when available.
        split_info = dataset.info.splits.get(spec.get("split", "train"))
        if split_info is not None:
            return split_info.num_examples
    return None

def dataset_label(spec):
    # Produce a human-readable label for dataset logging.
    if spec["kind"] == "hf":
        return spec["repo_id"]
    if spec["kind"] == "txt":
        return spec["path"]
    return spec.get("repo_id") or spec.get("path") or "unknown"


def build_tokenizer(tokenizer, text_key="text"):
    # Wrap tokenizer to return token ids for datasets.map.
    def tokenizer_batch(batch):
        token_batch = tokenizer.encode_batch(batch[text_key])
        return {"input_ids": [e.ids for e in token_batch]}

    return tokenizer_batch


def pack_tokens(batch, block_size):
    # Pack token lists into fixed-size blocks, dropping remainder tokens.
    concatenated = []
    for ids in batch["input_ids"]:
        concatenated.extend(ids)

    total_length = (len(concatenated) // block_size) * block_size
    if total_length == 0:
        return {"input_ids": [], "attention_mask": []}

    input_ids = [
        concatenated[i:i + block_size]
        for i in range(0, total_length, block_size)
    ]
    attention_mask = [[1] * block_size for _ in input_ids]
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_packed_dataset(
    dataset,
    tokenizer,
    block_size,
    text_key="text",
    pack_batch_size=1000,
):
    # Tokenize and pack a dataset into fixed-length blocks.
    tokenizer_batch = build_tokenizer(tokenizer, text_key=text_key)
    column_names = dataset.column_names or [text_key]
    tokenized = dataset.map(
        tokenizer_batch,
        batched=True,
        remove_columns=column_names,
    )
    packed = tokenized.map(
        lambda batch: pack_tokens(batch, block_size),
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
