"""Helpers for building streaming datasets with packing."""

import math
import random

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
