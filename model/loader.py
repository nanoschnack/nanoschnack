"""Helpers for loading datasets in shard-sized chunks."""

from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from torch.utils.data import DataLoader


class ShardedDataset:
    """Download and load datasets shard by shard for deterministic training.

    Uses the Hugging Face Hub API to list parquet shards and download them
    on demand into a local directory.
    """

    def __init__(self, repo_id, data_dir, shard_limit=None):
        # Store configuration for loading shards from the hub.
        self.repo_id = repo_id
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Discover parquet shard files in the dataset repository.
        shard_files = [
            filename
            for filename in list_repo_files(repo_id, repo_type="dataset")
            if filename.endswith(".parquet")
        ]
        shard_files.sort()
        if shard_limit is not None:
            shard_files = shard_files[:shard_limit]

        if not shard_files:
            raise ValueError("No parquet shards found in the dataset repo.")

        self.shard_files = shard_files

    @property
    def num_shards(self):
        # Report how many shards are available for iteration.
        return len(self.shard_files)

    def load_shard(self, shard_index):
        # Download a shard locally (cached) and return a datasets.Dataset.
        filename = self.shard_files[shard_index]
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(self.data_dir),
        )
        return load_dataset(
            "parquet",
            data_files={"train": [local_path]},
            split="train",
            cache_dir=str(self.data_dir),
        )

    def shard_name(self, shard_index):
        # Report the upstream shard filename for logging.
        return self.shard_files[shard_index]

    def prefetch_shard(self, shard_index):
        # Download a shard into the local cache for faster subsequent loads.
        filename = self.shard_files[shard_index]
        return hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(self.data_dir),
        )


def chunk_ids(ids, max_len, stride, pad_id):
    if len(ids) == 0:
        return []
    step = max_len - stride
    chunks = []
    for start in range(0, len(ids), step):
        chunk = ids[start:start + max_len]
        if len(chunk) == 0:
            continue
        if len(chunk) < max_len:
            chunk = chunk + [pad_id] * (max_len - len(chunk))
        chunks.append(chunk)
        if start + max_len >= len(ids):
            break
    return chunks


def build_chunking_tokenizer(tokenizer, pad_id, max_len, stride):
    def tokenizer_batch(batch):
        input_ids = []
        attention_mask = []
        for text in batch["text"]:
            ids = tokenizer.encode(text).ids
            for chunk in chunk_ids(ids, max_len=max_len, stride=stride, pad_id=pad_id):
                input_ids.append(chunk)
                attention_mask.append([1 if t != pad_id else 0 for t in chunk])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return tokenizer_batch


def build_tokenizer(tokenizer):
    def tokenizer_batch(batch):
        token_batch = tokenizer.encode_batch(batch["text"])
        return {
            "input_ids": [e.ids for e in token_batch],
            "attention_mask": [e.attention_mask for e in token_batch],
        }

    return tokenizer_batch


class ShardedBatchLoader:
    """Iterate over shard-sized datasets with deterministic shuffle and resume.

    Wraps ShardedDataset to provide batches and track a (shard, offset) position.
    The position points to the next sample after the most recent batch.
    """

    def __init__(
        self,
        repo_id,
        data_dir,
        tokenizer_batch,
        batch_size,
        seed=42,
        shard_limit=None,
        num_proc=None,
        prefetch=True,
    ):
        # Keep configuration for loading shards and building batches.
        self.shards = ShardedDataset(repo_id, data_dir, shard_limit=shard_limit)
        self.tokenizer_batch = tokenizer_batch
        self.batch_size = batch_size
        self.seed = seed
        self.num_proc = num_proc or max(1, (os.cpu_count() or 1) - 1)
        self.prefetch = prefetch

        # Track the last consumed position (shard index, sample offset).
        self.position = (0, 0)
        self._shard_lengths = {}
        self._prefetch_executor = None
        self._prefetch_future = None
        self._prefetch_index = None

    @property
    def num_shards(self):
        # Expose shard count for logging or validation.
        return self.shards.num_shards

    def iter_batches(self, start_position=None):
        # Iterate over shards and yield batches with the next resume position.
        if self.prefetch and self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1)

        if start_position is None:
            start_shard, start_offset = 0, 0
        else:
            start_shard, start_offset = start_position

        self.position = (start_shard, start_offset)

        try:
            for shard_index in range(start_shard, self.num_shards):
                if self.prefetch:
                    self._prefetch_next(shard_index + 1)

                # Announce the upstream shard filename before loading.
                print(
                    f"Loading shard {shard_index + 1}/{self.num_shards}: "
                    f"{self.shards.shard_name(shard_index)}",
                    flush=True,
                )
                dataset = self._load_tokenized_shard(shard_index)
                shard_len = len(dataset)
                self._shard_lengths[shard_index] = shard_len

                if shard_index == start_shard and start_offset > 0:
                    dataset_len = len(dataset)
                    if start_offset >= dataset_len:
                        start_offset = 0
                    else:
                        dataset = dataset.select(range(start_offset, dataset_len))

                shard_offset = start_offset if shard_index == start_shard else 0
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                for batch in loader:
                    shard_offset += batch["input_ids"].size(0)
                    self.position = (shard_index, shard_offset)
                    yield batch, self.position, shard_index, shard_len

                start_offset = 0
        finally:
            if self._prefetch_executor is not None:
                self._prefetch_executor.shutdown(wait=False)
                self._prefetch_executor = None
                self._prefetch_future = None
                self._prefetch_index = None

    def estimate_total_samples(self):
        # Estimate total sample count assuming uniform shard sizes.
        first_len = self._get_shard_length(0)
        return first_len * self.num_shards

    def _get_shard_length(self, shard_index):
        # Cache the tokenized shard length for progress and scheduling.
        if shard_index in self._shard_lengths:
            return self._shard_lengths[shard_index]

        dataset = self._load_tokenized_shard(shard_index)
        shard_len = len(dataset)
        self._shard_lengths[shard_index] = shard_len
        return shard_len

    def _load_tokenized_shard(self, shard_index):
        # Load, shuffle, and tokenize a shard consistently.
        raw_ds = self.shards.load_shard(shard_index)
        shuffled = raw_ds.shuffle(seed=self.seed + shard_index)
        dataset = shuffled.map(self.tokenizer_batch, batched=True, num_proc=self.num_proc)
        return dataset.with_format(type="torch")

    def _prefetch_next(self, shard_index):
        # Prefetch the next shard into the local cache in the background.
        if shard_index >= self.num_shards:
            return

        if self._prefetch_index == shard_index and self._prefetch_future is not None:
            return

        self._prefetch_index = shard_index
        self._prefetch_future = self._prefetch_executor.submit(
            self.shards.prefetch_shard,
            shard_index,
        )
