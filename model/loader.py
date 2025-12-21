"""Helpers for loading datasets in shard-sized chunks."""

from pathlib import Path

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
    ):
        # Keep configuration for loading shards and building batches.
        self.shards = ShardedDataset(repo_id, data_dir, shard_limit=shard_limit)
        self.tokenizer_batch = tokenizer_batch
        self.batch_size = batch_size
        self.seed = seed

        # Track the last consumed position (shard index, sample offset).
        self.position = (0, 0)

    @property
    def num_shards(self):
        # Expose shard count for logging or validation.
        return self.shards.num_shards

    def iter_batches(self, start_position=None):
        # Iterate over shards and yield batches with the next resume position.
        if start_position is None:
            start_shard, start_offset = 0, 0
        else:
            start_shard, start_offset = start_position

        self.position = (start_shard, start_offset)

        for shard_index in range(start_shard, self.num_shards):
            # Announce the upstream shard filename before loading.
            print(
                f"Loading shard {shard_index + 1}/{self.num_shards}: "
                f"{self.shards.shard_name(shard_index)}",
                flush=True,
            )
            raw_ds = self.shards.load_shard(shard_index)
            shuffled = raw_ds.shuffle(seed=self.seed + shard_index)
            dataset = shuffled.map(self.tokenizer_batch, batched=True)
            dataset = dataset.with_format(type="torch")
            shard_len = len(dataset)

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
