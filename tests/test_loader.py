import unittest
from unittest import mock

import torch

from model import loader as loader_module


class FakeEncoding:
    def __init__(self, ids, attention_mask):
        self.ids = ids
        self.attention_mask = attention_mask


class FakeTokenizer:
    def encode(self, text):
        ids = list(range(len(text)))
        return FakeEncoding(ids, [1] * len(ids))

    def encode_batch(self, texts):
        encodings = []
        for text in texts:
            ids = list(range(len(text)))
            encodings.append(FakeEncoding(ids, [1] * len(ids)))
        return encodings


class FakeDataset:
    def __init__(self, items):
        self._items = items
        self.last_shuffle_seed = None
        self.map_num_proc = None

    def shuffle(self, seed):
        self.last_shuffle_seed = seed
        return self

    def map(self, tokenizer_batch, batched=False, num_proc=None):
        if not batched:
            raise AssertionError("Expected batched=True in map()")
        self.map_num_proc = num_proc
        batch = {"text": [item["text"] for item in self._items]}
        tokenized = tokenizer_batch(batch)
        items = []
        for input_ids, attention_mask in zip(
            tokenized["input_ids"], tokenized["attention_mask"]
        ):
            items.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }
            )
        new_dataset = FakeDataset(items)
        new_dataset.last_shuffle_seed = self.last_shuffle_seed
        new_dataset.map_num_proc = num_proc
        return new_dataset

    def with_format(self, type=None):
        return self

    def select(self, indices):
        return FakeDataset([self._items[i] for i in indices])

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class FakeShardedDataset:
    def __init__(self, repo_id, data_dir, shard_limit=None):
        shard_files = ["shard0.parquet", "shard1.parquet"]
        if shard_limit is not None:
            shard_files = shard_files[:shard_limit]
        self.shard_files = shard_files

    @property
    def num_shards(self):
        return len(self.shard_files)

    def shard_name(self, shard_index):
        return self.shard_files[shard_index]

    def load_shard(self, shard_index):
        if shard_index == 0:
            items = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        else:
            items = [{"text": "d"}, {"text": "e"}]
        return FakeDataset(items)

    def prefetch_shard(self, shard_index):
        return self.shard_files[shard_index]


def tokenizer_batch(batch):
    input_ids = []
    attention_mask = []
    for idx, _text in enumerate(batch["text"]):
        tokens = [idx, idx + 1, idx + 2]
        input_ids.append(tokens)
        attention_mask.append([1] * len(tokens))
    return {"input_ids": input_ids, "attention_mask": attention_mask}


class ShardedBatchLoaderTests(unittest.TestCase):
    def _make_loader(self, batch_size=2, seed=42, num_proc=4):
        with mock.patch.object(loader_module, "ShardedDataset", FakeShardedDataset):
            return loader_module.ShardedBatchLoader(
                repo_id="repo",
                data_dir="data",
                tokenizer_batch=tokenizer_batch,
                batch_size=batch_size,
                seed=seed,
                num_proc=num_proc,
                prefetch=False,
            )

    def test_iter_batches_tracks_positions(self):
        loader = self._make_loader(batch_size=2)
        positions = []
        shard_lengths = []
        for _batch, position, shard_index, shard_len in loader.iter_batches():
            positions.append(position)
            shard_lengths.append((shard_index, shard_len))

        self.assertEqual(positions, [(0, 2), (0, 3), (1, 2)])
        self.assertEqual(shard_lengths, [(0, 3), (0, 3), (1, 2)])

    def test_iter_batches_start_position(self):
        loader = self._make_loader(batch_size=2)
        positions = []
        for _batch, position, _shard_index, _shard_len in loader.iter_batches(start_position=(0, 2)):
            positions.append(position)

        self.assertEqual(positions, [(0, 3), (1, 2)])

    def test_load_tokenized_shard_uses_seed_and_num_proc(self):
        loader = self._make_loader(batch_size=2, seed=7, num_proc=3)
        dataset = loader._load_tokenized_shard(1)
        self.assertEqual(dataset.last_shuffle_seed, 8)
        self.assertEqual(dataset.map_num_proc, 3)

    def test_estimate_total_samples(self):
        loader = self._make_loader(batch_size=2)
        self.assertEqual(loader.estimate_total_samples(), 6)


class ChunkingTests(unittest.TestCase):
    def test_chunk_ids_with_padding(self):
        ids = list(range(5))
        chunks = loader_module.chunk_ids(ids, max_len=4, stride=2, pad_id=99)
        self.assertEqual(chunks, [[0, 1, 2, 3], [2, 3, 4, 99]])

    def test_chunk_ids_empty(self):
        chunks = loader_module.chunk_ids([], max_len=4, stride=2, pad_id=0)
        self.assertEqual(chunks, [])

    def test_chunk_ids_no_chunk_when_short(self):
        ids = list(range(3))
        chunks = loader_module.chunk_ids(ids, max_len=5, stride=0, pad_id=99)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], [0, 1, 2, 99, 99])

    def test_chunk_ids_cover_all_tokens(self):
        ids = list(range(10))
        chunks = loader_module.chunk_ids(ids, max_len=4, stride=0, pad_id=99)
        flattened = []
        for chunk in chunks:
            flattened.extend([token for token in chunk if token != 99])
        self.assertEqual(flattened, ids)

    def test_build_chunking_tokenizer(self):
        tokenizer = FakeTokenizer()
        tokenizer_batch = loader_module.build_chunking_tokenizer(
            tokenizer,
            pad_id=99,
            max_len=4,
            stride=2,
        )
        batch = {"text": ["hello"]}
        output = tokenizer_batch(batch)
        self.assertEqual(output["input_ids"], [[0, 1, 2, 3], [2, 3, 4, 99]])
        self.assertEqual(output["attention_mask"], [[1, 1, 1, 1], [1, 1, 1, 0]])

    def test_build_tokenizer(self):
        tokenizer = FakeTokenizer()
        tokenizer_batch = loader_module.build_tokenizer(tokenizer)
        batch = {"text": ["hi", "hey"]}
        output = tokenizer_batch(batch)
        self.assertEqual(output["input_ids"], [[0, 1], [0, 1, 2]])
        self.assertEqual(output["attention_mask"], [[1, 1], [1, 1, 1]])


if __name__ == "__main__":
    unittest.main()
