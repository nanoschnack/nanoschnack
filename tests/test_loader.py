import math
import random
import unittest

from datasets import Dataset
from datasets import IterableDataset

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


class TokenEstimatorTests(unittest.TestCase):
    def test_estimate_tokens_from_sample(self):
        tokenizer = FakeTokenizer()
        dataset = Dataset.from_dict({"text": ["aa", "bbbb", "c"]})
        estimator = loader_module.TokenEstimator(tokenizer, sample_size=2, seed=7)
        avg_tokens, est_total = estimator.estimate_dataset(dataset)

        rng = random.Random(7)
        indices = [rng.randrange(len(dataset)) for _ in range(2)]
        expected_avg = sum(len(dataset[idx]["text"]) for idx in indices) / 2
        expected_total = int(math.ceil(expected_avg * len(dataset)))

        self.assertEqual(avg_tokens, expected_avg)
        self.assertEqual(est_total, expected_total)

    def test_estimate_streaming(self):
        tokenizer = FakeTokenizer()
        items = [{"text": "aaa"}, {"text": "b"}, {"text": "cc"}]
        dataset = IterableDataset.from_generator(lambda: iter(items))
        estimator = loader_module.TokenEstimator(tokenizer, sample_size=2)
        avg_tokens, est_total = estimator.estimate_streaming(dataset, total_rows=3)

        expected_avg = (len(items[0]["text"]) + len(items[1]["text"])) / 2
        expected_total = int(math.ceil(expected_avg * 3))

        self.assertEqual(avg_tokens, expected_avg)
        self.assertEqual(est_total, expected_total)


class PackingTests(unittest.TestCase):
    def test_pack_tokens(self):
        batch = {"input_ids": [[0, 1, 2], [3, 4, 5, 6]]}
        packed = loader_module.pack_tokens(batch, block_size=4)
        self.assertEqual(packed["input_ids"], [[0, 1, 2, 3]])
        self.assertEqual(packed["attention_mask"], [[1, 1, 1, 1]])


class BuildPackedDatasetTests(unittest.TestCase):
    def test_build_packed_dataset(self):
        tokenizer = FakeTokenizer()
        dataset = Dataset.from_dict({"text": ["abcd", "ef", "ghij"]})
        packed = loader_module.build_packed_dataset(
            dataset,
            tokenizer=tokenizer,
            block_size=4,
            pack_batch_size=2,
        )
        self.assertEqual(len(packed), 2)
        first = packed[0]
        self.assertEqual(len(first["input_ids"]), 4)
        self.assertEqual(first["attention_mask"].tolist(), [1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
