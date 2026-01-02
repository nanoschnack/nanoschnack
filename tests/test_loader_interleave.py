import itertools
import unittest

from datasets import IterableDataset

from loader import build_interleaved_dataset, normalize_interleave_probabilities


class LoaderInterleaveTests(unittest.TestCase):
    """Cover interleave probability normalization helpers.

    Confirms normalization preserves ratios across weights.
    Validates error handling for empty or invalid inputs.
    Keeps data minimal for fast unit runs.
    """
    def test_normalize_interleave_probabilities(self):
        weights = [1.0, 1.0, 2.0]
        normalized = normalize_interleave_probabilities(weights)
        self.assertEqual(normalized, [0.25, 0.25, 0.5])

    def test_normalize_interleave_probabilities_none(self):
        self.assertIsNone(normalize_interleave_probabilities(None))

    def test_normalize_interleave_probabilities_empty(self):
        with self.assertRaises(ValueError):
            normalize_interleave_probabilities([])

    def test_normalize_interleave_probabilities_invalid_sum(self):
        with self.assertRaises(ValueError):
            normalize_interleave_probabilities([0.0, 0.0])

    def test_build_interleaved_dataset_least_tokens(self):
        def _make_dataset(prefix):
            def _gen():
                for idx in range(3):
                    yield {
                        "value": f"{prefix}{idx}",
                        "attention_mask": [1, 1],
                    }
            return IterableDataset.from_generator(_gen)

        token_counts = [0, 100]
        dataset = build_interleaved_dataset(
            [_make_dataset("a"), _make_dataset("b")],
            seed=123,
            token_counts=token_counts,
        )
        samples = list(itertools.islice(dataset, 3))

        self.assertEqual([item["value"] for item in samples], ["a0", "a1", "a2"])


if __name__ == "__main__":
    unittest.main()
