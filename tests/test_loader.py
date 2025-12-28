import unittest

from model import loader


class LoaderHelperTests(unittest.TestCase):
    """Validate loader label helpers used for HF parquet discovery.

    These tests focus on string normalization and directory matching.
    They avoid network calls by exercising pure helper functions.
    """
    def test_normalize_label_strips_non_alnum(self):
        # Normalize labels by removing spaces and punctuation.
        self.assertEqual(loader._normalize_label("One Million Posts"), "onemillionposts")

    def test_extract_named_dir_matches_subset(self):
        # Match subset directory names case-insensitively.
        entries = [
            "datasets/repo/subset=Web",
            "datasets/repo/subset=Cultural",
        ]
        self.assertEqual(
            loader._extract_named_dir(entries, "subset", "web"),
            "datasets/repo/subset=Web",
        )

    def test_extract_named_dir_matches_source(self):
        # Match source directory names with normalized tokens.
        entries = [
            "datasets/repo/subset=Web/source=One Million Posts",
            "datasets/repo/subset=Web/source=Wikipedia",
        ]
        self.assertEqual(
            loader._extract_named_dir(entries, "source", "onemillionposts"),
            "datasets/repo/subset=Web/source=One Million Posts",
        )

