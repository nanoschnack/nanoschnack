import tempfile
import time
import unittest
from unittest import mock

from datasets import IterableDataset

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

    def test_select_hf_data_files_limits_repo(self):
        # Only german-commons should opt into parquet file selection.
        spec = {"repo_id": "other/repo", "split": "train", "name": None}
        with mock.patch.object(loader, "_load_hf_parquet_index") as mocked:
            self.assertIsNone(loader._select_hf_data_files(spec, cache_dir="cache"))
            mocked.assert_not_called()

    def test_select_hf_data_files_uses_parquet_index(self):
        # Use parquet file list when present for german-commons.
        spec = {"repo_id": "coral-nlp/german-commons", "split": "wiki", "name": "web"}
        with mock.patch.object(loader, "_load_hf_parquet_index", return_value=(["file.parquet"], [10])):
            self.assertEqual(loader._select_hf_data_files(spec, cache_dir="cache"), ["file.parquet"])

    def test_cap_streaming_rows_limits_stream(self):
        dataset = IterableDataset.from_generator(lambda: ({"x": i} for i in range(5)))

        capped = loader.cap_streaming_rows(dataset, 2)

        self.assertEqual([row["x"] for row in capped], [0, 1])

    def test_with_initial_log_logs_once(self):
        dataset = IterableDataset.from_generator(lambda: ({"x": i} for i in range(2)))
        start_time = time.time() - 3

        with mock.patch.object(loader, "print") as mocked_print:
            wrapped = loader.with_initial_log(
                "skip done rows=5 duration={duration_s:.0f}s",
                dataset,
                start_time=start_time,
            )
            iterator = iter(wrapped)
            next(iterator)
            next(iterator)

        mocked_print.assert_called_once()
        printed = mocked_print.call_args[0][0]
        self.assertIn("skip done rows=5", printed)

    def test_estimate_cache_round_trip(self):
        spec = {"spec": "hf:repo:train:text", "text_key": "text"}

        class _Tokenizer:
            def get_vocab_size(self):
                return 7

        cache_key = loader.estimate_cache_key(spec, _Tokenizer(), sample_size=5)
        payload = {"avg_tokens": 1.5, "est_total_tokens": 42, "total_rows": 10}

        with tempfile.TemporaryDirectory() as tmpdir:
            loader.write_estimate_cache(tmpdir, cache_key, payload)
            cached = loader.read_estimate_cache(tmpdir, cache_key)

        self.assertEqual(cached, payload)

    def test_estimate_streaming_cached_reuses_value(self):
        spec = {"spec": "hf:repo:train:text", "text_key": "text"}

        class _Tokenizer:
            def encode(self, text):
                return mock.Mock(ids=[0] * len(text))

            def token_to_id(self, token):
                return None

            def get_vocab_size(self):
                return 11

        first_dataset = IterableDataset.from_generator(
            lambda: iter([{"text": "a"}, {"text": "bb"}])
        )
        second_dataset = IterableDataset.from_generator(
            lambda: iter([{"text": "cccc"}])
        )
        estimator = loader.TokenEstimator(_Tokenizer(), sample_size=1, text_key="text")

        with tempfile.TemporaryDirectory() as tmpdir:
            avg_tokens, est_total = estimator.estimate_streaming_cached(
                first_dataset,
                total_rows=2,
                spec=spec,
                cache_dir=tmpdir,
            )
            cached_avg, cached_total = estimator.estimate_streaming_cached(
                second_dataset,
                total_rows=2,
                spec=spec,
                cache_dir=tmpdir,
            )

        self.assertEqual(avg_tokens, cached_avg)
        self.assertEqual(est_total, cached_total)

    def test_pack_tokens_uses_row_count(self):
        batch = {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "row_count": [2, 3],
        }

        packed = loader.pack_tokens(batch, block_size=3, source_id=1)

        self.assertEqual(packed["row_count"][0], 5)

    def test_pack_tokens_respects_row_count_over_expanded_inputs(self):
        batch = {
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "row_count": [1, 1, 1],
        }

        packed = loader.pack_tokens(batch, block_size=3, source_id=1)

        self.assertEqual(packed["row_count"][0], 3)

    def test_build_packed_dataset_drops_extra_columns(self):
        class _Encoding:
            def __init__(self, ids):
                self.ids = ids

        class _Tokenizer:
            def encode_batch(self, texts):
                return [_Encoding([1, 2, 3]) for _ in texts]

            def token_to_id(self, token):
                return None

        dataset = IterableDataset.from_generator(
            lambda: ({"text": "hello", "extra": i} for i in range(3))
        )
        packed = loader.build_packed_dataset(
            dataset,
            tokenizer=_Tokenizer(),
            block_size=2,
            text_key="text",
            pack_batch_size=2,
            source_id=0,
        )

        first = next(iter(packed))

        self.assertEqual(
            set(first.keys()),
            {"input_ids", "attention_mask", "row_count", "source_id"},
        )

    def test_resolve_column_names_prefers_features(self):
        class _Dataset:
            column_names = None
            features = {"text": None, "id": None}

        self.assertEqual(loader._resolve_column_names(_Dataset()), ["text", "id"])

    def test_resolve_column_names_falls_back_to_fallback(self):
        class _Dataset:
            column_names = None
            features = None

        self.assertEqual(loader._resolve_column_names(_Dataset(), fallback=["content"]), ["content"])

    def test_pack_tokens_raises_on_row_count_mismatch(self):
        batch = {
            "input_ids": [[1, 2], [3, 4]],
            "row_count": [1],
        }

        with self.assertRaisesRegex(RuntimeError, "Pack row_count mismatch"):
            loader.pack_tokens(batch, block_size=2, source_id=1)
