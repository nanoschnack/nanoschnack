import tempfile
import time
from pathlib import Path
import unittest
from unittest import mock

from datasets import IterableDataset

from model import loader

_FAKE_SHARD_PATHS = {}
_FAKE_ROWS_BY_PATH = {}
_FAKE_ENSURE_CALLS = []


def _fake_ensure_local_shard(repo_id, rel_path, cache_dir, split, local_files_only=False):
    _FAKE_ENSURE_CALLS.append(rel_path)
    return _FAKE_SHARD_PATHS[rel_path]


def _fake_load_dataset(kind, data_files=None, split=None, streaming=None):
    return _FAKE_ROWS_BY_PATH[str(data_files)]


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

    def test_hf_local_shard_path_uses_basename(self):
        cache_dir = Path("/tmp/cache-root")
        path = loader._hf_local_shard_path(
            cache_dir,
            "org/repo",
            "train",
            "data/train-00001-of-00010.parquet",
        )
        self.assertEqual(
            path.as_posix(),
            "/tmp/cache-root/hf_shards/org/repo/train/train-00001-of-00010.parquet",
        )

    def test_cleanup_shard_cache_removes_untracked_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            keep = root / "keep.parquet"
            drop = root / "drop.parquet"
            keep.write_text("keep")
            drop.write_text("drop")

            loader._cleanup_shard_cache(root, [keep])

            self.assertTrue(keep.exists())
            self.assertFalse(drop.exists())

    def test_shard_prefetcher_cleanup_keeps_tmp_for_kept_paths(self):
        # Keep tracked shards and temp files when cleanup runs.
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            rel_files = ["data/keep", "data/drop.parquet"]
            prefetcher = loader.ShardPrefetcher("org/repo", "train", cache_dir, rel_files, "auto")
            try:
                keep_path = loader._hf_local_shard_path(cache_dir, "org/repo", "train", rel_files[0])
                drop_path = loader._hf_local_shard_path(cache_dir, "org/repo", "train", rel_files[1])
                keep_path.parent.mkdir(parents=True, exist_ok=True)
                keep_path.write_text("keep")
                drop_path.write_text("drop")
                keep_tmp = keep_path.with_name(keep_path.name + ".tmp")
                drop_tmp = drop_path.with_name(drop_path.name + ".tmp")
                keep_tmp.write_text("keep tmp")
                drop_tmp.write_text("drop tmp")

                prefetcher.cleanup([0])
            finally:
                prefetcher.close()

            self.assertTrue(keep_path.exists())
            self.assertTrue(keep_tmp.exists())
            self.assertFalse(drop_path.exists())
            self.assertTrue(drop_tmp.exists())

    def test_hf_load_dataset_sharded_prefetches_next_shard(self):
        rel_files = [
            "data/train-00000.parquet",
            "data/train-00001.parquet",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            local_paths = {
                rel_path: cache_dir / "hf_shards" / "org/repo" / "train" / Path(rel_path).name
                for rel_path in rel_files
            }
            _FAKE_SHARD_PATHS.clear()
            _FAKE_ROWS_BY_PATH.clear()
            _FAKE_ENSURE_CALLS.clear()
            _FAKE_SHARD_PATHS.update(local_paths)
            _FAKE_ROWS_BY_PATH[str(local_paths[rel_files[0]])] = [{"id": 0}, {"id": 1}]
            _FAKE_ROWS_BY_PATH[str(local_paths[rel_files[1]])] = [{"id": 2}]

            original_ensure = loader._ensure_local_shard
            original_load_dataset = loader.load_dataset
            loader._ensure_local_shard = _fake_ensure_local_shard
            loader.load_dataset = _fake_load_dataset
            try:
                dataset = loader.hf_load_dataset_sharded(
                    "org/repo",
                    split="train",
                    cache_dir=cache_dir,
                    data_files=rel_files,
                    prefetch=1,
                )
                rows = list(dataset)
            finally:
                loader._ensure_local_shard = original_ensure
                loader.load_dataset = original_load_dataset

        self.assertEqual(rows, [{"id": 0}, {"id": 1}, {"id": 2}])
        self.assertIn(rel_files[0], _FAKE_ENSURE_CALLS)
        self.assertIn(rel_files[1], _FAKE_ENSURE_CALLS)

    def test_hf_load_dataset_sharded_returns_none_without_files(self):
        with mock.patch.object(loader, "_hf_parquet_files", return_value=[]):
            dataset = loader.hf_load_dataset_sharded("org/repo", split="train", cache_dir="cache")

        self.assertIsNone(dataset)

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

    def test_build_packed_dataset_filters_empty(self):
        class _Encoding:
            def __init__(self, ids):
                self.ids = ids

        class _Tokenizer:
            def encode_batch(self, texts):
                return [_Encoding([]) for _ in texts]

            def token_to_id(self, token):
                return None

        dataset = IterableDataset.from_generator(
            lambda: ({"text": "hi"} for _ in range(2))
        )
        packed = loader.build_packed_dataset(
            dataset,
            tokenizer=_Tokenizer(),
            block_size=4,
            text_key="text",
            pack_batch_size=2,
            source_id=0,
        )

        self.assertIsNone(next(iter(packed), None))

    def test_build_packed_dataset_drops_unused_columns_before_tokenize(self):
        class _Encoding:
            def __init__(self, ids):
                self.ids = ids

        class _Tokenizer:
            def encode_batch(self, texts):
                return [_Encoding([1, 2, 3]) for _ in texts]

            def token_to_id(self, token):
                return None

        dataset = IterableDataset.from_generator(
            lambda: ({"text": "hello", "extra": i} for i in range(2))
        )

        def _assert_tokenizer_batch(batch):
            self.assertIn("text", batch)
            self.assertIn("row_count", batch)
            self.assertNotIn("extra", batch)
            row_count = batch.get("row_count")
            encoded = _Tokenizer().encode_batch(batch["text"])
            input_ids = [encoding.ids for encoding in encoded]
            return {"input_ids": input_ids, "row_count": row_count}

        with mock.patch.object(loader, "build_tokenizer", return_value=_assert_tokenizer_batch):
            packed = loader.build_packed_dataset(
                dataset,
                tokenizer=_Tokenizer(),
                block_size=2,
                text_key="text",
                pack_batch_size=2,
                source_id=0,
            )

            next(iter(packed))

    def test_cleanup_shard_cache_skips_tmp_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "hf_shards"
            cache_root.mkdir(parents=True, exist_ok=True)
            tmp_file = cache_root / "shard.parquet.tmp"
            tmp_file.write_text("tmp")

            loader._cleanup_shard_cache(cache_root, [])

            self.assertTrue(tmp_file.exists())

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
