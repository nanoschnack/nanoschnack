import tempfile
import unittest
from pathlib import Path

from model import loader as loader_module


class ResumePlanTests(unittest.TestCase):
    """Validate row-offset resume planning for local text datasets.

    Uses globbed text shards to map a row offset to a shard and in-shard offset.
    Ensures the returned shard list starts at the correct file.
    """
    def test_resolve_resume_plan_for_txt_glob(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard1 = Path(tmpdir) / "shard_001.txt"
            shard2 = Path(tmpdir) / "shard_002.txt"
            shard1.write_text("a\nb\nc\n", encoding="utf-8")
            shard2.write_text("d\ne\nf\n", encoding="utf-8")
            spec = {
                "kind": "txt",
                "path": str(Path(tmpdir) / "shard_*.txt"),
                "split": "train",
                "text_key": "text",
                "spec": f"txt:{Path(tmpdir) / 'shard_*.txt'}:text",
            }

            data_files, in_shard_offset, shard_label = loader_module.resolve_resume_plan(
                spec,
                row_offset=4,
                cache_dir=None,
            )

            self.assertEqual(Path(data_files[0]).name, "shard_002.txt")
            self.assertEqual(in_shard_offset, 1)
            self.assertEqual(shard_label, data_files[0])


if __name__ == "__main__":
    unittest.main()
