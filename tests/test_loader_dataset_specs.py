import tempfile
import unittest
from pathlib import Path

from loader import parse_dataset_specs, load_dataset_from_spec, resolve_total_rows


class DatasetSpecTests(unittest.TestCase):
    """Verify dataset spec parsing and local txt handling.

    Ensures txt sources are parsed, loaded, and counted correctly.
    Avoids network access by relying on temporary local files.
    """

    def test_parse_dataset_specs(self):
        specs = parse_dataset_specs(
            "hf:org/repo:train:text,txt:/tmp/goethe.txt:body"
        )
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0]["kind"], "hf")
        self.assertEqual(specs[0]["repo_id"], "org/repo")
        self.assertEqual(specs[0]["split"], "train")
        self.assertEqual(specs[0]["text_key"], "text")
        self.assertEqual(specs[1]["kind"], "txt")
        self.assertEqual(specs[1]["path"], "/tmp/goethe.txt")
        self.assertEqual(specs[1]["text_key"], "body")

    def test_load_dataset_from_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_text("Erste Zeile\n\nZweite Zeile\n", encoding="utf-8")
            spec = {"kind": "txt", "path": str(path), "split": "train", "text_key": "body"}
            dataset = load_dataset_from_spec(spec, streaming=False)
            self.assertIn("body", dataset.column_names)
            self.assertNotIn("text", dataset.column_names)
            total_rows = resolve_total_rows(dataset, spec)
            self.assertEqual(total_rows, 2)


if __name__ == "__main__":
    unittest.main()
