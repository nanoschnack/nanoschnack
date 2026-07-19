import os
import tempfile
import unittest
from unittest import mock

from scripts import build_icelandic_sagas


class BuildIcelandicSagasTests(unittest.TestCase):
    """Validate Saga Corpus materialization without network access.

    Ensures empty records are dropped and source paragraphs are preserved.
    Confirms callers receive the number of written saga documents.
    """
    def test_build_saga_corpus(self):
        # Supply representative records without downloading the real parquet file.
        dataset = [
            {"text": "Fyrsti kafli.\n\nAnnar kafli."},
            {"text": "  "},
            {"text": "Lok."},
        ]

        # Materialize the records and verify the plain-text boundary format.
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "sagas.txt")
            with mock.patch.object(build_icelandic_sagas, "load_dataset", return_value=dataset):
                count = build_icelandic_sagas.build_saga_corpus(output_path, cache_dir="cache")
            with open(output_path, "r", encoding="utf-8") as handle:
                output = handle.read()

        self.assertEqual(count, 2)
        self.assertEqual(output, "Fyrsti kafli.\n\nAnnar kafli.\n\nLok.\n\n")


if __name__ == "__main__":
    unittest.main()
