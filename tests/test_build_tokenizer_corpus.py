import os
import tempfile
import unittest

import build_tokenizer_corpus


class BuildTokenizerCorpusTests(unittest.TestCase):
    """Validate corpus builder output on local txt inputs.

    Ensures both datasets contribute when space allows.
    """
    def test_build_corpus_two_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write two small input files.
            first_path = os.path.join(tmpdir, "a.txt")
            second_path = os.path.join(tmpdir, "b.txt")
            with open(first_path, "w", encoding="utf-8") as handle:
                handle.write("hallo welt\n")
            with open(second_path, "w", encoding="utf-8") as handle:
                handle.write("guten morgen\n")

            output_path = os.path.join(tmpdir, "corpus.txt")
            specs = f"txt:{first_path}:text,txt:{second_path}:text"
            build_tokenizer_corpus.build_corpus(
                output_path=output_path,
                target_size=128,
                specs_str=specs,
                cache_dir=None,
                seed=7,
                log_every=0,
            )

            with open(output_path, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]

        self.assertIn("hallo welt", lines)
        self.assertIn("guten morgen", lines)


if __name__ == "__main__":
    unittest.main()
