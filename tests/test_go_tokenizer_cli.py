import os
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which("go"), "go toolchain not available")
class GoTokenizerCliTests(unittest.TestCase):
    """Exercise the Go tokenizer CLI over stdin.
    Ensures flags are accepted and output is produced.
    Skips when the Go toolchain is unavailable.
    """
    def test_stdin_train_and_output_file(self):
        stdin_text = "hallo welt\n\nhallo welt\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.txt")
            cmd = [
                "go",
                "-C",
                "tokenizer",
                "run",
                ".",
                "--target",
                "260",
                "-f",
                out_path,
            ]
            subprocess.run(
                cmd,
                input=stdin_text,
                text=True,
                capture_output=True,
                check=True,
            )

            with open(out_path, "r", encoding="utf-8") as handle:
                data = handle.read().strip()

        self.assertTrue(data)
        self.assertRegex(data, r"^\d+( \d+)*$")


if __name__ == "__main__":
    unittest.main()
