import json
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
            json_path = os.path.join(tmpdir, "tokenizer.json")
            cmd = [
                "go",
                "-C",
                "tokenizer",
                "run",
                ".",
                "--target",
                "260",
                "-f",
                json_path,
                "--in",
                "hallo welt",
            ]
            result = subprocess.run(
                cmd,
                input=stdin_text,
                text=True,
                capture_output=True,
                check=True,
            )

            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

        stdout = result.stdout.strip()
        self.assertTrue(stdout)
        self.assertRegex(stdout, r"^\d+( \d+)*$")

        self.assertEqual(payload["model"]["type"], "BPE")
        self.assertEqual(payload["pre_tokenizer"]["type"], "Sequence")

    def test_stdin_top_tokens(self):
        stdin_text = "hallo welt\n\nhallo welt\n"
        cmd = [
            "go",
            "-C",
            "tokenizer",
            "run",
            ".",
            "--target",
            "260",
            "--top",
            "3",
        ]
        result = subprocess.run(
            cmd,
            input=stdin_text,
            text=True,
            capture_output=True,
            check=True,
        )

        lines = [line for line in result.stdout.splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 3)
        self.assertRegex(lines[0], r"^\d+: .*\(count=\d+\)$")


if __name__ == "__main__":
    unittest.main()
