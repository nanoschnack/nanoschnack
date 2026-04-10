import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from model import setup_paths

setup_paths(Path(__file__).resolve().parent.parent / "model")

from chat import _build_byte_decoder


def _decode_vocab_token(token):
    # Decode byte-level GPT-2 token strings into readable UTF-8 text.
    byte_decoder = _build_byte_decoder()
    payload = bytes(byte_decoder[ch] for ch in token)
    return payload.decode("utf-8", errors="replace")


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

    def test_stdin_umlauts_round_trip_without_zero_byte_corruption(self):
        stdin_text = "für für für\nKäse Käse Käse\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "tokenizer.json")
            cmd = [
                "go",
                "-C",
                "tokenizer",
                "run",
                ".",
                "--target",
                "270",
                "-f",
                json_path,
                "--top",
                "4",
                "--in",
                "für Käse",
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

        decoded_vocab = {_decode_vocab_token(token) for token in payload["model"]["vocab"]}

        self.assertIn("für", decoded_vocab)
        self.assertIn("Käse", decoded_vocab)
        self.assertNotIn("\x00", result.stdout)
        self.assertNotIn("\\xc3\\x00", result.stdout)


if __name__ == "__main__":
    unittest.main()
