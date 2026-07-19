import importlib
import os
from pathlib import Path
import unittest

from model import setup_paths

setup_paths()

from checkpointer import apply_checkpoint_config
import config
import tokenizer


class CheckpointTokenizerConfigTests(unittest.TestCase):
    def setUp(self):
        self._orig_tokenizer_filename = config.TOKENIZER_FILENAME

    def tearDown(self):
        config.TOKENIZER_FILENAME = self._orig_tokenizer_filename

    def test_apply_checkpoint_config_defaults_legacy_tokenizer_for_old_checkpoints(self):
        config.TOKENIZER_FILENAME = "tokenizer-v3.json"
        apply_checkpoint_config({"CONTEXT_LEN": config.CONTEXT_LEN})
        self.assertEqual(config.TOKENIZER_FILENAME, "tokenizer.json")

    def test_apply_checkpoint_config_respects_explicit_tokenizer_filename(self):
        apply_checkpoint_config({"TOKENIZER_FILENAME": "tokenizer-v3.json"})
        self.assertEqual(config.TOKENIZER_FILENAME, "tokenizer-v3.json")

    def test_tokenizer_module_uses_runtime_config_after_checkpoint_apply(self):
        config.TOKENIZER_FILENAME = "tokenizer-v3.json"
        apply_checkpoint_config({"TOKENIZER_FILENAME": "tokenizer.json"})

        resolved = tokenizer._resolve_tokenizer_path()

        self.assertEqual(config.TOKENIZER_FILENAME, "tokenizer.json")
        self.assertEqual(Path(resolved).name, "tokenizer.json")


class TokenizerDefaultConfigTests(unittest.TestCase):
    def test_default_tokenizer_filename_is_icelandic_v1(self):
        old_filename = os.environ.get("TOKENIZER_FILENAME")
        try:
            os.environ.pop("TOKENIZER_FILENAME", None)
            importlib.reload(config)
            self.assertEqual(config.TOKENIZER_FILENAME, "tokenizer-icelandic-v1.json")
        finally:
            if old_filename is None:
                os.environ.pop("TOKENIZER_FILENAME", None)
            else:
                os.environ["TOKENIZER_FILENAME"] = old_filename
            importlib.reload(config)


if __name__ == "__main__":
    unittest.main()
