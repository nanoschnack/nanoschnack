import importlib
import os
from pathlib import Path
import tempfile
import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


class TokenizerLoadPathTests(unittest.TestCase):
    """Verify tokenizer loading respects explicit paths and versioned defaults.

    Builds a temporary tokenizer.json and ensures it is used.
    Keeps VOCAB_SIZE at zero to avoid down-sizing the vocab.
    """
    def test_load_tokenizer_uses_local_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer_path = os.path.join(tmpdir, "tokenizer.json")
            tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hallo": 1}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.save(tokenizer_path)

            # Swap env vars before reloading config/tokenizer modules.
            old_tokenizer_path = os.environ.get("TOKENIZER_JSON_PATH")
            old_vocab_size = os.environ.get("VOCAB_SIZE")
            os.environ["TOKENIZER_JSON_PATH"] = tokenizer_path
            os.environ["VOCAB_SIZE"] = "0"
            try:
                config = importlib.import_module("config")
                tokenizer_module = importlib.import_module("tokenizer")
                importlib.reload(config)
                importlib.reload(tokenizer_module)
                loaded = tokenizer_module.load_tokenizer()
            finally:
                if old_tokenizer_path is None:
                    os.environ.pop("TOKENIZER_JSON_PATH", None)
                else:
                    os.environ["TOKENIZER_JSON_PATH"] = old_tokenizer_path
                if old_vocab_size is None:
                    os.environ.pop("VOCAB_SIZE", None)
                else:
                    os.environ["VOCAB_SIZE"] = old_vocab_size
                config = importlib.import_module("config")
                tokenizer_module = importlib.import_module("tokenizer")
                importlib.reload(config)
                importlib.reload(tokenizer_module)

        self.assertIsNotNone(loaded.token_to_id("hallo"))

    def test_load_tokenizer_raises_when_configured_versioned_file_is_missing(self):
        old_tokenizer_filename = os.environ.get("TOKENIZER_FILENAME")
        old_tokenizer_path = os.environ.get("TOKENIZER_JSON_PATH")
        old_vocab_size = os.environ.get("VOCAB_SIZE")
        try:
            os.environ.pop("TOKENIZER_JSON_PATH", None)
            os.environ["TOKENIZER_FILENAME"] = "missing-tokenizer-v3.json"
            os.environ["VOCAB_SIZE"] = "0"
            config = importlib.import_module("config")
            tokenizer_module = importlib.import_module("tokenizer")
            importlib.reload(config)
            importlib.reload(tokenizer_module)
            with self.assertRaises(FileNotFoundError) as ctx:
                tokenizer_module.load_tokenizer()
            self.assertIn(str(Path("tokenizer") / "missing-tokenizer-v3.json"), str(ctx.exception))
        finally:
            if old_tokenizer_filename is None:
                os.environ.pop("TOKENIZER_FILENAME", None)
            else:
                os.environ["TOKENIZER_FILENAME"] = old_tokenizer_filename
            if old_tokenizer_path is None:
                os.environ.pop("TOKENIZER_JSON_PATH", None)
            else:
                os.environ["TOKENIZER_JSON_PATH"] = old_tokenizer_path
            if old_vocab_size is None:
                os.environ.pop("VOCAB_SIZE", None)
            else:
                os.environ["VOCAB_SIZE"] = old_vocab_size
            config = importlib.import_module("config")
            tokenizer_module = importlib.import_module("tokenizer")
            importlib.reload(config)
            importlib.reload(tokenizer_module)


if __name__ == "__main__":
    unittest.main()
