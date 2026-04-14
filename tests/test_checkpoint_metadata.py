import os
import tempfile
import unittest

import torch

from model import setup_paths

setup_paths()

import config
from checkpoint_metadata import load_checkpoint_metadata
from gpt import GPT


class CheckpointMetadataTests(unittest.TestCase):
    """Validate the shared checkpoint metadata loader contract.

    Keeps checkpoint loading semantics centralized for chat and inference.
    Covers legacy defaults and state-dict prefix normalization.
    Uses small models to keep runtime and memory low.
    """

    def setUp(self):
        self._orig_config = {
            "CONTEXT_LEN": config.CONTEXT_LEN,
            "EMBED_SIZE": config.EMBED_SIZE,
            "POS_EMBED_TYPE": config.POS_EMBED_TYPE,
            "ROPE_BASE": config.ROPE_BASE,
            "NUM_LAYERS": config.NUM_LAYERS,
            "NUM_HEADS": config.NUM_HEADS,
            "HIDDEN_SIZE": config.HIDDEN_SIZE,
            "TOKENIZER_FILENAME": config.TOKENIZER_FILENAME,
        }

    def tearDown(self):
        for name, value in self._orig_config.items():
            setattr(config, name, value)

    def test_load_checkpoint_metadata_defaults_legacy_fields(self):
        ckpt = {
            "model": {"tok.weight": torch.zeros(7, 8)},
            "config": {"CONTEXT_LEN": 32},
            "vocab_size": 7,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            torch.save(ckpt, path)
            metadata = load_checkpoint_metadata(path, apply_config=True)

        self.assertEqual(metadata.config["TOKENIZER_FILENAME"], "tokenizer.json")
        self.assertEqual(metadata.config["POS_EMBED_TYPE"], "learned")
        self.assertEqual(metadata.vocab_size, 7)
        self.assertEqual(config.TOKENIZER_FILENAME, "tokenizer.json")
        self.assertEqual(config.POS_EMBED_TYPE, "learned")

    def test_load_checkpoint_metadata_normalizes_state_dict_prefixes(self):
        config.CONTEXT_LEN = 4
        config.EMBED_SIZE = 8
        config.NUM_LAYERS = 1
        config.NUM_HEADS = 1
        config.HIDDEN_SIZE = 16

        model = GPT(
            vocab_size=11,
            embed_size=config.EMBED_SIZE,
            num_layers=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            hidden_size=config.HIDDEN_SIZE,
            context_len=config.CONTEXT_LEN,
            pos_embed_type=config.POS_EMBED_TYPE,
            rope_base=config.ROPE_BASE,
        )
        ckpt = {
            "model": {f"_orig_mod.{k}": v for k, v in model.state_dict().items()},
            "config": config.snapshot(),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            torch.save(ckpt, path)
            metadata = load_checkpoint_metadata(path)

        self.assertIn("tok.weight", metadata.state_dict)
        self.assertNotIn("_orig_mod.tok.weight", metadata.state_dict)
        self.assertTrue(torch.equal(metadata.state_dict["tok.weight"], model.tok.weight))


if __name__ == "__main__":
    unittest.main()
