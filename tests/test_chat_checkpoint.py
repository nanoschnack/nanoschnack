import os
import tempfile
import unittest

import torch

from model import setup_paths

setup_paths()

import chat as chat_module
import config
from gpt import GPT


class ChatCheckpointTests(unittest.TestCase):
    """Validate chat checkpoint compatibility helpers.
    Covers legacy key prefixes and alternate checkpoint layouts.
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
        }

    def tearDown(self):
        for name, value in self._orig_config.items():
            setattr(config, name, value)

    def test_load_model_strips_module_prefix(self):
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
        state_dict = {f"module.{k}": v for k, v in model.state_dict().items()}
        ckpt = {
            "model": state_dict,
            "config": config.snapshot(),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            torch.save(ckpt, path)
            loaded_model, context_len = chat_module.load_model(
                path,
                vocab_size=11,
                device=torch.device("cpu"),
            )

        self.assertEqual(context_len, config.CONTEXT_LEN)
        self.assertTrue(torch.equal(loaded_model.tok.weight, model.tok.weight))

    def test_load_model_accepts_model_state_dict(self):
        config.CONTEXT_LEN = 6
        config.EMBED_SIZE = 12
        config.NUM_LAYERS = 2
        config.NUM_HEADS = 2
        config.HIDDEN_SIZE = 24

        model = GPT(
            vocab_size=13,
            embed_size=config.EMBED_SIZE,
            num_layers=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            hidden_size=config.HIDDEN_SIZE,
            context_len=config.CONTEXT_LEN,
            pos_embed_type=config.POS_EMBED_TYPE,
            rope_base=config.ROPE_BASE,
        )
        ckpt = {
            "model_state_dict": model.state_dict(),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            torch.save(ckpt, path)
            loaded_model, context_len = chat_module.load_model(
                path,
                vocab_size=13,
                device=torch.device("cpu"),
            )

        self.assertEqual(context_len, config.CONTEXT_LEN)
        self.assertTrue(torch.equal(loaded_model.lm.weight, model.lm.weight))

    def test_load_model_strips_orig_mod_prefix(self):
        config.CONTEXT_LEN = 5
        config.EMBED_SIZE = 10
        config.NUM_LAYERS = 1
        config.NUM_HEADS = 2
        config.HIDDEN_SIZE = 20
        config.POS_EMBED_TYPE = "learned"

        model = GPT(
            vocab_size=7,
            embed_size=config.EMBED_SIZE,
            num_layers=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            hidden_size=config.HIDDEN_SIZE,
            context_len=config.CONTEXT_LEN,
            pos_embed_type=config.POS_EMBED_TYPE,
            rope_base=config.ROPE_BASE,
        )
        state_dict = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
        ckpt = {
            "model": state_dict,
            "config": config.snapshot(),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            torch.save(ckpt, path)
            loaded_model, context_len = chat_module.load_model(
                path,
                vocab_size=7,
                device=torch.device("cpu"),
            )

        self.assertEqual(context_len, config.CONTEXT_LEN)
        self.assertTrue(torch.equal(loaded_model.ln.weight, model.ln.weight))

    def test_load_model_expands_vocab(self):
        config.CONTEXT_LEN = 4
        config.EMBED_SIZE = 8
        config.NUM_LAYERS = 1
        config.NUM_HEADS = 1
        config.HIDDEN_SIZE = 16

        torch.manual_seed(1234)
        model_small = GPT(
            vocab_size=8,
            embed_size=config.EMBED_SIZE,
            num_layers=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            hidden_size=config.HIDDEN_SIZE,
            context_len=config.CONTEXT_LEN,
            pos_embed_type=config.POS_EMBED_TYPE,
            rope_base=config.ROPE_BASE,
        )
        ckpt = {
            "model": model_small.state_dict(),
            "config": config.snapshot(),
            "vocab_size": 8,
        }

        torch.manual_seed(5678)
        expected_large = GPT(
            vocab_size=12,
            embed_size=config.EMBED_SIZE,
            num_layers=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            hidden_size=config.HIDDEN_SIZE,
            context_len=config.CONTEXT_LEN,
            pos_embed_type=config.POS_EMBED_TYPE,
            rope_base=config.ROPE_BASE,
        )
        expected_tail = expected_large.tok.weight.detach().clone()[8:]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            torch.save(ckpt, path)
            torch.manual_seed(5678)
            loaded_model, context_len = chat_module.load_model(
                path,
                vocab_size=12,
                device=torch.device("cpu"),
            )

        self.assertEqual(context_len, config.CONTEXT_LEN)
        self.assertTrue(torch.equal(loaded_model.tok.weight[:8], model_small.tok.weight))
        self.assertTrue(torch.equal(loaded_model.tok.weight[8:], expected_tail))


if __name__ == "__main__":
    unittest.main()
