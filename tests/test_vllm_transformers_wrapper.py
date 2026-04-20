import tempfile
import unittest
from pathlib import Path

import torch

try:
    import transformers  # noqa: F401
except ImportError as exc:
    raise unittest.SkipTest("transformers not installed") from exc

from model import config as base_config
from model.gpt import GPT
from vllm_model.configuration_nanoschnack import NanoSchnackConfig
from vllm_model.modeling_nanoschnack import NanoSchnackForCausalLM


def _snapshot_base_config():
    return {
        "CONTEXT_LEN": base_config.CONTEXT_LEN,
        "VOCAB_SIZE": base_config.VOCAB_SIZE,
        "EMBED_SIZE": base_config.EMBED_SIZE,
        "NUM_LAYERS": base_config.NUM_LAYERS,
        "NUM_HEADS": base_config.NUM_HEADS,
        "HIDDEN_SIZE": base_config.HIDDEN_SIZE,
    }


def _restore_base_config(snapshot):
    for name, value in snapshot.items():
        setattr(base_config, name, value)


class VllmTransformersWrapperTest(unittest.TestCase):
    """Covers the minimal Transformers wrapper for NanoSchnack.

    Builds a tiny checkpoint and ensures the wrapper loads it.
    Verifies logits shape for a short forward pass.
    """
    def test_loads_checkpoint_and_runs_forward(self):
        snapshot = _snapshot_base_config()
        try:
            base_config.CONTEXT_LEN = 8
            base_config.VOCAB_SIZE = 32
            base_config.EMBED_SIZE = 16
            base_config.NUM_LAYERS = 2
            base_config.NUM_HEADS = 2
            base_config.HIDDEN_SIZE = 64
            model = GPT(
                vocab_size=base_config.VOCAB_SIZE,
                embed_size=base_config.EMBED_SIZE,
                num_layers=base_config.NUM_LAYERS,
                num_heads=base_config.NUM_HEADS,
                hidden_size=base_config.HIDDEN_SIZE,
                context_len=base_config.CONTEXT_LEN,
            )
            ckpt = {
                "model": model.state_dict(),
                "config": base_config.snapshot(),
                "vocab_size": base_config.VOCAB_SIZE,
            }
            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_path = Path(tmpdir) / "tiny.pt"
                torch.save(ckpt, ckpt_path)
                config = NanoSchnackConfig(
                    context_len=base_config.CONTEXT_LEN,
                    vocab_size=base_config.VOCAB_SIZE,
                    embed_size=base_config.EMBED_SIZE,
                    num_layers=base_config.NUM_LAYERS,
                    num_heads=base_config.NUM_HEADS,
                    hidden_size=base_config.HIDDEN_SIZE,
                    checkpoint_path=str(ckpt_path),
                )
                wrapper = NanoSchnackForCausalLM.from_pretrained(tmpdir, config=config)
                input_ids = torch.randint(0, base_config.VOCAB_SIZE, (1, 4))
                output = wrapper(input_ids)
                self.assertEqual(output.logits.shape, (1, 4, base_config.VOCAB_SIZE))
        finally:
            _restore_base_config(snapshot)
