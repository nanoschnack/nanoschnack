import tempfile
import unittest

import torch

from checkpointer import Checkpointer
from gpt import GPT


class CheckpointerLegacyTests(unittest.TestCase):
    """Validate legacy checkpoint remapping for TransformerEncoder blocks.

    Builds a legacy-style state dict and verifies remapped model weights.
    Uses a small GPT for speed and deterministic checks.
    """
    def _legacy_state_from_model(self, model):
        # Convert current GPT keys into legacy TransformerEncoder equivalents.
        legacy = {}
        for key, value in model.state_dict().items():
            if not key.startswith("blocks."):
                legacy[key] = value
                continue
            parts = key.split(".")
            layer_index = parts[1]
            suffix = ".".join(parts[2:])
            mapping = {
                "ln1.weight": f"blocks.layers.{layer_index}.norm1.weight",
                "ln1.bias": f"blocks.layers.{layer_index}.norm1.bias",
                "ln2.weight": f"blocks.layers.{layer_index}.norm2.weight",
                "ln2.bias": f"blocks.layers.{layer_index}.norm2.bias",
                "attn.qkv.weight": f"blocks.layers.{layer_index}.self_attn.in_proj_weight",
                "attn.qkv.bias": f"blocks.layers.{layer_index}.self_attn.in_proj_bias",
                "attn.proj.weight": f"blocks.layers.{layer_index}.self_attn.out_proj.weight",
                "attn.proj.bias": f"blocks.layers.{layer_index}.self_attn.out_proj.bias",
                "mlp.input.weight": f"blocks.layers.{layer_index}.linear1.weight",
                "mlp.input.bias": f"blocks.layers.{layer_index}.linear1.bias",
                "mlp.output.weight": f"blocks.layers.{layer_index}.linear2.weight",
                "mlp.output.bias": f"blocks.layers.{layer_index}.linear2.bias",
            }
            legacy_key = mapping.get(suffix)
            if legacy_key:
                legacy[legacy_key] = value
        return legacy

    def test_load_latest_remaps_legacy_transformer_encoder(self):
        model = GPT(
            vocab_size=11,
            embed_size=8,
            num_layers=2,
            num_heads=2,
            hidden_size=16,
            context_len=4,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        legacy_state = self._legacy_state_from_model(model)
        ckpt = {
            "model": legacy_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": 1,
            "step": 2,
            "global_step": 2,
            "sample_index": 3,
            "total_tokens": 4,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/latest.pt"
            torch.save(ckpt, path)
            new_model = GPT(
                vocab_size=11,
                embed_size=8,
                num_layers=2,
                num_heads=2,
                hidden_size=16,
                context_len=4,
            )
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
            new_scheduler = torch.optim.lr_scheduler.LambdaLR(new_optimizer, lr_lambda=lambda _: 1.0)
            checkpointer = Checkpointer(tmpdir, new_model, new_optimizer, new_scheduler, device="cpu")
            resume_epoch, resume_step, global_step, sample_index, total_tokens, resume_state = checkpointer.load_latest()

        self.assertEqual(resume_epoch, 0)
        self.assertEqual(resume_step, 2)
        self.assertEqual(global_step, 2)
        self.assertEqual(sample_index, 3)
        self.assertEqual(total_tokens, 4)
        self.assertIsNone(resume_state)
        self.assertTrue(torch.equal(new_model.tok.weight, model.tok.weight))
        self.assertTrue(torch.equal(new_model.blocks[0].attn.qkv.weight, model.blocks[0].attn.qkv.weight))


if __name__ == "__main__":
    unittest.main()
