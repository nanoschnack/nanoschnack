import math
import unittest

import torch

from gpt import GPT


class GPTInitializationTests(unittest.TestCase):
    """Validate GPT-2 initialization conventions for the GPT module.
    Uses a deterministic seed to keep statistical checks stable.
    Verifies base weights, residual scaling, and normalization defaults.
    Keeps sizes modest to stay fast on CPU while reducing sampling noise.
    """
    def _assert_stats(self, tensor, expected_std, mean_tol=0.002, std_tol=0.002):
        # Summarize distribution stats for init checks.
        mean = tensor.mean().item()
        std = tensor.std(unbiased=False).item()
        self.assertLess(abs(mean), mean_tol, f"mean {mean} exceeds tolerance {mean_tol}")
        self.assertLess(abs(std - expected_std), std_tol, f"std {std} differs from {expected_std}")

    def test_gpt2_initialization_values(self):
        # Fix seed for stable statistics in init checks.
        torch.manual_seed(1234)
        num_layers = 2
        model = GPT(
            vocab_size=128,
            embed_size=64,
            num_layers=num_layers,
            num_heads=4,
            hidden_size=256,
            context_len=32,
        )

        # Check base GPT-2 std on embeddings and non-residual projections.
        base_std = 0.02
        self._assert_stats(model.tok.weight, base_std)
        self._assert_stats(model.blocks[0].attn.qkv.weight, base_std)
        self._assert_stats(model.blocks[0].mlp.input.weight, base_std)
        self.assertTrue(torch.equal(model.lm.weight, model.tok.weight))

        # Check residual scaling plus zero/one defaults on biases and LayerNorm.
        scaled_std = base_std / math.sqrt(2 * num_layers)
        for block in model.blocks:
            self._assert_stats(block.attn.proj.weight, scaled_std, std_tol=0.0015)
            self._assert_stats(block.mlp.output.weight, scaled_std, std_tol=0.0015)
            self.assertTrue(torch.all(block.attn.qkv.bias == 0))
            self.assertTrue(torch.all(block.attn.proj.bias == 0))
            self.assertTrue(torch.all(block.mlp.input.bias == 0))
            self.assertTrue(torch.all(block.mlp.output.bias == 0))
            self.assertTrue(torch.allclose(block.ln1.weight, torch.ones_like(block.ln1.weight)))
            self.assertTrue(torch.all(block.ln1.bias == 0))
            self.assertTrue(torch.allclose(block.ln2.weight, torch.ones_like(block.ln2.weight)))
            self.assertTrue(torch.all(block.ln2.bias == 0))
        self.assertTrue(torch.allclose(model.ln.weight, torch.ones_like(model.ln.weight)))
        self.assertTrue(torch.all(model.ln.bias == 0))


if __name__ == "__main__":
    unittest.main()
