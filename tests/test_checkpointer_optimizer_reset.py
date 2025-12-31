import tempfile
import unittest

import torch

from checkpointer import Checkpointer, _load_optimizer_state
from gpt import GPT


class CheckpointerOptimizerResetTests(unittest.TestCase):
    """Ensure optimizer mismatches fall back to fresh state.

    Uses mismatched optimizer state to trigger the fallback path.
    Verifies loading continues with an empty optimizer state.
    """
    def test_load_latest_resets_on_mismatched_optimizer_state(self):
        model_a = GPT(
            vocab_size=11,
            embed_size=8,
            num_layers=1,
            num_heads=1,
            hidden_size=16,
            context_len=4,
        )
        optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
        scheduler_a = torch.optim.lr_scheduler.LambdaLR(optimizer_a, lr_lambda=lambda _: 1.0)
        # Populate optimizer state with a dummy step.
        loss = sum(param.sum() for param in model_a.parameters())
        loss.backward()
        optimizer_a.step()

        model_b = GPT(
            vocab_size=11,
            embed_size=12,
            num_layers=1,
            num_heads=1,
            hidden_size=24,
            context_len=4,
        )
        optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
        scheduler_b = torch.optim.lr_scheduler.LambdaLR(optimizer_b, lr_lambda=lambda _: 1.0)
        ckpt = {
            "model": model_b.state_dict(),
            "optimizer": optimizer_a.state_dict(),
            "scheduler": scheduler_a.state_dict(),
            "epoch": 1,
            "step": 1,
            "global_step": 1,
            "sample_index": 0,
            "total_tokens": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/latest.pt"
            torch.save(ckpt, path)
            checkpointer = Checkpointer(tmpdir, model_b, optimizer_b, scheduler_b, device="cpu")
            checkpointer.load_latest()

        self.assertEqual(len(optimizer_b.state), 0)

    def test_load_optimizer_state_reports_shape_mismatch(self):
        model = GPT(
            vocab_size=11,
            embed_size=8,
            num_layers=1,
            num_heads=1,
            hidden_size=16,
            context_len=4,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = sum(param.sum() for param in model.parameters())
        loss.backward()
        optimizer.step()
        state_dict = optimizer.state_dict()
        state = next(iter(state_dict["state"].values()))
        tensor_key = next(key for key, value in state.items() if torch.is_tensor(value))
        state[tensor_key] = torch.zeros(1, dtype=state[tensor_key].dtype)

        loaded, info = _load_optimizer_state(optimizer, state_dict)

        self.assertFalse(loaded)
        self.assertEqual(info.get("reason"), "shape_mismatch")
        self.assertTrue(info.get("mismatches"))

    def test_load_optimizer_state_accepts_scalar_state(self):
        model = GPT(
            vocab_size=11,
            embed_size=8,
            num_layers=1,
            num_heads=1,
            hidden_size=16,
            context_len=4,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = sum(param.sum() for param in model.parameters())
        loss.backward()
        optimizer.step()
        state_dict = optimizer.state_dict()
        optimizer_fresh = torch.optim.AdamW(model.parameters(), lr=1e-3)

        loaded, info = _load_optimizer_state(optimizer_fresh, state_dict)

        self.assertTrue(loaded)
        self.assertEqual(info, {})


if __name__ == "__main__":
    unittest.main()
