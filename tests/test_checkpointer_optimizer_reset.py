import tempfile
import unittest

import torch

from checkpointer import Checkpointer
from gpt import GPT


class CheckpointerOptimizerResetTests(unittest.TestCase):
    """Ensure optimizer mismatches raise instead of resetting state.

    Uses mismatched optimizer state to trigger the failure path.
    Verifies loading fails loudly instead of continuing.
    """
    def test_load_latest_raises_on_mismatched_optimizer_state(self):
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
            with self.assertRaises(RuntimeError):
                checkpointer.load_latest()


if __name__ == "__main__":
    unittest.main()
