import tempfile
import unittest

import torch

from checkpointer import Checkpointer
from gpt import GPT


class CheckpointerVocabResizeTests(unittest.TestCase):
    """Ensure checkpoints with smaller vocab expand cleanly.

    Loads a checkpoint with fewer token rows into a larger model.
    Validates copied rows and preserved initialization for new rows.
    Confirms tied weights remain aligned after loading.
    """
    def test_load_latest_expands_vocab_weights(self):
        model_small = GPT(
            vocab_size=8,
            embed_size=8,
            num_layers=1,
            num_heads=1,
            hidden_size=16,
            context_len=4,
            pos_embed_type="learned",
        )
        optimizer_small = torch.optim.AdamW(model_small.parameters(), lr=1e-3)
        scheduler_small = torch.optim.lr_scheduler.LambdaLR(optimizer_small, lr_lambda=lambda _: 1.0)
        ckpt = {
            "model": model_small.state_dict(),
            "optimizer": optimizer_small.state_dict(),
            "scheduler": scheduler_small.state_dict(),
            "epoch": 1,
            "step": 2,
            "global_step": 2,
            "sample_index": 3,
            "total_tokens": 4,
            "vocab_size": 8,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/latest.pt"
            torch.save(ckpt, path)
            model_large = GPT(
                vocab_size=12,
                embed_size=8,
                num_layers=1,
                num_heads=1,
                hidden_size=16,
                context_len=4,
                pos_embed_type="learned",
            )
            initial_weights = model_large.tok.weight.detach().clone()
            optimizer_large = torch.optim.AdamW(model_large.parameters(), lr=1e-3)
            scheduler_large = torch.optim.lr_scheduler.LambdaLR(optimizer_large, lr_lambda=lambda _: 1.0)
            checkpointer = Checkpointer(tmpdir, model_large, optimizer_large, scheduler_large, device="cpu")
            checkpointer.load_latest()

        self.assertTrue(torch.equal(model_large.tok.weight[:8], model_small.tok.weight))
        self.assertTrue(torch.equal(model_large.tok.weight[8:], initial_weights[8:]))
        self.assertTrue(torch.equal(model_large.lm.weight, model_large.tok.weight))


if __name__ == "__main__":
    unittest.main()
