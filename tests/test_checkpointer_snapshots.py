import tempfile
import unittest

import torch

from checkpointer import Checkpointer, SNAPSHOT_INTERVALS
from gpt import GPT


class CheckpointerSnapshotTests(unittest.TestCase):
    """Ensure snapshot checkpoints are written alongside latest.

    Saves a checkpoint and verifies snapshot files are created.
    Uses a small model to keep runtime and memory low.
    """
    def test_save_latest_writes_snapshot_copies(self):
        model = GPT(
            vocab_size=11,
            embed_size=8,
            num_layers=1,
            num_heads=1,
            hidden_size=16,
            context_len=4,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = Checkpointer(tmpdir, model, optimizer, scheduler, device="cpu")
            checkpointer.save_latest(epoch=0, step=0, global_step=0, sample_index=0, total_tokens=0)

            for label, _ in SNAPSHOT_INTERVALS:
                snapshot_path = checkpointer._snapshot_path(label)
                self.assertTrue(snapshot_path.exists())


if __name__ == "__main__":
    unittest.main()
