import unittest

import torch

from scheduler import build_warmup_cosine_tokens


class TokenSchedulerTests(unittest.TestCase):
    """Verify token-driven warmup and cosine decay behavior.

    Uses a single parameter to expose optimizer LR changes.
    Confirms warmup reaches base LR and decay reaches the min LR.
    """
    def test_token_schedule_reaches_expected_levels(self):
        param = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([param], lr=1.0)
        scheduler = build_warmup_cosine_tokens(optimizer, total_tokens=100, warmup_pct=0.1)

        optimizer.step()
        scheduler.last_epoch = -1
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6, places=9)

        scheduler.last_epoch = 9
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0, places=6)

        scheduler.last_epoch = 99
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1, places=6)

    def test_token_schedule_clamps_after_total_tokens(self):
        param = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([param], lr=1.0)
        scheduler = build_warmup_cosine_tokens(optimizer, total_tokens=100, warmup_pct=0.1)

        scheduler.last_epoch = 149
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
