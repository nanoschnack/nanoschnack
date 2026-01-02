import unittest
from unittest import mock

import torch

from model.gpt import RotaryEmbedding


class RotaryEmbeddingTests(unittest.TestCase):
    """Verify rotary embedding cache handling for compilation.
    Ensures torch.compile does not mutate cache state.
    Uses a minimal head size to keep CPU tests fast.
    """
    def test_rotary_cache_skips_compiling(self):
        if not hasattr(torch, "_dynamo"):
            self.skipTest("torch._dynamo is unavailable.")
        rope = RotaryEmbedding(4)
        q = torch.zeros(1, 1, 2, 4)
        k = torch.zeros(1, 1, 2, 4)

        with mock.patch.object(torch._dynamo, "is_compiling", return_value=True):
            rope(q, k)

        self.assertEqual(rope._cache, {})


if __name__ == "__main__":
    unittest.main()
