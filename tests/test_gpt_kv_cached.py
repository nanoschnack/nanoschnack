import unittest

import torch

from model import setup_paths

setup_paths()

from gpt import GPT
from gpt_kv_cached import GPTKVCached


class GPTKVCachedTests(unittest.TestCase):
    """Validate the PyTorch KV-cached reference path against baseline GPT.

    Uses modest model sizes to keep parity checks fast on CPU.
    Covers prompt prefill and incremental decode growth.
    Keeps dropout disabled to make logits comparisons exact.
    """

    _ATOL = 1e-6
    _RTOL = 1e-6

    def _build_models(self, pos_embed_type="rope"):
        torch.manual_seed(1234)
        base = GPT(
            vocab_size=32,
            embed_size=16,
            num_layers=2,
            num_heads=4,
            hidden_size=32,
            context_len=16,
            dropout=0.0,
            pos_embed_type=pos_embed_type,
        )
        cached = GPTKVCached(
            vocab_size=32,
            embed_size=16,
            num_layers=2,
            num_heads=4,
            hidden_size=32,
            context_len=16,
            dropout=0.0,
            pos_embed_type=pos_embed_type,
        )
        cached.load_state_dict(base.state_dict())
        base.eval()
        cached.eval()
        return base, cached

    def test_prefill_matches_full_forward_for_rope(self):
        base, cached = self._build_models(pos_embed_type="rope")
        tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)

        expected = base(tokens)
        actual, cache = cached.forward_cached(tokens)

        self.assertTrue(torch.allclose(actual, expected, atol=self._ATOL, rtol=self._RTOL))
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache[0].key.shape, (2, 4, 4, 4))
        self.assertEqual(cache[0].value.shape, (2, 4, 4, 4))

    def test_full_forward_matches_plain_gpt(self):
        base, cached = self._build_models(pos_embed_type="learned")
        tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)

        expected = base(tokens)
        actual = cached(tokens)

        self.assertTrue(torch.allclose(actual, expected, atol=self._ATOL, rtol=self._RTOL))

    def test_incremental_decode_matches_last_token_logits(self):
        base, cached = self._build_models(pos_embed_type="rope")
        prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        next_token = torch.tensor([[5]], dtype=torch.long)

        _, cache = cached.forward_cached(prompt)
        cached_logits, next_cache = cached.forward_cached(next_token, cache)
        expected_logits = base(torch.cat((prompt, next_token), dim=1))[:, -1:, :]

        self.assertTrue(torch.allclose(cached_logits, expected_logits, atol=self._ATOL, rtol=self._RTOL))
        self.assertEqual(next_cache[0].seq_len, 5)

    def test_incremental_decode_matches_multiple_single_token_steps(self):
        base, cached = self._build_models(pos_embed_type="learned")
        full = torch.tensor([[7, 8, 9, 10, 11]], dtype=torch.long)

        cache = None
        for end in range(1, full.shape[1] + 1):
            token = full[:, end - 1:end]
            logits, cache = cached.forward_cached(token, cache)
            expected = base(full[:, :end])[:, -1:, :]
            self.assertTrue(torch.allclose(logits, expected, atol=self._ATOL, rtol=self._RTOL))
            self.assertEqual(cache[0].seq_len, end)


if __name__ == "__main__":
    unittest.main()
