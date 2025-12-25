import importlib
import os
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


def _build_tokenizer():
    vocab = {
        "[PAD]": 0,
        "[BOS]": 1,
        "test": 2,
        "one": 3,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[PAD]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


class NanoSchnackTransparentLlmTestCase(unittest.TestCase):
    """Cover the adapter outputs needed for transparency tooling.

    Uses a tiny GPT config to keep tests fast on CPU.
    Avoids network by injecting a minimal tokenizer.
    Exercises key tensor shapes and residual relationships.
    """

    @classmethod
    def setUpClass(cls):
        os.environ["CONTEXT_LEN"] = "8"
        os.environ["VOCAB_SIZE"] = "0"
        os.environ["EMBED_SIZE"] = "16"
        os.environ["NUM_LAYERS"] = "2"
        os.environ["NUM_HEADS"] = "2"
        os.environ["HIDDEN_SIZE"] = "32"

        import config
        importlib.reload(config)

        import model.transparent_llm_adapter as adapter
        importlib.reload(adapter)

        cls.adapter = adapter
        cls.tokenizer = _build_tokenizer()
        cls.llm = adapter.NanoSchnackTransparentLlm(
            tokenizer=cls.tokenizer,
            device="cpu",
            prepend_bos=True,
            bos_token="[BOS]",
        )

    def setUp(self):
        self.llm.run(["test", "test one"])
        self.eps = 1e-5

    def test_tokens(self):
        tokens = self.llm.tokens()
        expected = [[1, 2, 0], [1, 2, 3]]
        self.assertEqual(tokens.tolist(), expected)

    def test_tokens_to_strings(self):
        tokens = torch.tensor([1, 2, 3], dtype=torch.long)
        strings = self.llm.tokens_to_strings(tokens)
        self.assertEqual(strings, ["[BOS]", "test", "one"])

    def test_residual_consistency(self):
        layer = 0
        batch = 0
        pos = 0

        resid_in = self.llm.residual_in(layer)[batch][pos]
        resid_mid = self.llm.residual_after_attn(layer)[batch][pos]
        resid_out = self.llm.residual_out(layer)[batch][pos]
        attn_out = self.llm.attention_output(batch, layer, pos)
        ffn_out = self.llm.ffn_out(layer)[batch][pos]

        diff_attn = torch.max(torch.abs(resid_mid - (resid_in + attn_out))).item()
        self.assertLess(diff_attn, self.eps)

        diff_ffn = torch.max(torch.abs(resid_out - (resid_mid + ffn_out))).item()
        self.assertLess(diff_ffn, self.eps)

    def test_tensor_shapes(self):
        vocab_size = 4
        n_batch = 2
        n_tokens = 3
        d_model = 16
        d_hidden = 32
        n_heads = 2
        layer = 1

        for name, tensor, expected_shape in [
            ("resid_in", self.llm.residual_in(layer), [n_batch, n_tokens, d_model]),
            ("resid_mid", self.llm.residual_after_attn(layer), [n_batch, n_tokens, d_model]),
            ("resid_out", self.llm.residual_out(layer), [n_batch, n_tokens, d_model]),
            ("logits", self.llm.logits(), [n_batch, n_tokens, vocab_size]),
            ("ffn_out", self.llm.ffn_out(layer), [n_batch, n_tokens, d_model]),
            (
                "decomposed_ffn_out",
                self.llm.decomposed_ffn_out(0, layer, 0),
                [d_hidden, d_model],
            ),
            ("neuron_activations", self.llm.neuron_activations(0, layer, 0), [d_hidden]),
            ("neuron_output", self.llm.neuron_output(layer, 0), [d_model]),
            (
                "attention_matrix",
                self.llm.attention_matrix(0, layer, 0),
                [n_tokens, n_tokens],
            ),
            (
                "attention_output",
                self.llm.attention_output(0, layer, 0),
                [d_model],
            ),
            (
                "attention_output_per_head",
                self.llm.attention_output_per_head(0, layer, 0, 0),
                [d_model],
            ),
            (
                "decomposed_attn",
                self.llm.decomposed_attn(0, layer),
                [n_tokens, n_tokens, n_heads, d_model],
            ),
            (
                "unembed",
                self.llm.unembed(torch.zeros([d_model]), normalize=True),
                [vocab_size],
            ),
        ]:
            self.assertEqual(list(tensor.shape), expected_shape, name)
            self.assertFalse(torch.any(tensor.isnan()), name)


if __name__ == "__main__":
    unittest.main()
