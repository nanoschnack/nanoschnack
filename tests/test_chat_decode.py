import types
import unittest

import torch

from model import setup_paths

setup_paths()

from model.chat import _build_byte_decoder, generate_reply_stream


class ChatDecodeTests(unittest.TestCase):
    """Exercise chat decoding fallbacks for invalid UTF-8 bytes.
    Simulates byte-level tokens that are not valid UTF-8.
    Ensures decoding falls back to tokenizer text without crashing.
    Uses a minimal dummy model and tokenizer for speed.
    """
    def test_generate_reply_stream_falls_back_on_decode_error(self):
        byte_decoder = _build_byte_decoder()
        token_char = next(key for key, value in byte_decoder.items() if value == 0x80)

        class _Tokenizer:
            def __init__(self, token):
                self._token = token
                self._vocab = {token: 0}

            def get_vocab(self):
                return self._vocab

            def encode(self, text):
                return types.SimpleNamespace(ids=[0])

            def decode(self, ids):
                return "fallback"

            def token_to_id(self, token):
                return self._vocab.get(token)

        class _Model(torch.nn.Module):
            def forward(self, input_ids):
                logits = torch.full((1, 1, 1), -1.0, device=input_ids.device)
                logits[..., 0] = 1.0
                return logits

        tokenizer = _Tokenizer(token_char)
        model = _Model()

        reply = "".join(
            generate_reply_stream(
                model,
                tokenizer,
                prompt="hi",
                context_len=8,
                max_new_tokens=1,
                temperature=0.0,
                top_k=0,
                device=torch.device("cpu"),
            )
        )

        self.assertEqual(reply, "fallback")


if __name__ == "__main__":
    unittest.main()
