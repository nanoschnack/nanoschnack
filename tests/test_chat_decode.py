import types
import unittest

import torch

from model import setup_paths

setup_paths()

import model.chat as chat_module
from model.chat import (
    _build_byte_decoder,
    _build_prompt,
    _cache_status_label,
    _checkpoint_status_label,
    DEFAULT_SAMPLE_PROMPT,
    _handle_repl_command,
    _sample_replies,
    generate_reply_stream,
)


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

    def test_generate_reply_stream_uses_cached_decode_when_enabled(self):
        class _Tokenizer:
            def get_vocab(self):
                return {"a": 0, "<|EOS|>": 1}

            def encode(self, text):
                return types.SimpleNamespace(ids=[0])

            def decode(self, ids):
                return "a"

            def token_to_id(self, token):
                return {"<|EOS|>": 1}.get(token)

        class _CachedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, input_ids):
                raise AssertionError("full forward should not be used when cache is enabled")

            def forward_cached(self, input_ids, cache=None):
                self.calls.append((input_ids.clone(), cache))
                logits = torch.full((1, input_ids.shape[1], 2), -1.0, device=input_ids.device)
                logits[..., 0] = 1.0
                return logits, object()

        model = _CachedModel()
        tokenizer = _Tokenizer()

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
                use_cache=True,
            )
        )

        self.assertEqual(reply, "a")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0][0].shape, (1, 1))
        self.assertEqual(model.calls[1][0].shape, (1, 1))

    def test_handle_repl_command_toggles_cache(self):
        class _CachedModel:
            def forward_cached(self):
                return None

        handled, message, temperature, top_k, show_tokens, use_cache = _handle_repl_command(
            "/cache on",
            temperature=0.8,
            top_k=50,
            show_tokens=False,
            use_cache=False,
            model=_CachedModel(),
        )

        self.assertTrue(handled)
        self.assertEqual(message, "KV cache enabled.")
        self.assertEqual(temperature, 0.8)
        self.assertEqual(top_k, 50)
        self.assertFalse(show_tokens)
        self.assertTrue(use_cache)

    def test_handle_repl_command_toggles_cache_without_arg(self):
        class _CachedModel:
            def forward_cached(self):
                return None

        handled, message, temperature, top_k, show_tokens, use_cache = _handle_repl_command(
            "/cache",
            temperature=0.8,
            top_k=50,
            show_tokens=False,
            use_cache=True,
            model=_CachedModel(),
        )

        self.assertTrue(handled)
        self.assertEqual(message, "KV cache disabled.")
        self.assertEqual(temperature, 0.8)
        self.assertEqual(top_k, 50)
        self.assertFalse(show_tokens)
        self.assertFalse(use_cache)

    def test_handle_repl_command_rejects_unknown_slash_command(self):
        handled, message, temperature, top_k, show_tokens, use_cache = _handle_repl_command(
            "/unknowncommand",
            temperature=0.8,
            top_k=50,
            show_tokens=False,
            use_cache=True,
            model=object(),
        )

        self.assertTrue(handled)
        self.assertEqual(message, "Unknown command: /unknowncommand")
        self.assertEqual(temperature, 0.8)
        self.assertEqual(top_k, 50)
        self.assertFalse(show_tokens)
        self.assertTrue(use_cache)

    def test_cache_status_label(self):
        self.assertEqual(_cache_status_label(True), "enabled")
        self.assertEqual(_cache_status_label(False), "disabled")

    def test_checkpoint_status_label(self):
        self.assertEqual(_checkpoint_status_label(None), "none (random init)")
        self.assertEqual(_checkpoint_status_label("models/example.pt"), "models/example.pt")

    def test_default_sample_prompt(self):
        self.assertEqual(DEFAULT_SAMPLE_PROMPT, "Der Sinn des Lebens ist was?")

    def test_build_prompt_uses_completion_format_when_not_post_training(self):
        old_post_training = chat_module.config.POST_TRAINING
        chat_module.config.POST_TRAINING = False
        try:
            self.assertEqual(_build_prompt("Hallo"), "<|EOS|>Hallo")
        finally:
            chat_module.config.POST_TRAINING = old_post_training

    def test_sample_replies_runs_multiple_fresh_generations(self):
        class _Tokenizer:
            def get_vocab(self):
                return {"a": 0, "<|EOS|>": 1}

            def encode(self, text):
                return types.SimpleNamespace(ids=[0])

            def decode(self, ids):
                return "a"

            def token_to_id(self, token):
                return {"<|EOS|>": 1}.get(token)

        class _CachedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward_cached(self, input_ids, cache=None):
                self.calls += 1
                logits = torch.full((1, input_ids.shape[1], 2), -1.0, device=input_ids.device)
                logits[..., 0] = 1.0
                return logits, object()

        tokenizer = _Tokenizer()
        model = _CachedModel()

        replies = _sample_replies(
            model,
            tokenizer,
            prompt="hi",
            context_len=8,
            num_samples=3,
            max_new_tokens=1,
            temperature=0.0,
            top_k=0,
            device=torch.device("cpu"),
            stop_id=None,
            use_cache=True,
        )

        self.assertEqual(replies, ["a", "a", "a"])
        self.assertEqual(model.calls, 6)


if __name__ == "__main__":
    unittest.main()
