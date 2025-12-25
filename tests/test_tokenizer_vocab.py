import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from tokenizer import ensure_vocab_size, _aligned_vocab_size


class TokenizerVocabSizeTests(unittest.TestCase):
    """Validate vocab expansion for configured targets.

    Uses a minimal WordLevel tokenizer to avoid network downloads.
    Confirms new tokens are appended to reach the requested size.
    """
    def test_ensure_vocab_size_expands(self):
        vocab = {"[UNK]": 0, "hello": 1}
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        original_size = tokenizer.get_vocab_size()
        new_size = ensure_vocab_size(tokenizer, original_size + 5)

        self.assertEqual(new_size, original_size + 5)
        self.assertEqual(tokenizer.get_vocab_size(), original_size + 5)

    def test_ensure_vocab_size_rejects_smaller(self):
        vocab = {"[UNK]": 0, "hello": 1}
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        original_size = tokenizer.get_vocab_size()

        with self.assertRaises(ValueError):
            ensure_vocab_size(tokenizer, original_size - 1)


class TokenizerVocabAlignTests(unittest.TestCase):
    """Confirm power-of-two alignment for small increases.

    Computes the highest power-of-two multiple that stays within 2%.
    Validates representative vocab sizes against expected alignments.
    """
    def test_aligned_vocab_size_examples(self):
        cases = {
            50258: 51200,
            32000: 32256,
            40000: 40448,
            48000: 48128,
            65000: 65536,
            100000: 100352,
            120000: 120832,
            131072: 131072,
        }
        for base_size, expected in cases.items():
            self.assertEqual(_aligned_vocab_size(base_size), expected)


if __name__ == "__main__":
    unittest.main()
