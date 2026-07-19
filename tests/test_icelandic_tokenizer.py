import unittest

from tokenizers import Tokenizer


class IcelandicTokenizerTests(unittest.TestCase):
    """Validate representative inflections in tokenizer-icelandic-v1.

    Pins both token boundaries and IDs for the generated artifact.
    Ensures byte-level token labels decode to readable Icelandic pieces.
    """
    @classmethod
    def setUpClass(cls):
        # Load the generated Icelandic tokenizer once for all assertions.
        cls.tokenizer = Tokenizer.from_file("tokenizer/tokenizer-icelandic-v1.json")

    def test_kottur_inflections(self):
        # Pin token pieces and IDs requested for the Icelandic tokenizer.
        expected = {
            "köttur": (["k", "öttur"], [107, 25831]),
            "kötturinn": (["k", "ött", "urinn"], [107, 4785, 671]),
            "ketti": (["ke", "tti"], [10874, 602]),
            "kettinum": (["k", "ettinum"], [107, 22670]),
        }
        for word, (pieces, token_ids) in expected.items():
            with self.subTest(word=word):
                encoding = self.tokenizer.encode(word)
                decoded_pieces = [self.tokenizer.decode([token_id]) for token_id in encoding.ids]
                self.assertEqual(decoded_pieces, pieces)
                self.assertEqual(encoding.ids, token_ids)
                self.assertEqual(self.tokenizer.decode(encoding.ids), word)


if __name__ == "__main__":
    unittest.main()
