import functools
import pickle
import unittest

from model import setup_paths

setup_paths()

from model import loader


class _Tokenizer:
    """Stub tokenizer for pickling tests.
    Mimics encode_batch and token_to_id APIs.
    Returns fixed token ids for simplicity.
    """
    def encode_batch(self, texts):
        return [type("Enc", (), {"ids": [1, 2, 3]})() for _ in texts]

    def token_to_id(self, token):
        return None


class LoaderPickleTests(unittest.TestCase):
    """Check loader helpers are picklable for worker processes.
    Uses a stub tokenizer to simulate encode_batch calls.
    Covers tokenization, packing, and filter callables.
    """
    def test_tokenizer_batch_picklable(self):
        batch_fn = loader.build_tokenizer(_Tokenizer(), text_key="text")
        pickle.dumps(batch_fn)

    def test_pack_and_filter_picklable(self):
        pack_fn = functools.partial(loader._pack_tokens_batch, block_size=4, source_id=1)
        pickle.dumps(pack_fn)
        pickle.dumps(loader._non_empty_sample)

    def test_add_row_count_picklable(self):
        add_fn = functools.partial(loader._add_row_count_batch, text_key="text")
        pickle.dumps(add_fn)


if __name__ == "__main__":
    unittest.main()
