import io
import random
import unittest
from contextlib import redirect_stdout

import torch

from model.progress import ProgressLogger


class ProgressLoggerTests(unittest.TestCase):
    """Validate progress logger sample output formatting.

    Uses a stub tokenizer to keep decode behavior deterministic.
    Ensures the indentation and source id label stay consistent.
    Avoids randomness by sampling from a single-row batch.
    """

    def test_print_text_samples_indents_lines(self):
        class _Tokenizer:
            def decode(self, ids):
                return "line one\nline two"

        logger = ProgressLogger()
        inputs = torch.tensor([[1, 2, 3]])
        source_ids = torch.tensor([3])

        random.seed(0)
        stream = io.StringIO()
        with redirect_stdout(stream):
            logger.print_text_samples(
                inputs,
                attention_mask=None,
                tokenizer=_Tokenizer(),
                sample_count=1,
                source_ids=source_ids,
            )

        output = stream.getvalue().splitlines()
        self.assertEqual(output[0], "Sample (source 3):")
        self.assertEqual(output[1], "  line one")
        self.assertEqual(output[2], "  line two")
