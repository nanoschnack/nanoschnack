import unittest

import model.config as config


class AlignMicroBatchSizeTest(unittest.TestCase):
    """Check micro batch alignment against macro batch constraints.
    Ensures divisibility rules are respected.
    Covers downscaling and macro cap scenarios.
    Guards against non-positive sizes.
    """

    def test_returns_same_when_divisible(self):
        self.assertEqual(config.align_micro_batch_size(64, 512), 64)

    def test_reduces_to_largest_divisor(self):
        self.assertEqual(config.align_micro_batch_size(100, 512), 64)

    def test_caps_at_macro_batch(self):
        self.assertEqual(config.align_micro_batch_size(1024, 512), 512)

    def test_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            config.align_micro_batch_size(0, 512)
