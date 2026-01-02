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


class DataLoaderConfigTest(unittest.TestCase):
    """Validate DataLoader overlap config defaults.
    Ensures prefetch factor is positive.
    Checks persistent and pin memory flags are booleans.
    """
    def test_prefetch_factor_positive(self):
        # Require a positive prefetch factor for worker overlap.
        self.assertGreater(config.DATA_LOADER_PREFETCH_FACTOR, 0)

    def test_persistent_worker_flag(self):
        # Validate persistent worker flag is boolean.
        self.assertIsInstance(config.DATA_LOADER_PERSISTENT, bool)

    def test_pin_memory_flag(self):
        # Validate pin memory flag is boolean.
        self.assertIsInstance(config.DATA_LOADER_PIN_MEMORY, bool)

    def test_fast_pipeline_flag(self):
        # Validate fast pipeline flag is boolean.
        self.assertIsInstance(config.DATASET_FAST_PIPELINE, bool)
