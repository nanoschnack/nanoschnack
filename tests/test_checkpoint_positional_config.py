import unittest

from checkpointer import apply_checkpoint_config
import config


class CheckpointPositionalConfigTests(unittest.TestCase):
    """Ensure checkpoint config wires positional encoding choices.

    Uses explicit checkpoint config values when present.
    Falls back to learned embeddings for older checkpoints.
    Protects configuration defaults for new runs.
    """
    def setUp(self):
        self._orig_pos_type = config.POS_EMBED_TYPE
        self._orig_rope_base = config.ROPE_BASE

    def tearDown(self):
        config.POS_EMBED_TYPE = self._orig_pos_type
        config.ROPE_BASE = self._orig_rope_base

    def test_apply_checkpoint_config_defaults_to_learned(self):
        config.POS_EMBED_TYPE = "rope"
        apply_checkpoint_config({"CONTEXT_LEN": config.CONTEXT_LEN})
        self.assertEqual(config.POS_EMBED_TYPE, "learned")

    def test_apply_checkpoint_config_accepts_rope(self):
        apply_checkpoint_config({"POS_EMBED_TYPE": "rope", "ROPE_BASE": 7777.0})
        self.assertEqual(config.POS_EMBED_TYPE, "rope")
        self.assertEqual(config.ROPE_BASE, 7777.0)


if __name__ == "__main__":
    unittest.main()
