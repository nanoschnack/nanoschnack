import unittest

from model.resume import (
    build_resume_state,
    cap_resume_rows,
    is_resume_exhausted,
    normalize_resume_rows,
    normalize_resume_tokens,
    seed_missing_token_offsets,
)


class ResumeStateTests(unittest.TestCase):
    """Verify resume-state helpers preserve offsets for inactive datasets.
    Ensures current specs are always present with defaults.
    Confirms checkpoint emission keeps stable ordering for readability.
    """

    def test_normalize_resume_rows_keeps_unknown_specs(self):
        resume_state = {
            "datasets": [
                {"spec": "old:spec", "row_offset": 12},
            ]
        }
        dataset_specs = [
            {"spec": "new:spec"},
        ]

        resume_rows = normalize_resume_rows(resume_state, dataset_specs)

        self.assertEqual(resume_rows["old:spec"], 12)
        self.assertEqual(resume_rows["new:spec"], 0)

    def test_build_resume_state_orders_current_specs_first(self):
        source_row_counts = {
            "old:spec": 5,
            "new:spec": 1,
        }
        source_token_counts = {
            "old:spec": 50,
            "new:spec": 10,
        }
        dataset_specs = [
            {"spec": "new:spec"},
        ]

        resume_state = build_resume_state(
            source_row_counts,
            dataset_specs,
            source_token_counts=source_token_counts,
        )

        self.assertEqual(
            resume_state["datasets"],
            [
                {"spec": "new:spec", "row_offset": 1, "token_offset": 10},
                {"spec": "old:spec", "row_offset": 5, "token_offset": 50},
            ],
        )

    def test_build_resume_state_keeps_token_only_specs(self):
        source_row_counts = {
            "new:spec": 2,
        }
        source_token_counts = {
            "new:spec": 20,
            "old:spec": 50,
        }
        dataset_specs = [
            {"spec": "new:spec"},
        ]

        resume_state = build_resume_state(
            source_row_counts,
            dataset_specs,
            source_token_counts=source_token_counts,
        )

        self.assertEqual(
            resume_state["datasets"],
            [
                {"spec": "new:spec", "row_offset": 2, "token_offset": 20},
                {"spec": "old:spec", "row_offset": 0, "token_offset": 50},
            ],
        )

    def test_normalize_resume_tokens_keeps_unknown_specs(self):
        resume_state = {
            "datasets": [
                {"spec": "old:spec", "row_offset": 12, "token_offset": 120},
            ]
        }
        dataset_specs = [
            {"spec": "new:spec"},
        ]

        resume_tokens = normalize_resume_tokens(resume_state, dataset_specs)

        self.assertEqual(resume_tokens["old:spec"], 120)
        self.assertEqual(resume_tokens["new:spec"], 0)

    def test_seed_missing_token_offsets_uses_baseline(self):
        resume_state = {
            "datasets": [
                {"spec": "old:spec", "row_offset": 12, "token_offset": 120},
            ]
        }
        dataset_specs = [
            {"spec": "old:spec"},
            {"spec": "new:spec"},
        ]

        resume_tokens = normalize_resume_tokens(resume_state, dataset_specs)
        seeded = seed_missing_token_offsets(resume_tokens, resume_state, dataset_specs)

        self.assertEqual(seeded["old:spec"], 120)
        self.assertEqual(seeded["new:spec"], 120)

    def test_seed_missing_token_offsets_backfills_missing_tokens(self):
        resume_state = {
            "datasets": [
                {"spec": "old:spec", "row_offset": 12, "token_offset": 120},
                {"spec": "missing:tokens", "row_offset": 8},
            ]
        }
        dataset_specs = [
            {"spec": "old:spec"},
            {"spec": "missing:tokens"},
        ]

        resume_tokens = normalize_resume_tokens(resume_state, dataset_specs)
        seeded = seed_missing_token_offsets(resume_tokens, resume_state, dataset_specs)

        self.assertEqual(seeded["old:spec"], 120)
        self.assertEqual(seeded["missing:tokens"], 120)

    def test_is_resume_exhausted(self):
        self.assertFalse(is_resume_exhausted(3, None))
        self.assertFalse(is_resume_exhausted(3, 4))
        self.assertTrue(is_resume_exhausted(4, 4))

    def test_cap_resume_rows_clamps_excess_offsets(self):
        resume_rows = {"a": 15, "b": 3}
        totals = {"a": 10, "b": 10}

        capped = cap_resume_rows(resume_rows, totals)

        self.assertEqual(capped["a"], 10)
        self.assertEqual(capped["b"], 3)

    def test_cap_resume_rows_aligns_completed_passes(self):
        resume_rows = {"a": 12, "b": 15}
        totals = {"a": 10, "b": 10}

        capped = cap_resume_rows(resume_rows, totals)

        self.assertEqual(capped["a"], 2)
        self.assertEqual(capped["b"], 5)
