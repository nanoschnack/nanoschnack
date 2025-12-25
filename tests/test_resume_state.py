import unittest

from model.resume import build_resume_state, normalize_resume_rows


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
        dataset_specs = [
            {"spec": "new:spec"},
        ]

        resume_state = build_resume_state(source_row_counts, dataset_specs)

        self.assertEqual(
            resume_state["datasets"],
            [
                {"spec": "new:spec", "row_offset": 1},
                {"spec": "old:spec", "row_offset": 5},
            ],
        )
