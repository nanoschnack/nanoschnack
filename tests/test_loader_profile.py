import unittest
from unittest import mock

from model import setup_paths

setup_paths()

from model.loader import _LoaderProfiler


class LoaderProfileTests(unittest.TestCase):
    """Validate loader profiling counters.
    Ensures event stats aggregate as expected.
    Avoids noisy logs by patching print.
    """
    def test_profile_aggregates_stats(self):
        profiler = _LoaderProfiler(every=2)
        with mock.patch("builtins.print"):
            profiler.record("tokenize", 0.5, rows=10, tokens=100)
            profiler.record("tokenize", 1.0, rows=5, tokens=50)

        stats = profiler.stats["tokenize"]
        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["rows"], 15)
        self.assertEqual(stats["tokens"], 150)
        self.assertAlmostEqual(stats["duration"], 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
