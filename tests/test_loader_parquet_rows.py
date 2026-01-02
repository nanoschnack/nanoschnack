import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from model import setup_paths

setup_paths()

from model.loader import _iter_parquet_rows


class LoaderParquetRowsTests(unittest.TestCase):
    """Verify PyArrow-based parquet row iteration.
    Ensures rows are emitted with all columns preserved.
    Uses a tiny parquet file for fast validation.
    """
    def test_iter_parquet_rows_yields_dicts(self):
        table = pa.table({"text": ["a", "b"], "id": [1, 2]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/sample.parquet"
            pq.write_table(table, path)
            rows = list(_iter_parquet_rows(path))

        self.assertEqual(rows, [{"text": "a", "id": 1}, {"text": "b", "id": 2}])


if __name__ == "__main__":
    unittest.main()
