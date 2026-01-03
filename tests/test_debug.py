import os
import tempfile
import unittest
from unittest import mock

from model import debug


class DebugWorkerInitTests(unittest.TestCase):
    """Exercise DataLoader worker init logging helpers.

    These tests avoid touching real worker processes.
    They validate log path creation and no-op behavior.
    """
    def test_worker_init_noop_without_worker_info(self):
        worker_init_fn = debug.build_dataloader_worker_init()

        with mock.patch.object(debug, "get_worker_info", return_value=None) as mocked:
            worker_init_fn(None)

        mocked.assert_called_once()

    def test_worker_init_writes_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            worker_init_fn = debug.build_dataloader_worker_init(log_prefix="test_worker")

            info = mock.Mock()
            info.id = 3
            log_path = os.path.join(tmpdir, "test_worker_3.log")

            with mock.patch.object(debug, "get_worker_info", return_value=info), \
                mock.patch.object(debug.tempfile, "gettempdir", return_value=tmpdir), \
                mock.patch.object(debug.faulthandler, "enable"), \
                mock.patch.object(debug.os, "dup2"):
                worker_init_fn(None)

            self.assertTrue(os.path.exists(log_path))
            for log_file in worker_init_fn.log_files:
                log_file.close()
