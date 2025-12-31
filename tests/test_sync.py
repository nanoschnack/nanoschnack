import unittest
from dataclasses import dataclass

import torch

import sync


class _FakeDist:
    class ReduceOp:
        SUM = "sum"
        MIN = "min"
        MAX = "max"

    def all_reduce(self, tensor, op=None):
        return tensor

    def broadcast(self, tensor, src=0):
        return tensor

    def all_gather(self, gathered, tensor):
        for idx in range(len(gathered)):
            gathered[idx].copy_(tensor)

    def get_world_size(self):
        return 1


@dataclass
class FlagState:
    """Exercise flag sync semantics for boolean fields.
    Uses reduce and broadcast flag helpers.
    Keeps bools at the call site.
    """
    stop_flag: bool = sync.flag_reduce("max")
    input_flag: bool = sync.flag_broadcast(src=0)


class SyncFlagTests(unittest.TestCase):
    def setUp(self):
        self._orig_dist = sync.dist
        self._orig_reduce_ops = sync._REDUCE_OPS
        sync.dist = _FakeDist()
        sync._REDUCE_OPS = {
            "sum": sync.dist.ReduceOp.SUM,
            "min": sync.dist.ReduceOp.MIN,
            "max": sync.dist.ReduceOp.MAX,
        }

    def tearDown(self):
        sync.dist = self._orig_dist
        sync._REDUCE_OPS = self._orig_reduce_ops

    def test_flag_sync_round_trip(self):
        state = FlagState(stop_flag=True, input_flag=False)
        synced = sync.sync(state, device=torch.device("cpu"))
        self.assertIs(synced.stop_flag, True)
        self.assertIs(synced.input_flag, False)

    def test_flag_sync_bool_type(self):
        state = FlagState(stop_flag=False, input_flag=True)
        synced = sync.sync(state, device=torch.device("cpu"))
        self.assertIsInstance(synced.stop_flag, bool)
        self.assertIsInstance(synced.input_flag, bool)


if __name__ == "__main__":
    unittest.main()
