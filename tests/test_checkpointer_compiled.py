import unittest

import torch

from checkpointer import load_model_state_dict


class CompiledWrapper(torch.nn.Module):
    """Simulate a torch.compile wrapper for state_dict keys.

    Registers the original module under _orig_mod to mirror OptimizedModule.
    Keeps the wrapper minimal for fast unit tests.
    """
    def __init__(self, module):
        super().__init__()
        self._orig_mod = module

    def forward(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)


class CheckpointerCompiledTests(unittest.TestCase):
    """Exercise compiled wrapper checkpoint loading fallback.

    Uses a linear layer to keep state dictionaries small and deterministic.
    Confirms unprefixed weights load into a wrapped model instance.
    """
    def test_load_model_state_dict_into_compiled_wrapper(self):
        source = torch.nn.Linear(2, 3)
        wrapped = CompiledWrapper(torch.nn.Linear(2, 3))
        with torch.no_grad():
            wrapped._orig_mod.weight.zero_()
            wrapped._orig_mod.bias.zero_()

        result = load_model_state_dict(wrapped, source.state_dict())

        self.assertEqual(result, "compiled")
        self.assertTrue(torch.equal(wrapped._orig_mod.weight, source.weight))
        self.assertTrue(torch.equal(wrapped._orig_mod.bias, source.bias))


if __name__ == "__main__":
    unittest.main()
