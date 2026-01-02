import unittest

import torch

from model import setup_paths

setup_paths()

from model.plot import _unwrap_model


class PlotCompletionModelTests(unittest.TestCase):
    """Check completion model unwrapping behavior.
    Ensures wrappers exposing a module attribute are unwrapped.
    Keeps coverage lightweight for training utilities.
    """
    def test_unwrap_model_prefers_module(self):
        class _Wrapper:
            def __init__(self, module):
                self.module = module

        inner = torch.nn.Linear(2, 2)
        wrapped = _Wrapper(inner)

        self.assertIs(_unwrap_model(wrapped), inner)

    def test_unwrap_model_returns_original(self):
        module = torch.nn.Linear(2, 2)

        self.assertIs(_unwrap_model(module), module)


if __name__ == "__main__":
    unittest.main()
