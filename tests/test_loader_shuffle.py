import unittest

import loader


def _single_item():
    yield 1


class LoaderShuffleTests(unittest.TestCase):
    """Validate buffer shuffling for torch pipeline datasets.
    Ensures deterministic ordering with a fixed seed.
    Confirms output preserves the original elements.
    """
    def test_shuffle_deterministic(self):
        items = list(range(10))
        first = list(loader.shuffle_dataset(items, buffer_size=4, seed=123))
        second = list(loader.shuffle_dataset(items, buffer_size=4, seed=123))

        self.assertEqual(first, second)
        self.assertCountEqual(first, items)

    def test_shuffle_buffer_size_one(self):
        items = list(range(6))
        shuffled = list(loader.shuffle_dataset(items, buffer_size=1, seed=7))

        self.assertEqual(shuffled, items)

    def test_resolve_dataloader_workers_for_generator(self):
        dataset = loader.GeneratorIterableDataset(_single_item, {})

        self.assertEqual(loader.resolve_dataloader_workers(dataset, 2), 0)


if __name__ == "__main__":
    unittest.main()
