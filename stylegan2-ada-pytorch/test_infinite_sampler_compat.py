import unittest

from torch_utils.misc import InfiniteSampler


class TinyDataset:
    def __len__(self):
        return 3


class InfiniteSamplerCompatTest(unittest.TestCase):
    def test_infinite_sampler_constructs_on_current_torch(self):
        sampler = InfiniteSampler(TinyDataset())
        self.assertIn(next(iter(sampler)), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
