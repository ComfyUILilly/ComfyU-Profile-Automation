import random
import unittest

from main import _count, _mode


class MainHelperTests(unittest.TestCase):
    def test_count_range_is_inclusive(self):
        rng = random.Random(1)
        self.assertIn(_count([2, 4], rng), {2, 3, 4})

    def test_count_rejects_reversed_or_negative_ranges(self):
        rng = random.Random(1)
        with self.assertRaisesRegex(ValueError, "Invalid count range"):
            _count([4, 2], rng)
        with self.assertRaisesRegex(ValueError, "Invalid count range"):
            _count([-1, 2], rng)

    def test_empty_mode_mapping_falls_back_to_weight(self):
        self.assertEqual(_mode({}, random.Random(1)), "Weight")


if __name__ == "__main__":
    unittest.main()
