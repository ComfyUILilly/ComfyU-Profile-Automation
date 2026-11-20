import random
import unittest

from comfy_profile_automation.profiles import ProfileManager


class ProfileManagerTests(unittest.TestCase):
    def test_batch_stays_with_selected_profile(self):
        manager = ProfileManager(
            {"IL": {"weight": 1, "batch_images": [3, 3]},
             "Anime": {"weight": 0, "batch_images": 8}},
            rng=random.Random(1),
        )
        self.assertEqual(manager.next_batch(), ("IL", 3))

    def test_rejects_zero_total_weight(self):
        manager = ProfileManager({"IL": {"weight": 0}})
        with self.assertRaises(ValueError):
            manager.next_batch()


if __name__ == "__main__":
    unittest.main()
