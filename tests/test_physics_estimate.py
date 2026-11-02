import unittest
from pathlib import Path
from unittest.mock import patch

from video2urdf import physics


class PhysicsEstimateTest(unittest.TestCase):
    def test_saved_estimate_becomes_apply_arguments(self):
        estimate = {"parts": [{"link": "base", "material": "aluminum", "mass_kg": 0.9, "friction": 0.6, "why": "w"}],
                    "released": [{"joint": "lid_joint", "start_s": 0.0, "end_s": 2.2},
                                 {"joint": "lid", "start_s": 7.5, "end_s": 10.5}]}  # a joint named without _joint
        with patch.object(physics, "apply", return_value={}) as apply:
            physics.apply_estimate(Path("run"), estimate, 51.476)
        (track_dir, parts, windows, fps), kwargs = apply.call_args
        self.assertEqual(parts, {"base": {"material": "aluminum", "mass_kg": 0.9, "friction": 0.6, "why": "w"}})
        self.assertEqual(windows, {"lid_joint": [(0, 113), (386, 540)]})  # frames at 51.476 fps
        self.assertEqual((track_dir, fps), (Path("run"), 51.476))


if __name__ == "__main__":
    unittest.main()
