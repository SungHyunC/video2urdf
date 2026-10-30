import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from video2urdf.calibration import reference_scale, check_references


class CalibrationTest(unittest.TestCase):
    def test_reference_distance_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp)
            (capture / "depth").mkdir()
            cv2.imwrite(str(capture / "depth/000000.png"), np.full((50, 50), 1000, np.uint16))
            np.savetxt(capture / "camera_matrix.csv", [[100, 0, 50], [0, 100, 50], [0, 0, 1]], delimiter=",")
            with patch("video2urdf.calibration.video_info", return_value=((100, 100), 1, 30)):
                result = reference_scale(capture, 0, (25, 50), (75, 50), 250)
                self.assertAlmostEqual(result["estimated_length_before_mm"], 500)
                self.assertAlmostEqual(result["scale_factor"], .5)
                with self.assertRaises(ValueError):
                    reference_scale(capture, 0, (-1, 50), (75, 50), 250)
                depth = np.full((50, 50), 1000, np.uint16)
                depth[24:27, 24:27] = np.array([[500, 500, 500], [500, 1000, 1000], [1000, 1000, 1000]])
                cv2.imwrite(str(capture / "depth/000000.png"), depth)
                with self.assertRaisesRegex(ValueError, "depth edge"):
                    reference_scale(capture, 0, (50, 50), (75, 50), 250)

    def test_independent_reference_reports_residual_after_global_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp)
            (capture / "depth").mkdir()
            cv2.imwrite(str(capture / "depth/000000.png"), np.full((50, 50), 1000, np.uint16))
            np.savetxt(capture / "camera_matrix.csv", [[100, 0, 50], [0, 100, 50], [0, 0, 1]], delimiter=",")
            with patch("video2urdf.calibration.video_info", return_value=((100, 100), 1, 30)):
                result = check_references(capture, [[0, 25, 50, 75, 50, 200]], .5)[0]
                self.assertAlmostEqual(result["predicted_after_calibration_mm"], 250)
                self.assertAlmostEqual(result["relative_error"], .25)


if __name__ == "__main__":
    unittest.main()
