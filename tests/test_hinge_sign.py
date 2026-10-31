import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from video2urdf.lidar import read_hinge
from video2urdf.physics import hinge_q


def run_dir(tmp: Path, q, drawn=None) -> Path:
    (tmp / "hinge.json").write_text(json.dumps({"axis": [0.0, 0.6, 0.8], "point": [0, 0, 0],
                                                "q": [None if v is None else v for v in q]}))
    if drawn is not None:
        np.savez(tmp / "replay3d.npz", q_lid_joint=np.array(drawn, float))
    return tmp


class HingeSignTest(unittest.TestCase):
    def test_agreeing_replay_keeps_the_sign(self):
        with tempfile.TemporaryDirectory() as d:
            h = read_hinge(run_dir(Path(d), [0.1, None, 0.3], [0.1, np.nan, 0.3]))
            self.assertEqual(h["q"], [0.1, None, 0.3])
            self.assertEqual(h["axis"], [0.0, 0.6, 0.8])

    def test_door_fitted_before_the_fix_reads_in_the_models_sense(self):
        # build_door_model turned the axis up and negated q; the replay drew the negated values
        with tempfile.TemporaryDirectory() as d:
            run = run_dir(Path(d), [0.1, None, 0.3], [-0.1, np.nan, -0.3])
            h = read_hinge(run)
            self.assertEqual(h["q"], [-0.1, None, -0.3])
            np.testing.assert_allclose(h["axis"], [0.0, -0.6, -0.8])
            np.testing.assert_allclose(hinge_q(run), [-0.1, np.nan, -0.3])

    def test_no_replay_reads_the_file_as_is(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(read_hinge(run_dir(Path(d), [0.2, 0.4]))["q"], [0.2, 0.4])


if __name__ == "__main__":
    unittest.main()
