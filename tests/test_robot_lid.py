import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from scripts.robot_tasks import (grip_width, left_alone, lid_ends, lid_start, lid_start_candidates, plan_grasp,
                                 rest_model)

# A laptop seen from the side: a 16 cm base whose back edge (the hinge) is at x = 0, and a 15 cm lid on a
# hinge along -y, so a positive q tips its top over the base (towards -x). Range -10..80 deg: 80 is shut.
# LEAN_X: how far the lid's centre of mass sits towards the base at q = 0 (it stands near vertical there).
# The base's <inertial> marks the model as physics-measured, so the robot scene keeps its joint friction.
LID = """<mujoco model="laptop"><compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <inertial pos="-0.08 0 -0.01" mass="1" diaginertia="0.008 0.002 0.01"/>
      <geom type="box" pos="-0.08 0 -0.01" size="0.08 0.15 0.01"/>
      <body name="lid">
        <joint name="lid_joint" type="hinge" axis="0 -1 0" range="-0.1745 1.3963" limited="true"
               frictionloss="FRICTION" damping="0.02"/>
        <geom type="box" pos="-LEAN_X 0 0.075" size="0.004 0.15 0.075" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <contact><exclude body1="base" body2="lid"/></contact>
</mujoco>"""


def lid_model(friction: float, lean_x: float) -> tuple[mujoco.MjModel, Path]:
    """The robot's view of the lid alone (rest_model: names prefixed obj_, the scene's solver options)."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "model.xml").write_text(LID.replace("FRICTION", str(friction)).replace("LEAN_X", str(lean_x)))
    return rest_model(tmp / "model.xml", hold="lid_joint"), tmp


class RobotLidTest(unittest.TestCase):
    def test_candidates_start_with_the_reference_pose_then_the_open_end(self):
        shut, open_end = np.deg2rad(74.5), np.deg2rad(-3.9)  # the side-view notebook: shut at the top
        c = np.rad2deg(lid_start_candidates(0.0, shut, open_end))
        np.testing.assert_allclose(c[:3], [0.0, -3.9, -1.4], atol=1e-9)
        self.assertLessEqual(c[-1], (74.5 - 3.9) / 2 + 1e-9)  # never past the middle of the range
        self.assertEqual(len(set(np.round(c, 6))), len(c))

    def test_reference_pose_too_near_shut_is_skipped(self):
        shut, open_end = np.deg2rad(-100.0), np.deg2rad(20.0)  # shut at the bottom of the range
        c = np.rad2deg(lid_start_candidates(np.deg2rad(-70.0), shut, open_end))
        self.assertAlmostEqual(c[0], 20.0)  # -70 leaves only 30 of 120 deg to close
        np.testing.assert_allclose(np.diff(c), -2.5, atol=1e-9)
        c = np.rad2deg(lid_start_candidates(0.0, shut, open_end))
        self.assertEqual(c[0], 0.0)

    def test_shut_is_where_the_free_edge_sits_lowest(self):
        model, _ = lid_model(0.03, 0.004)
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        shut, open_end = lid_ends(model, data, "obj_lid_joint", plan_grasp(model, data, "plate", "obj_lid_joint"))
        self.assertAlmostEqual(shut, 1.3963, places=4)
        self.assertAlmostEqual(open_end, -0.1745, places=4)

    def test_grip_width_is_the_lid_thickness_at_its_free_edge(self):
        model, _ = lid_model(0.03, 0.004)
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        grasp = plan_grasp(model, data, "plate", "obj_lid_joint")
        self.assertAlmostEqual(grip_width(model, data, grasp), 0.008, places=6)  # the 8 mm lid, not its 15 cm
        data.joint("obj_lid_joint").qpos[0] = 1.0  # the same in the lid's frame at any opening
        mujoco.mj_kinematics(model, data)
        self.assertAlmostEqual(grip_width(model, data, grasp), 0.008, places=6)

    def test_a_lid_that_falls_from_the_reference_pose_starts_against_its_open_limit(self):
        # friction 0.005 N m can't hold the lid's 0.02 N m at q = 0; at -10 deg it leans back onto its limit
        model, _ = lid_model(0.005, 0.004)
        self.assertGreater(left_alone(model, "obj_lid_joint", 0.0), 1.3)  # falls shut
        q, held, tried = lid_start(model, "obj_lid_joint", 0.0, 1.3963, -0.1745)
        self.assertAlmostEqual(q, -0.1745)
        self.assertFalse(held)
        self.assertEqual([round(float(t), 4) for t, _ in tried], [0.0, -0.1745])

    def test_a_lid_its_friction_holds_starts_at_the_reference_pose(self):
        model, _ = lid_model(0.03, 0.004)
        q, held, tried = lid_start(model, "obj_lid_joint", 0.0, 1.3963, -0.1745)
        self.assertEqual((q, held, len(tried)), (0.0, False, 1))

    def test_a_lid_that_rests_nowhere_open_is_held_at_its_first_candidate(self):
        model, _ = lid_model(0.005, 0.03)  # leans 22 deg towards shut at q = 0: still 12 deg at its open limit
        q, held, tried = lid_start(model, "obj_lid_joint", 0.0, 1.3963, -0.1745)
        self.assertEqual((q, held), (0.0, True))
        self.assertTrue(all(abs(end - start) > np.deg2rad(1) for start, end in tried))


if __name__ == "__main__":
    unittest.main()
