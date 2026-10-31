import unittest

import mujoco
import numpy as np

from scripts.robot_tasks import (door_ends, door_open_target, door_push_point, door_start_candidates, door_yaw,
                                 is_door_axis, kind_of, opening_deg, slerp_rot)

# A cabinet 0.4 m wide with its front at y = 0, and a 0.38 m door on a vertical hinge at its front-left edge;
# the door opens outwards (towards -y) as q goes negative, so q = 0 (the top of its range) is shut.
DOOR = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="obj_base"><geom type="box" pos="0.2 0.2 0" size="0.2 0.2 0.15"/>
    <body name="obj_lid"><joint name="obj_lid_joint" type="hinge" axis="0 0 1" range="-1.2 0" limited="true"/>
      <geom type="box" pos="0.19 -0.01 0" size="0.19 0.01 0.14"/></body></body>
</worldbody></mujoco>"""


class RobotDoorTest(unittest.TestCase):
    def test_vertical_hinge_is_a_door(self):
        self.assertTrue(is_door_axis([0.069, -0.079, 0.995]))  # the microwave's, 6 deg off vertical
        self.assertTrue(is_door_axis([0, 0, -2.0]))  # either sense, any length
        self.assertFalse(is_door_axis([1.0, 0, 0]))  # a laptop lid's
        self.assertFalse(is_door_axis([0.6, 0, 0.8]))  # 37 deg off vertical

    def test_kind_of_a_door_model(self):
        model = mujoco.MjModel.from_xml_string(DOOR)
        self.assertEqual(kind_of(model), ("door", "obj_lid_joint"))
        lid = DOOR.replace('axis="0 0 1"', 'axis="1 0 0"')  # the same plate on a horizontal hinge
        self.assertEqual(kind_of(mujoco.MjModel.from_xml_string(lid)), ("plate", "obj_lid_joint"))

    def test_door_ends_find_shut_and_the_outward_normal(self):
        model = mujoco.MjModel.from_xml_string(DOOR)
        shut, open_end, out = door_ends(model, "obj_lid_joint")
        self.assertEqual((shut, open_end), (0.0, -1.2))
        np.testing.assert_allclose(out, [0, -1, 0], atol=1e-9)  # the outer face, the way the free edge heads

    def test_push_point_on_the_outer_face(self):
        model = mujoco.MjModel.from_xml_string(DOOR)
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        point, normal = door_push_point(model, data, model.joint("obj_lid_joint").id, np.array([0, -1.0, 0]))
        np.testing.assert_allclose(point, [0.38 - 0.07, -0.02, 0], atol=1e-9)  # 7 cm in from the free edge
        np.testing.assert_allclose(normal, [0, -1, 0], atol=1e-9)

    def test_yaw_turns_the_door_towards_the_robot(self):
        for normal in ([0, -1.0], [1.0, 0], [0.3, 0.7], [-1.0, 0.0]):
            yaw = door_yaw(normal)
            c, s = np.cos(yaw), np.sin(yaw)
            turned = np.array([[c, -s], [s, c]]) @ np.asarray(normal)
            np.testing.assert_allclose(turned / np.linalg.norm(turned), [-1, 0], atol=1e-9)
            self.assertLessEqual(abs(yaw), np.pi + 1e-9)

    def test_opening_counts_from_shut_towards_the_open_end(self):
        shut, open_end = 0.2131, -0.7336  # the microwave: shut at the top of its range
        self.assertAlmostEqual(opening_deg(shut, shut, open_end), 0.0)
        self.assertAlmostEqual(opening_deg(open_end, shut, open_end), np.rad2deg(0.9467), places=6)
        self.assertAlmostEqual(opening_deg(0.0, shut, open_end), np.rad2deg(0.2131), places=6)
        self.assertAlmostEqual(opening_deg(0.3, 0.0, 1.0), np.rad2deg(0.3))  # shut at the bottom
        target = door_open_target(shut, open_end)
        self.assertAlmostEqual(opening_deg(target, shut, open_end), 0.85 * np.rad2deg(0.9467), places=6)

    def test_start_candidates(self):
        self.assertEqual(door_start_candidates(12.2, 46.0)[:3], [12.2, 15.0, 17.5])
        self.assertLessEqual(max(door_start_candidates(12.2, 46.0)), 35.0)
        self.assertEqual(door_start_candidates(0.0, 46.0)[:2], [0.0, 2.5])
        self.assertEqual(door_start_candidates(40.0, 46.0), [])  # too little left to pull
        self.assertTrue(all(o <= 20.0 for o in door_start_candidates(5.0, 30.0)))

    def test_slerp_rot_ends(self):
        a, b = np.eye(3), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        np.testing.assert_allclose(slerp_rot(a, b, 0), a, atol=1e-9)
        np.testing.assert_allclose(slerp_rot(a, b, 1), b, atol=1e-9)
        np.testing.assert_allclose(slerp_rot(a, b, 0.5) @ [1, 0, 0], [np.cos(np.pi / 4), np.sin(np.pi / 4), 0], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
