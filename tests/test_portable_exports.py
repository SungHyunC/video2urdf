"""The downloadable model files (video2urdf/portable.py) and the loader-free checks behind
scripts/check_exports.py (video2urdf/urdf_checks.py)."""

import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import mujoco
import numpy as np

from video2urdf import doctor, physics
from video2urdf import portable as P
from video2urdf import urdf_checks as uc
from video2urdf.emit3d import Link3D, Visual, mjcf, urdf
from video2urdf.shape import write_stl

# The LiDAR laptop base's principal frame (model.xml): turned exactly 90 deg about y, where the
# old conversion took roll and yaw apart from near-zero entries and came out 5 deg off.
GIMBAL_QUAT = [0.0313872, 0.70641, -0.0313872, 0.70641]


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def toy(d: Path, physics_like: bool = True) -> list[Link3D]:
    """A base slab and a hinged lid as the 3D path writes them, with inertials as physics.py
    writes them (the base's principal frame at gimbal lock, its URDF rpy broken the old way)."""
    (d / "meshes").mkdir(parents=True)
    write_stl(d / "meshes/base.stl", *box([-0.15, 0, -0.02], [0.15, 0.2, 0]))
    write_stl(d / "meshes/lid.stl", *box([-0.15, -0.01, 0], [0.15, 0, 0.2]))
    links = [Link3D("base", [Visual("meshes/base.stl", (.8, .8, .8))]),
             Link3D("lid", [Visual("meshes/lid.stl", (.7, .7, .7))], parent="base", joint_type="revolute",
                    axis=np.array([1.0, 0, 0]), limits=(-1.8613, 0.1638))]
    (d / "model.xml").write_text(mjcf("toy", links, root_pos=(0, 0, 0.0198)))
    (d / "model.urdf").write_text(urdf("toy", links))
    if physics_like:
        inert = {"base": {"mass": 0.9, "pos": [-0.02, 0.11, -0.013], "quat": GIMBAL_QUAT,
                          "diaginertia": [0.0107799, 0.00733006, 0.00347765]},
                 "lid": {"mass": 0.5, "pos": [0.0, -0.005, 0.1], "quat": [1, 0, 0, 0],
                         "diaginertia": [0.006, 0.004, 0.002]}}
        physics._write_mjcf(d / "model.xml", inert, {"base": {"friction": 0.6}, "lid": {"friction": 0.4}},
                            {"lid_joint": {"friction": 0.28, "damping": 0.02}})
        physics._write_urdf(d / "model.urdf", inert, {"lid_joint": {"friction": 0.28, "damping": 0.02}})
        root = ET.parse(d / "model.urdf")
        o = root.getroot().find("link[@name='base']/inertial/origin")
        o.set("rpy", "0 -1.5708 3.14159")  # what the published laptop URDF says
        root.write(d / "model.urdf", xml_declaration=True, encoding="unicode")
    return links


def old_quat_to_rpy(q):
    """physics._quat_to_rpy before the fix."""
    r = P.quat_matrix(q)
    return math.atan2(r[2, 1], r[2, 2]), math.asin(-max(-1, min(1, r[2, 0]))), math.atan2(r[1, 0], r[0, 0])


class RotationTest(unittest.TestCase):
    def test_rpy_round_trip_including_gimbal_lock(self):
        rng = np.random.default_rng(0)
        quats = [rng.normal(size=4) for _ in range(200)] + [GIMBAL_QUAT, [0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5],
                                                            [math.cos(0.3), 0, math.sin(0.3), 0]]
        for q in quats:
            R = P.quat_matrix(q)
            np.testing.assert_allclose(uc.rpy_matrix(P.quat_rpy(q)), R, atol=1e-9)

    def test_the_old_conversion_was_wrong_at_gimbal_lock(self):
        R = P.quat_matrix(GIMBAL_QUAT)
        old = uc.rpy_matrix(old_quat_to_rpy(GIMBAL_QUAT))
        self.assertGreater(np.abs(old - R).max(), 0.05)  # the regression this fixes
        np.testing.assert_allclose(uc.rpy_matrix(physics._quat_to_rpy(GIMBAL_QUAT)), R, atol=1e-9)


class MujocoBlockTest(unittest.TestCase):
    def test_added_merged_and_idempotent(self):
        text = '<?xml version="1.0"?>\n<!-- <robot name="x"> <mujoco> -->\n<robot name="r">\n  <link name="a"/>\n</robot>\n'
        once = P.with_mujoco_compiler(text, strippath="false")
        self.assertEqual(P.mujoco_compiler(once), {"strippath": "false"})
        self.assertIn("<!-- <robot name=\"x\"> <mujoco> -->", once)  # comments survive, aren't matched
        self.assertEqual(P.with_mujoco_compiler(once, strippath="false"), once)
        merged = P.with_mujoco_compiler(once, meshdir="/m/", fusestatic="false")
        self.assertEqual(P.mujoco_compiler(merged), {"strippath": "false", "meshdir": "/m/", "fusestatic": "false"})
        self.assertEqual(ET.fromstring(merged.split("-->", 1)[1]).findall("mujoco/compiler")[0].get("meshdir"), "/m/")
        empty = '<robot name="r"><mujoco/><link name="a"/></robot>'
        self.assertEqual(P.mujoco_compiler(P.with_mujoco_compiler(empty, strippath="false")), {"strippath": "false"})
        other = '<robot name="r"><mujoco><option timestep="0.001"/></mujoco></robot>'
        out = P.with_mujoco_compiler(other, strippath="false")
        self.assertEqual(ET.fromstring(out).find("mujoco/option").get("timestep"), "0.001")
        self.assertEqual(P.mujoco_compiler(out), {"strippath": "false"})

    def test_emitted_urdf_carries_it_and_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            toy(d, physics_like=False)
            robot = uc.parse_urdf(d / "model.urdf")
            self.assertEqual(robot.mujoco_compiler, {"strippath": "false"})
            model = mujoco.MjModel.from_xml_path(str(d / "model.urdf"))
            self.assertEqual(model.nmesh, 2)

    def test_doctor_loads_a_urdf_that_has_the_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "run"
            toy(d)
            model = doctor.load(d / "model.urdf")  # merged: meshdir, no fusing, visuals kept
            self.assertEqual([model.body(i).name for i in range(model.nbody)], ["world", "base", "lid"])
            self.assertEqual(model.ngeom, 4)  # two visuals + two collisions
            sub = d / "sub"
            sub.mkdir()
            text = P.with_mujoco_compiler((d / "model.urdf").read_text(), meshdir="../")
            (sub / "x.urdf").write_text(text)  # a file's own relative meshdir is taken from the file
            self.assertEqual(doctor.load(sub / "x.urdf").nmesh, 2)


class MeshTest(unittest.TestCase):
    def test_obj_twin_has_the_same_triangles(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            v, f = box([-0.123456789, 0, 0], [0.3, 0.2, 0.01])
            write_stl(d / "a.stl", v, f)
            self.assertEqual(P.stl_to_obj(d / "a.stl", d / "a.obj"), 12)
            lines = (d / "a.obj").read_text().splitlines()
            verts = np.array([[float(x) for x in l.split()[1:]] for l in lines if l.startswith("v ")])
            faces = np.array([[int(x) - 1 for x in l.split()[1:]] for l in lines if l.startswith("f ")])
            np.testing.assert_array_equal(verts[faces].astype(np.float32), P.read_stl(d / "a.stl").astype(np.float32))
            self.assertTrue(P.is_twin(d / "a.obj"))
            (d / "b.obj").write_text("v 0 0 0\n")  # someone else's file of that name
            self.assertEqual(P.twin_name("b.stl", lambda n: (d / n).exists() and not P.is_twin(d / n)), "b_twin.obj")
            self.assertEqual(P.twin_name("meshes/a.stl", lambda n: False), "meshes/a.obj")

    def test_ascii_stl(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.stl"
            p.write_text("solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
                         "endloop\nendfacet\nendsolid t\n")
            np.testing.assert_array_equal(P.read_stl(p), [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]])


class PortableTest(unittest.TestCase):
    def test_keeps_every_measured_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            toy(d)
            before = uc.parse_urdf(d / "model.urdf")
            done = P.make_portable(d, "toy")
            after = uc.parse_urdf(d / "model.urdf")
            self.assertEqual(done["twins"], {"meshes/base.stl": "meshes/base.obj", "meshes/lid.stl": "meshes/lid.obj"})
            self.assertEqual(done["inertial_fixed"], ["base"])
            for a, b in zip(before.joints, after.joints):
                self.assertEqual((a.name, a.type, a.parent, a.child, a.lower, a.upper, a.friction, a.damping),
                                 (b.name, b.type, b.parent, b.child, b.lower, b.upper, b.friction, b.damping))
                np.testing.assert_array_equal(a.axis, b.axis)
                np.testing.assert_array_equal(a.origin, b.origin)
            for name in before.links:
                la, lb = before.links[name], after.links[name]
                self.assertEqual(la.mass, lb.mass)
                np.testing.assert_array_equal(la.com, lb.com)
                np.testing.assert_allclose(np.linalg.eigvalsh(la.inertia), np.linalg.eigvalsh(lb.inertia), rtol=1e-12)
                self.assertEqual([g.filename.replace(".stl", ".obj") for g in la.collisions], [g.filename for g in lb.collisions])
            # The base's principal axes now match model.xml's (they were turned 5 deg).
            body = ET.parse(d / "model.xml").getroot().find(".//body[@name='base']/inertial")
            R = P.quat_matrix([float(v) for v in body.get("quat").split()])
            want = R @ np.diag([float(v) for v in body.get("diaginertia").split()]) @ R.T
            np.testing.assert_allclose(after.links["base"].inertia, want, atol=1e-8)
            self.assertGreater(np.abs(before.links["base"].inertia - want).max(), 1e-5)
            text = (d / "model.urdf").read_text()
            self.assertEqual(text.count(P.HEADER_MARK), 1)
            self.assertIn('0.0198 m above the floor', text)
            self.assertEqual(after.mujoco_compiler, {"strippath": "false"})
            self.assertIn(P.HEADER_MARK, (d / "model.xml").read_text())
            mujoco.MjModel.from_xml_path(str(d / "model.xml"))  # the comment doesn't bother MuJoCo
            mujoco.MjModel.from_xml_path(str(d / "model.urdf"))
            P.make_portable(d, "toy")  # again: nothing changes
            self.assertEqual((d / "model.urdf").read_text(), text)

    def test_repack_only_touches_the_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "laptop"
            toy(d)
            (d / "model.json").write_text(json.dumps({"name": "laptop", "bundle": "model.zip"}))
            spec = (d / "model.json").read_bytes()
            with zipfile.ZipFile(d / "model.zip", "w") as z:
                for f in ("model.urdf", "model.xml", "meshes/base.stl", "meshes/lid.stl"):
                    z.write(d / f, f)
                z.writestr("model3d.json", '{"hinge_limits_rad": [-1.8613, 0.1638]}')
            P.repack(d)
            self.assertEqual((d / "model.json").read_bytes(), spec)
            with zipfile.ZipFile(d / "model.zip") as z:
                names = z.namelist()
                self.assertEqual(z.read("model.urdf"), (d / "model.urdf").read_bytes())
                self.assertEqual(z.read("model3d.json"), b'{"hinge_limits_rad": [-1.8613, 0.1638]}')
            self.assertEqual(names, ["model.urdf", "model.xml", "meshes/base.stl", "meshes/lid.stl", "model3d.json",
                                     "meshes/base.obj", "meshes/lid.obj"])
            first = (d / "model.zip").read_bytes()
            P.repack(d)
            with zipfile.ZipFile(d / "model.zip") as z:
                self.assertEqual(len(z.namelist()), len(names))
            self.assertEqual(len((d / "model.zip").read_bytes()), len(first))
            write_stl(d / "meshes/lid.stl", *box([-0.15, -0.01, 0], [0.15, 0, 0.25]))  # a refit lid
            self.assertEqual(P.repack(d)["files"], ["meshes/base.obj", "meshes/lid.obj"])  # twins follow their STL
            with zipfile.ZipFile(d / "model.zip") as z:
                self.assertIn("0.25", z.read("meshes/lid.obj").decode())

    def test_fix_inertial_frames_leaves_other_inertials_alone(self):
        urdf_el = ET.fromstring('<robot name="r"><link name="a"><inertial><origin xyz="0 0 0" rpy="0 0 0"/>'
                                '<mass value="2"/><inertia ixx="3" ixy="0" ixz="0" iyy="2" iyz="0" izz="1.5"/>'
                                '</inertial></link></robot>')
        mjcf_el = ET.fromstring('<mujoco><worldbody><body name="a"><inertial pos="0 0 0" quat="0.7071068 0 0.7071068 0" '
                                'mass="1" diaginertia="3 2 1.5"/></body></worldbody></mujoco>')
        self.assertEqual(P.fix_inertial_frames(urdf_el, mjcf_el), [])  # a different mass: not the same inertial
        mjcf_el.find(".//inertial").set("mass", "2")
        self.assertEqual(P.fix_inertial_frames(urdf_el, mjcf_el), ["a"])
        R = uc.rpy_matrix([float(v) for v in urdf_el.find(".//origin").get("rpy").split()])
        np.testing.assert_allclose(R, P.quat_matrix([0.7071068, 0, 0.7071068, 0]), atol=1e-6)


class ChecksTest(unittest.TestCase):
    def test_inertia_problems(self):
        self.assertEqual(uc.inertia_problems(1.0, np.diag([1, 1, 1.5])), [])
        self.assertTrue(uc.inertia_problems(1.0, np.diag([1, 1, 2.5]))[0].endswith("break A + B >= C"))
        self.assertIn("not positive definite", uc.inertia_problems(1.0, np.diag([1, 0, 1]))[0])
        self.assertIn("not positive", uc.inertia_problems(0.0, np.eye(3))[0])
        self.assertIn("no <inertial>", uc.inertia_problems(None, None)[0])

    def test_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "b"
            (d / "meshes").mkdir(parents=True)
            write_stl(d / "meshes/a.stl", *box([0, 0, 0], [1, 1, 1]))
            (d / "meshes/v.obj").write_text("mtllib v.mtl\nv 0 0 0\n")
            (d / "meshes/v.mtl").write_text("newmtl m\nmap_Kd v.png\n")
            (d / "meshes/s.ply").write_text("ply\n")
            (d / "model.urdf").write_text(
                '<robot name="r"><link name="a">'
                '<visual><geometry><mesh filename="meshes/v.obj"/></geometry></visual>'
                '<visual><geometry><mesh filename="meshes/s.ply"/></geometry></visual>'
                '<visual><geometry><mesh filename="/abs/x.stl"/></geometry></visual>'
                '<visual><geometry><mesh filename="../outside.stl"/></geometry></visual>'
                '<visual><geometry><mesh filename="package://pkg/meshes/a.stl"/></geometry></visual>'
                '<collision><geometry><mesh filename="meshes/a.stl"/></geometry></collision>'
                '<collision><geometry><mesh filename="meshes/gone.stl"/></geometry></collision>'
                '</link></robot>')
            rep = uc.asset_problems(d, d / "model.urdf")
            text = " ".join(rep.problems)
            for want in ("meshes/v.png (named by meshes/v.obj): missing", "/abs/x.stl: absolute path",
                         "../outside.stl: outside the bundle", "meshes/gone.stl: missing"):
                self.assertIn(want, text)
            self.assertEqual(rep.unsupported["pybullet"], ["meshes/s.ply"])
            self.assertEqual(sorted(rep.unsupported["drake"]), ["meshes/a.stl", "meshes/s.ply"])
            self.assertEqual(rep.uris, ["package://pkg/meshes/a.stl"])

    def test_measured_joints_from_each_kind_of_record(self):
        def reader(files):
            return lambda name: files.get(name)
        chain = {"model3d.json": {"joints": [
            {"parent": "base", "child": "rod1", "type": "fixed", "axis_model": [1, 0, 0], "range_deg": [0, 0]},
            {"parent": "rod1", "child": "rod2", "type": "revolute", "axis_model": [0, 2, 0], "range_deg": [2, 89]}]}}
        m = uc.measured_joints(reader(chain))
        self.assertEqual(m["rod1_joint"]["type"], "fixed")
        self.assertAlmostEqual(m["rod2_joint"]["upper"], math.radians(89))
        self.assertEqual(m["rod2_joint"]["axis"], [0.0, 1.0, 0.0])
        drawers = {"model3d.json": {"joints": [{"name": "drawer1_joint", "travel_mm": 248.0}]}}
        self.assertEqual(uc.measured_joints(reader(drawers))["drawer1_joint"]["upper"], 0.248)
        hinge = {"model3d.json": {"hinge_limits_rad": [-1.8, 0.16]}}
        self.assertEqual(uc.measured_joints(reader(hinge))["lid_joint"]["lower"], -1.8)
        planar = {"joints.json": {"joints": [{"parent": "base", "child": "lid", "type": "revolute", "axis": [0, 0],
                                              "q": [0.1, None, -0.05, 1.2]}]}}
        self.assertEqual(uc.measured_joints(reader(planar))["lid_joint"]["lower"], -0.05)
        self.assertEqual(uc.measured_joints(reader({"static.json": {}})), {})
        self.assertIsNone(uc.measured_joints(reader({})))

    def test_compare_joints_catches_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            toy(d, physics_like=False)
            robot = uc.parse_urdf(d / "model.urdf")
            same = {"lid_joint": {"type": "revolute", "lower": -1.8613, "upper": 0.1638, "axis": [1, 0, 0]}}
            self.assertEqual(uc.compare_joints(robot, same), [])
            moved = {"lid_joint": {"type": "revolute", "lower": -1.8613, "upper": 0.2, "axis": [-1, 0, 0]}}
            problems = uc.compare_joints(robot, moved)
            self.assertTrue(any("upper limit" in p for p in problems) and any("axis" in p for p in problems))
            free = {"lid_joint": {"type": "revolute", "lower": -1.8613, "upper": 0.1638, "axis": [-1, 0, 0],
                                  "axis_sign_free": True}}
            self.assertEqual(uc.compare_joints(robot, free), [])
            self.assertEqual(uc.tree_problems(robot), [])
            frames = uc.joint_frames(robot)
            np.testing.assert_allclose(frames["lid_joint"][1], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
