"""URDF collision shapes trimmed to clear at a planar joint (video2urdf/joint_clearance.py),
as the export applies them (portable.make_portable)."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from video2urdf import joint_clearance as jc
from video2urdf import portable as P
from video2urdf import urdf_checks as uc
from video2urdf.shape import extrude, write_stl

# A side-view laptop as the planar fit writes it (x along the base, z up, extruded along y):
# the base runs 15 mm past the hinge, the lid reaches 12 mm below it, and closed (q = 1.45) the
# lid's inner face dips into the base's top, which rises toward the front.
BASE = np.array([[-0.15, -0.008], [0.015, -0.008], [0.015, 0.0], [0.0, 0.004], [-0.06, 0.012], [-0.15, 0.012]])
LID = np.array([[-0.004, -0.012], [0.003, -0.012], [0.003, 0.2], [-0.006, 0.2]])
HINGE = (0.0, 0.002)
INERTIAL = {"base": ('<origin xyz="-0.06 0 0.002" rpy="0 0 0"/><mass value="0.9"/>'
                     '<inertia ixx="0.003" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.004"/>'),
            "lid": ('<origin xyz="0 0 0.09" rpy="0 0 0"/><mass value="0.4"/>'
                    '<inertia ixx="0.002" ixy="0" ixz="0" iyy="0.0015" iyz="0" izz="0.0008"/>')}


def mesh(d: Path, name: str, poly: np.ndarray, depth: float = 0.3) -> str:
    rel = f"meshes/{name}.stl"
    (d / "meshes").mkdir(exist_ok=True)
    write_stl(d / rel, *extrude(poly, depth))
    return rel


def link(name: str, rel: str) -> str:
    return (f'<link name="{name}"><visual><geometry><mesh filename="{rel}"/></geometry></visual>'
            f'<inertial>{INERTIAL.get(name, INERTIAL["lid"])}</inertial>'
            f'<collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{rel}"/></geometry></collision></link>')


def joint(name: str, parent: str, child: str, xz, lower: float, upper: float) -> str:
    return (f'<joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{xz[0]} 0 {xz[1]}" rpy="0 0 0"/><axis xyz="0 -1 0"/>'
            f'<limit lower="{lower}" upper="{upper}" effort="10" velocity="2"/></joint>')


def laptop(d: Path, lid=LID, hinge=HINGE, upper=1.45) -> None:
    text = ('<robot name="laptop"><mujoco><compiler strippath="false"/></mujoco>'
            + link("base", mesh(d, "base", BASE)) + link("lid", mesh(d, "lid", lid))
            + joint("lid_joint", "base", "lid", hinge, -0.1, upper) + "</robot>")
    (d / "model.urdf").write_text(text)


def closest(urdf: Path, steps: int = 120) -> float:
    """MuJoCo's URDF import: the least distance between colliding geoms of different bodies
    over each joint's range (negative: they overlap)."""
    m = mujoco.MjModel.from_xml_path(str(urdf))
    d = mujoco.MjData(m)
    geoms = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    pairs = [(a, b) for a in geoms for b in geoms if a < b and m.geom_bodyid[a] != m.geom_bodyid[b]]
    worst = np.inf
    for j in range(m.njnt):
        for q in np.linspace(*m.jnt_range[j], steps):
            d.qpos[:] = 0
            d.qpos[m.jnt_qposadr[j]] = q
            mujoco.mj_forward(m, d)
            worst = min([worst] + [mujoco.mj_geomDistance(m, d, a, b, 0.05, None) for a, b in pairs])
    return worst


def inside(points: np.ndarray, convex: np.ndarray, tol: float = 1e-9) -> bool:
    n = len(convex)
    return all(jc._cross(convex[i], convex[(i + 1) % n], p) >= -tol for p in points for i in range(n))


class GeometryTest(unittest.TestCase):
    def test_hull_clip_merge(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        np.testing.assert_allclose(jc.area(jc.hull(np.vstack([sq, [[0.5, 0.5]]]))), 1.0)
        c = jc.clip_convex(sq + 0.5, sq)
        self.assertAlmostEqual(jc.area(c), 0.25)
        self.assertEqual(len(jc.clip_convex(sq + 2, sq)), 0)
        halves = [np.array([[0, 0], [1, 0], [1, 1]], float), np.array([[0, 0], [1, 1], [0, 1]], float)]
        self.assertEqual(len(jc.merge_convex(halves)), 1)  # a square again
        ell = [np.array([[0, 0], [2, 0], [2, 1], [0, 1]], float), np.array([[0, 1], [1, 1], [1, 2], [0, 2]], float)]
        self.assertEqual(len(jc.merge_convex(ell)), 2)  # an L stays two pieces

    def test_welded_to_root(self):
        root = ET.fromstring('<robot><link name="a"/><link name="b"/><link name="c"/><link name="d"/>'
                             '<joint name="j1" type="fixed"><parent link="a"/><child link="b"/></joint>'
                             '<joint name="j2" type="revolute"><parent link="b"/><child link="c"/></joint>'
                             '<joint name="j3" type="fixed"><parent link="c"/><child link="d"/></joint></robot>')
        self.assertEqual(jc.welded_to_root(root), {"a", "b"})


class TrimTest(unittest.TestCase):
    def test_overlapping_lid_is_cleared_over_its_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            laptop(d)
            before = uc.parse_urdf(d / "model.urdf")
            self.assertLess(closest(d / "model.urdf"), -0.005)  # the lid starts deep inside the base
            done = P.make_portable(d, "laptop")
            self.assertEqual(len(done["trimmed"]), 1)
            after = uc.parse_urdf(d / "model.urdf")
            trimmed = jc.trimmed_links(ET.fromstring(P._strip_header((d / "model.urdf").read_text())))
            self.assertEqual(len(trimmed), 1)
            name, other = trimmed[0], ({"base", "lid"} - set(trimmed)).pop()
            # Clear by about CLEARANCE everywhere in the measured range, as MuJoCo sees it.
            self.assertGreater(closest(d / "model.urdf"), 0.8 * jc.CLEARANCE)
            # Only the trimmed link's collision changed: inertials, visuals, joints, the other link.
            for n in ("base", "lid"):
                self.assertEqual(after.links[n].mass, before.links[n].mass)
                np.testing.assert_array_equal(after.links[n].inertia, before.links[n].inertia)
                np.testing.assert_array_equal(after.links[n].com, before.links[n].com)
                self.assertEqual([Path(v.filename).stem for v in after.links[n].visuals],  # (OBJ twins now)
                                 [Path(v.filename).stem for v in before.links[n].visuals])
            self.assertEqual([c.filename for c in after.links[other].collisions], [f"meshes/{other}.obj"])
            np.testing.assert_array_equal(after.joints[0].origin, before.joints[0].origin)
            self.assertEqual((after.joints[0].lower, after.joints[0].upper), (before.joints[0].lower, before.joints[0].upper))
            # Each piece is convex (so a loader's hull is the piece) and inside the original hull.
            orig = jc.hull(np.vstack([BASE if name == "base" else LID]))
            pieces = [jc.hull(jc.read_mesh(d / c.filename)[:, [0, 2]]) for c in after.links[name].collisions]
            self.assertGreater(len(pieces), 0)
            for p in pieces:
                self.assertTrue(inside(p, orig))
            kept = sum(jc.area(p) for p in pieces) / jc.area(orig)
            self.assertGreater(kept, 1 - jc.MAX_REMOVED)
            self.assertLess(kept, 0.995)
            # The same extrusion along y.
            v = np.vstack([jc.read_mesh(d / c.filename) for c in after.links[name].collisions])
            np.testing.assert_allclose([v[:, 1].min(), v[:, 1].max()], [-0.15, 0.15], atol=1e-6)
            # The header says so; a second export changes nothing.
            self.assertIn("_clear_", (d / "model.urdf").read_text().split("-->")[0])
            text = (d / "model.urdf").read_text()
            files = sorted(p.name for p in (d / "meshes").iterdir())
            again = P.make_portable(d, "laptop")
            self.assertEqual(again["trimmed"], [])
            self.assertEqual((d / "model.urdf").read_text(), text)
            self.assertEqual(sorted(p.name for p in (d / "meshes").iterdir()), files)

    def test_the_lid_no_longer_thrown_in_mujoco(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            laptop(d)
            P.make_portable(d, "laptop")
            m = mujoco.MjModel.from_xml_path(str(d / "model.urdf"))
            data = mujoco.MjData(m)
            mujoco.mj_forward(m, data)
            self.assertEqual(data.ncon, 0)  # nothing touches at the start
            m.dof_frictionloss[:] = 1.0  # hold it against gravity, as the measured friction does
            for _ in range(int(1.0 / m.opt.timestep)):
                mujoco.mj_step(m, data)
            self.assertLess(abs(data.qpos[0]), np.radians(2))

    def test_clear_parts_are_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            lid = LID.copy()
            lid[:, 1] = np.clip(lid[:, 1], 0.03, None)  # starts 30 mm up its own axis: never meets the base
            laptop(d, lid=lid, upper=0.5)
            self.assertGreater(closest(d / "model.urdf"), 0)
            done = P.make_portable(d, "laptop")
            self.assertEqual(done["trimmed"], [])
            r = uc.parse_urdf(d / "model.urdf")
            self.assertEqual([c.filename for c in r.links["base"].collisions], ["meshes/base.obj"])
            self.assertFalse(any("_clear_" in p.name for p in (d / "meshes").iterdir()))

    def test_joints_below_the_root_are_left_to_the_loader(self):
        """MuJoCo filters parent-child contacts unless the parent is fused into the world."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            stand = np.array([[-0.5, -0.5], [-0.45, -0.5], [-0.45, -0.45], [-0.5, -0.45]])
            text = ('<robot name="chain"><mujoco><compiler strippath="false"/></mujoco>'
                    + link("stand", mesh(d, "stand", stand)) + link("base", mesh(d, "base", BASE))
                    + link("lid", mesh(d, "lid", LID))
                    + joint("base_joint", "stand", "base", (0, 0), -0.01, 0.01)
                    + joint("lid_joint", "base", "lid", HINGE, -0.1, 1.45) + "</robot>")
            (d / "model.urdf").write_text(text)
            self.assertEqual(jc.plan(d, ET.fromstring(text)), [])


if __name__ == "__main__":
    unittest.main()
