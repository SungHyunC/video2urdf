import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from video2urdf import design as D
from video2urdf.emit3d import Link3D, Visual, mjcf, urdf
from video2urdf.shape import write_stl

LAMP = {"object": "lamp", "links": [
    {"link": "base", "primitives": [
        {"name": "plate", "type": "rounded_box", "center": [0, 0, -10], "size": [120, 80, 20], "corner_radius": 8, "rgb": [1, 1, 1]},
        {"name": "knob", "type": "sphere", "center": [0, 0, 5], "radius": 6, "rgb": [.5, .5, .5]}]},
    {"link": "arm", "primitives": [
        {"name": "pole", "type": "cylinder", "from": [0, 0, 0], "to": [0, 0, 200], "radius": 5, "rgb": [.8, .8, .8]},
        {"name": "head", "type": "bar", "from": [0, 0, 200], "to": [100, 0, 200], "width": 30, "thickness": 8, "rgb": [.9, .9, .9]},
        {"name": "panel", "type": "box", "center": [50, 0, 190], "size": [80, 20, 4], "euler_deg": [0, 0, 90], "rgb": [1, 1, .8]}]}]}


def toy_model(d: Path):
    """A measured-looking model: a base slab with a hinged arm, as emit3d writes it."""
    (d / "meshes").mkdir()
    base = D.primitive_mesh({"type": "box", "center": [0, 0, -10], "size": [120, 80, 20]})
    arm = D.primitive_mesh({"type": "box", "center": [0, 0, 100], "size": [10, 10, 200]})
    write_stl(d / "meshes/base.stl", base[0] / 1000, base[1])
    write_stl(d / "meshes/arm.stl", arm[0] / 1000, arm[1])
    links = [Link3D("base", [Visual("meshes/base.stl", (.9, .9, .9))]),
             Link3D("arm", [Visual("meshes/arm.stl", (.7, .7, .7))], parent="base", joint_type="revolute",
                    axis=np.array([0, 1.0, 0]), limits=(-1.0, 1.0))]
    (d / "model.xml").write_text(mjcf("toy", links))
    (d / "model.urdf").write_text(urdf("toy", links))


class DesignTest(unittest.TestCase):
    def test_primitive_extents(self):
        v, f = D.primitive_mesh({"type": "rounded_box", "center": [1, 2, 3], "size": [40, 20, 10], "corner_radius": 3})
        np.testing.assert_allclose(v.max(0) - v.min(0), [40, 20, 10], atol=0.3)
        np.testing.assert_allclose((v.max(0) + v.min(0)) / 2, [1, 2, 3], atol=0.3)
        v, _ = D.primitive_mesh({"type": "cylinder", "from": [0, 0, 0], "to": [100, 0, 0], "radius": 5})
        np.testing.assert_allclose(v.max(0) - v.min(0), [100, 10, 10], atol=0.1)
        v, _ = D.primitive_mesh({"type": "bar", "from": [0, 0, 0], "to": [0, 0, 50], "width": 20, "thickness": 4})
        np.testing.assert_allclose(sorted(v.max(0) - v.min(0)), [4, 20, 50], atol=1e-6)
        v, _ = D.primitive_mesh({"type": "box", "center": [0, 0, 0], "size": [10, 20, 30], "euler_deg": [0, 0, 90]})
        np.testing.assert_allclose(v.max(0) - v.min(0), [20, 10, 30], atol=1e-6)

    def test_validation_names_problems(self):
        self.assertEqual(D.validate_design(LAMP, ["base", "arm"]), [])
        bad = {"links": [{"link": "base", "primitives": [{"name": "p", "type": "box", "center": [0, 0, 0], "size": [1, -1, 1]}]},
                         {"link": "ghost", "primitives": []}]}
        problems = D.validate_design(bad, ["base", "arm"])
        self.assertTrue(any("'arm'" in p for p in problems))
        self.assertTrue(any("ghost" in p for p in problems))
        self.assertTrue(any("size must be positive" in p for p in problems))

    def test_apply_keeps_measured_collision_and_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            toy_model(d)
            original = (d / "model.xml").read_text()
            D.apply_design(d, LAMP)
            model = mujoco.MjModel.from_xml_path(str(d / "model.xml"))
            names = [model.geom(g).name for g in range(model.ngeom)]
            self.assertEqual(sum(n.startswith("design_") for n in names), 5)
            for g in range(model.ngeom):
                if model.geom(g).name.startswith("design_"):
                    self.assertEqual(model.geom_contype[g], 0)
                    self.assertEqual(model.geom_group[g], 0)
                else:
                    self.assertEqual(model.geom_group[g], 3)  # measured: hidden, still collides
                    self.assertEqual(model.geom_contype[g], 1)
            urdf_root = ET.parse(d / "model.urdf").getroot()
            arm = next(l for l in urdf_root.findall("link") if l.get("name") == "arm")
            self.assertEqual(len(arm.findall("visual")), 3)
            self.assertEqual(arm.find("collision/geometry/mesh").get("filename"), "meshes/arm.stl")
            self.assertTrue((d / "meshes/design_base_0.stl").exists())
            # applying again starts from the saved measured model, not on top of the design
            D.apply_design(d, LAMP)
            model = mujoco.MjModel.from_xml_path(str(d / "model.xml"))
            self.assertEqual(sum(model.geom(g).name.startswith("design_") for g in range(model.ngeom)), 5)
            self.assertEqual((d / "model.predesign.xml").read_text(), original)

    def test_evidence_in_link_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            toy_model(d)
            ev = D.evidence(d)
            arm = next(l for l in ev["links"] if l["link"] == "arm")
            self.assertEqual(arm["joint"]["type"], "revolute")
            np.testing.assert_allclose(arm["measured_bbox_mm"]["size"], [10, 10, 200], atol=3)
            sizes = D.bbox_check(d, LAMP, ev)
            self.assertAlmostEqual(sizes["arm"]["z"]["design"], 204, delta=1)
            views = D.three_views(d, LAMP, ["base", "arm"])
            self.assertEqual(views.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
