import mujoco
import numpy as np
import pytest

from video2urdf.camera import PinholeCamera
from video2urdf.mjcf import Scene, scene_xml
from video2urdf.model import ArticulatedModel, Joint, Link, forward_kinematics, matrix_to_quat, rpy_to_matrix
from video2urdf.synth.presets import PRESETS, build


@pytest.mark.parametrize(
    "rpy",
    [
        (0, 0, 0),
        (0.3, -1.2, 2.0),
        (0, np.pi, 0),
        (np.pi, 0, 0),
        (0, 0, np.pi),
        (np.pi / 2, np.pi, -np.pi / 2),
        (2.9, 0.1, -3.0),
    ],
)
def test_quat_matches_mujoco(rpy):
    rot = rpy_to_matrix(rpy)
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, matrix_to_quat(rot))
    np.testing.assert_allclose(out.reshape(3, 3), rot, atol=1e-12)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_forward_kinematics_matches_mujoco(preset):
    rng = np.random.default_rng(1)
    model = build(preset, rng).model
    cam = PinholeCamera.look_at((0, -1, 0), (0, 0, 0), 64, 64, 45.0)
    m = mujoco.MjModel.from_xml_string(scene_xml(model, Scene(cam, floor_z=None)))
    d = mujoco.MjData(m)
    for _ in range(5):
        q = {j.name: rng.uniform(j.lower, j.upper) for j in model.joints}
        for name, value in q.items():
            d.qpos[m.jnt_qposadr[m.joint(name).id]] = value
        mujoco.mj_kinematics(m, d)
        poses = forward_kinematics(model, q)
        for link in model.links:
            b = m.body(link.name).id
            np.testing.assert_allclose(poses[link.name][:3, 3], d.xpos[b], atol=1e-9)
            np.testing.assert_allclose(poses[link.name][:3, :3], d.xmat[b].reshape(3, 3), atol=1e-9)


def test_model_roundtrip_and_validation():
    model = build("lamp", np.random.default_rng(0)).model
    assert ArticulatedModel.from_dict(model.to_dict()) == model
    links = [Link("a"), Link("b")]
    with pytest.raises(ValueError, match="root"):
        ArticulatedModel("two_roots", links + [Link("c")], [Joint("j", "revolute", "a", "b", (0, 0, 0))])
    with pytest.raises(ValueError, match="unknown type"):
        ArticulatedModel("bad", links, [Joint("j", "ball", "a", "b", (0, 0, 0))])
    with pytest.raises(ValueError, match="lower > upper"):
        ArticulatedModel("bad", links, [Joint("j", "revolute", "a", "b", (0, 0, 0), lower=1, upper=0)])
