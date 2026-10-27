"""End-to-end checks of the synthetic generator at low resolution. Needs an OpenGL context (MUJOCO_GL=egl headless)."""

import json

import cv2
import numpy as np
import pytest

from video2urdf.model import REVOLUTE, ArticulatedModel, forward_kinematics, rpy_to_matrix, transform
from video2urdf.synth import SynthConfig, generate
from video2urdf.synth.render import CARD_LABEL

SMALL = dict(width=320, height=180, fps=20, seconds=1.5)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    out = {}
    for preset, seed, extra in [
        ("lamp", 0, {}),
        ("drawer", 0, {}),
        ("scissors", 0, {}),
        ("cart_pendulum", 0, {"tilt_deg": 8}),
        ("random", 3, {}),
    ]:
        folder = tmp_path_factory.mktemp(f"{preset}{seed}")
        gt = generate(SynthConfig(preset=preset, seed=seed, **SMALL, **extra), folder)
        out[preset] = (folder, gt, np.load(folder / "masks.npz")["labels"])
    return out


@pytest.mark.parametrize("preset", ["lamp", "drawer", "scissors", "cart_pendulum", "random"])
def test_outputs_and_limits(rendered, preset):
    folder, gt, labels = rendered[preset]
    for name in ("video.mp4", "masks.npz", "gt.json", "scene.xml", "overlay.png"):
        assert (folder / name).stat().st_size > 0
    assert json.loads((folder / "gt.json").read_text())["frames"] == gt["frames"]
    assert labels.shape == (gt["video"]["num_frames"], 180, 320)
    q = np.array(gt["frames"]["q"])
    for k, joint in enumerate(gt["joints"]):
        # The observed range must equal the GT limits (the estimator reads limits from the observed range).
        assert q[:, k].min() == pytest.approx(joint["lower"], abs=1e-6)
        assert q[:, k].max() == pytest.approx(joint["upper"], abs=1e-6)
    present = set(np.unique(labels).tolist())
    assert {i + 1 for i in range(len(gt["model"]["links"]))} <= present
    assert CARD_LABEL in present


@pytest.mark.parametrize("preset", ["lamp", "drawer", "scissors", "cart_pendulum", "random"])
def test_card_mask_matches_projected_corners(rendered, preset):
    _, gt, labels = rendered[preset]
    corners = np.array(gt["card"]["corners_px"], dtype=np.float32)
    ys, xs = np.nonzero(labels[0] == CARD_LABEL)
    assert len(xs) == pytest.approx(cv2.contourArea(corners), rel=0.1)
    np.testing.assert_allclose([xs.mean(), ys.mean()], corners.mean(axis=0), atol=0.6)


def test_lamp_pivots_land_on_their_parts(rendered):
    _, gt, labels = rendered["lamp"]
    names = [link["name"] for link in gt["model"]["links"]]
    for t in (0, len(labels) // 2, len(labels) - 1):
        for j, joint in enumerate(gt["joints"]):
            u, v = np.round(gt["frames"]["anchor_px"][t][j]).astype(int)
            assert labels[t, v, u] in (names.index(joint["parent"]) + 1, names.index(joint["child"]) + 1)


@pytest.mark.parametrize("preset", ["lamp", "drawer", "cart_pendulum", "random"])
def test_gt_anchors_match_model_kinematics(rendered, preset):
    _, gt, _ = rendered[preset]
    if gt["motion"]["base_motion"]:
        pytest.skip("root drifts")
    model = ArticulatedModel.from_dict(gt["model"])
    q = np.array(gt["frames"]["q"])
    for t in range(0, len(q), 7):
        poses = forward_kinematics(model, {j.name: q[t, k] for k, j in enumerate(model.joints)})
        for k, joint in enumerate(model.joints):
            # Anchor = joint frame origin at q=0, fixed to the parent (pivot, or slide origin for prismatic).
            joint_frame = poses[joint.parent] @ transform(joint.origin_xyz, rpy_to_matrix(joint.origin_rpy))
            np.testing.assert_allclose(joint_frame[:3, 3], gt["frames"]["anchor_world"][t][k], atol=1e-5)


def test_same_seed_same_ground_truth(rendered, tmp_path):
    _, gt, _ = rendered["scissors"]
    again = generate(SynthConfig(preset="scissors", seed=0, **SMALL), tmp_path)
    assert again["frames"] == gt["frames"] and again["camera"] == gt["camera"]


def test_revolute_axes_point_at_the_camera(rendered):
    _, gt, _ = rendered["lamp"]
    axes = np.array(gt["frames"]["axis_world"])
    for k, joint in enumerate(gt["joints"]):
        if joint["type"] == REVOLUTE:
            np.testing.assert_allclose(axes[:, k], np.tile([0, -1, 0], (len(axes), 1)), atol=1e-9)
