import numpy as np

from video2urdf.camera import PinholeCamera


def test_look_at_target_projects_to_principal_point():
    cam = PinholeCamera.look_at((0.3, -1.2, 0.5), (0.1, 0.0, 0.2), 1280, 720, 45.0)
    np.testing.assert_allclose(cam.project(np.array([0.1, 0.0, 0.2])), [639.5, 359.5], atol=1e-9)
    np.testing.assert_allclose(cam.position, [0.3, -1.2, 0.5], atol=1e-12)


def test_image_axes_follow_world_axes_for_a_facing_camera():
    cam = PinholeCamera.look_at((0, -1, 0), (0, 0, 0), 640, 480, 60.0)
    center, right, up = cam.project(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0, 0.1]]))
    assert right[0] > center[0] and abs(right[1] - center[1]) < 1e-9
    assert up[1] < center[1] and abs(up[0] - center[0]) < 1e-9


def test_plane_homography_matches_projection():
    rng = np.random.default_rng(0)
    cam = PinholeCamera.look_at((0.2, -0.9, 0.4), (0.0, 0.0, 0.25), 1280, 720, 50.0)
    H = cam.plane_homography((0, 0, 0), (1, 0, 0), (0, 0, 1))
    st = rng.uniform(-0.3, 0.3, size=(20, 2))
    world = np.stack([st[:, 0], np.zeros(20), st[:, 1]], axis=1)
    h = np.concatenate([st, np.ones((20, 1))], axis=1) @ H.T
    np.testing.assert_allclose(h[:, :2] / h[:, 2:], cam.project(world), atol=1e-9)


def test_dict_roundtrip():
    cam = PinholeCamera.look_at((0.2, -0.9, 0.4), (0.0, 0.0, 0.25), 320, 240, 40.0)
    back = PinholeCamera.from_dict(cam.to_dict())
    np.testing.assert_allclose(back.K, cam.K)
    np.testing.assert_allclose(back.R, cam.R)
    np.testing.assert_allclose(back.t, cam.t)
