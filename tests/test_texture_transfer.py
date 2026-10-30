import unittest
import numpy as np
from video2urdf.texture_transfer import transfer_texture, visible_from_depth


class TextureTransferTest(unittest.TestCase):
    def test_only_visible_nearby_vertices_inside_source_triangle_receive_color(self):
        vertices = np.array([[.2, .2, .001], [.3, .2, .001], [.2, .2, .1], [2, 2, 0]])
        original = np.full((4, 3), 7, np.uint8)
        colors, hit = transfer_texture(vertices, original, [True, False, True, True],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0], [1, 0], [0, 1]],
            [[0, 1, 2]], [[0, 1, 2]], np.full((2, 2, 3), [10, 20, 30], np.uint8), .01)
        np.testing.assert_array_equal(hit, [True, False, False, False])
        np.testing.assert_array_equal(colors[0], [30, 20, 10])
        np.testing.assert_array_equal(colors[1:], original[1:])
        np.testing.assert_array_equal(original, np.full((4, 3), 7))

    def test_visibility_rejects_occlusion_and_points_outside_image(self):
        depth = np.ones((4, 4))
        points = np.array([[0, 0, 1], [0, 0, 1.1], [10, 0, 1], [0, 0, -1], [np.nan, 0, 1]])
        np.testing.assert_array_equal(visible_from_depth(points, depth, 1, 1, 1),
                                      [True, False, False, False, False])

    def test_auto_views_include_only_recorded_observed_faces(self):
        import json
        import tempfile
        import xml.etree.ElementTree as ET
        from pathlib import Path
        from unittest.mock import patch
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from transfer_sam3d_texture import observed_views
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'meshes').mkdir()
            (root / 'meshes/screen.stl').touch()
            (root / 'model.xml').write_text('<mujoco><asset><mesh name="lid_1" file="meshes/screen_tex.obj"/></asset></mujoco>')
            (root / 'texture.json').write_text(json.dumps({'meshes': {'lid_1': {
                'by_face': {'1': {'frame': 12, 'triangles': 1}}}}}))
            source = ([[0, 0, 0]] * 6, [[0, 0]] * 6, [[0, 1, 2], [3, 4, 5]],
                      [[0, 1, 2], [3, 4, 5]], np.zeros((1, 1, 3), np.uint8))
            with patch('transfer_sam3d_texture.read_stl'), patch('transfer_sam3d_texture.planar_faces',
                    return_value=[[0], [1]]), patch('transfer_sam3d_texture.read_video_obj', return_value=source):
                views = observed_views(root, 'lid')
            self.assertEqual(views[0][1], 12)
            self.assertEqual(views[0][2][2], [[3, 4, 5]])
