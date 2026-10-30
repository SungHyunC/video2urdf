import unittest
import numpy as np
from video2urdf.texture import depth_visible, lid_mask_on_open_frame, texture_link_mask
from video2urdf.sam import Detection

class TextureVisibilityTest(unittest.TestCase):
    def test_texture_mask_excludes_base_pixels_for_open_lid(self):
        def det(box):
            x0, y0, x1, y1 = box
            return Detection(0, 0, box, np.ones((y1-y0+1, x1-x0+1), bool))
        tracks = {'base': {0: det((0, 5, 9, 9))}, 'plate': {0: det((0, 0, 9, 9))},
                  'keyboard': {0: det((1, 6, 8, 8))}}
        mask = texture_link_mask('lid', 0, tracks, {'lid': ['plate']},
                                 {'base': 'base', 'plate': 'plate', 'open_indicator': 'keyboard'}, 10, 10)
        self.assertTrue(mask[:5].all())
        self.assertFalse(mask[5:].any())
        base_mask = texture_link_mask('base', 0, tracks, {'base': ['base']}, {}, 10, 10)
        self.assertTrue(base_mask[5:].all())

    def test_lid_candidate_that_slides_onto_base_is_discarded(self):
        base = np.zeros((10, 10), bool); base[2:8, 2:8] = True
        self.assertFalse(lid_mask_on_open_frame(base.copy(), base).any())
        lid = np.zeros((10, 10), bool); lid[:2, :2] = True
        np.testing.assert_array_equal(lid_mask_on_open_frame(lid, base), lid)
        mixed = lid | base
        np.testing.assert_array_equal(lid_mask_on_open_frame(mixed, base), lid)

    def test_occluded_surface_with_same_geom_is_not_observed(self):
        visible = depth_visible(np.array([.6, .8, .6, -1]), np.array([.606, .6, np.inf, .6]))
        np.testing.assert_array_equal(visible, [True, False, False, False])

if __name__ == '__main__': unittest.main()
