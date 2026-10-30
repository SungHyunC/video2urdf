import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scripts.infer_appearance import validate_classification, infer_appearance, object_crops
from video2urdf.sam import Detection

class ClassificationTest(unittest.TestCase):
    def test_static_crops_follow_target_not_full_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'static.json').write_text(json.dumps({'part': 'object'}))
            tracks = {'object': {t: Detection(t, 0, (40, 30, 59, 49), np.ones((20, 20), bool)) for t in (0, 5, 9)}}
            seen = []
            def image_url(image):
                seen.append(image.shape)
                return 'data:image/jpeg;base64,test'
            with patch('scripts.infer_appearance.load_parts', return_value=(['object'], tracks)), \
                 patch('scripts.infer_appearance.video_info', return_value=((100, 80), 10, 30)), \
                 patch('scripts.infer_appearance.read_frame', return_value=np.zeros((80, 100, 3), np.uint8)), \
                 patch('scripts.infer_appearance.jpeg_url', side_effect=image_url):
                crops = object_crops(root, root / 'video.mp4')
            self.assertEqual(len(crops), 3)
            self.assertTrue(all(shape[0] < 80 and shape[1] < 100 for shape in seen))

    def test_articulated_crops_include_base_and_moving_part_but_not_floor(self):
        tracks = {
            'base': {0: Detection(0, 0, (20, 40, 49, 59), np.ones((20, 30), bool))},
            'lid': {0: Detection(0, 0, (20, 10, 49, 39), np.ones((30, 30), bool))},
            'floor': {0: Detection(0, 0, (0, 0, 99, 79), np.ones((80, 100), bool))},
        }
        seen = []
        with patch('scripts.infer_appearance.load_parts', return_value=(list(tracks), tracks)), \
             patch('scripts.infer_appearance.link_masks', return_value={'base': ['base'], 'lid': ['lid']}), \
             patch('scripts.infer_appearance.video_info', return_value=((100, 80), 1, 30)), \
             patch('scripts.infer_appearance.read_frame', return_value=np.zeros((80, 100, 3), np.uint8)), \
             patch('scripts.infer_appearance.jpeg_url', side_effect=lambda image: seen.append(image.shape) or 'data:test'):
            self.assertEqual(len(object_crops(Path('/unused'), Path('/unused.mp4'))), 1)
        self.assertEqual(seen[0], (58, 38, 3))

    def test_invalid_response_is_retried_before_appearance_is_written(self):
        valid = {'object': 'cabinet', 'category': 'drawer', 'drawer_count': 3,
                 'drawer_has_handles': True, 'links': [{'link': 'base', 'material': 'wood', 'reason': 'wood'}]}
        responses = [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
                     for text in ('{"object":', json.dumps(valid))]
        calls = SimpleNamespace(create=unittest.mock.Mock(side_effect=responses))
        client = SimpleNamespace(chat=SimpleNamespace(completions=calls))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model.xml').write_text('<mujoco><worldbody/></mujoco>')
            shape = {'category': 'drawer', 'inferred_visuals': []}
            with patch('scripts.infer_appearance.OpenAI', return_value=client), \
                 patch('scripts.infer_appearance.link_masks', return_value={'base': []}), \
                 patch('scripts.infer_appearance.mp4_url', return_value='data:video/mp4;base64,'), \
                 patch('scripts.infer_appearance.VIDEO', True), \
                 patch.dict('os.environ', {'META_API_KEY': 'unused', 'BASETEN_API_KEY': 'unused'}), \
                 patch('scripts.infer_appearance.API_KEY_ENV', 'META_API_KEY'), \
                 patch('scripts.infer_appearance.complete_geometry', return_value=shape), \
                 patch('scripts.infer_appearance.refresh_fit_warnings'), \
                 patch('scripts.infer_appearance.assess', return_value={}):
                infer_appearance(root, root / 'video.mp4', texture=False)
            self.assertEqual(calls.create.call_count, 2)
            self.assertEqual(json.loads((root / 'appearance.json').read_text())['object'], 'cabinet')

    def test_accepts_valid_object_and_rejects_invalid_entries(self):
        valid = {'object': 'three-drawer cabinet', 'category': 'drawer', 'drawer_count': 3,
                 'drawer_has_handles': True, 'links': [{'link': 'base', 'material': 'wood', 'reason': 'visible grain'}]}
        self.assertEqual(validate_classification(valid), valid)
        for bad in [dict(valid, links=[{'link': 'base'}]), dict(valid, category='vehicle'),
                    dict(valid, drawer_count=True), dict(valid, drawer_has_handles='yes')]:
            with self.assertRaises(ValueError):
                validate_classification(bad)

if __name__ == '__main__': unittest.main()
