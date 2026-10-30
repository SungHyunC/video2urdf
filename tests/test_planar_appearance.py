import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from scripts.infer_appearance import infer_appearance
from scripts.video_to_model import main
from video2urdf.texture import link_masks
from video2urdf.agent import Agent
from video2urdf.assessment import assess


class PlanarAppearanceTest(unittest.TestCase):
    def test_classified_planar_fit_preserves_model_and_flags_assumptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'joints.json').write_text('{"joints": []}')
            (path / 'parts.json').write_text(json.dumps({'0': {'prompt': 'base plate'}, '1': {'name': 'upper arm'}}))
            (path / 'model.xml').write_text('<mujoco/>')
            (path / 'appearance.json').write_text(json.dumps({'object': 'monitor stand', 'category': 'other'}))
            self.assertEqual(link_masks(path), {'base_plate': ['base_plate'], 'upper_arm': ['upper_arm']})
            infer_appearance(path, path / 'video.mp4', reuse=True, texture=False)
            self.assertEqual((path / 'model.xml').read_text(), '<mujoco/>')
            self.assertEqual(json.loads((path / 'quality.json').read_text())['status'], 'review')
            self.assertEqual(json.loads((path / 'shape_inference.json').read_text())['category'], 'other')
            with self.assertRaisesRegex(ValueError, 'do not have depth'):
                infer_appearance(path, path / 'video.mp4', reuse=True, texture=True)

    def test_video_entrypoint_routes_planar_video_to_agent_and_viewer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / 'stand.mp4'
            video.write_bytes(b'video')
            args = ['video_to_model.py', str(video), '--planar', '--sam', 'open',
                    '--out', str(root / 'runs'), '--viewer-out', str(root / 'viewer')]
            with patch('sys.argv', args), patch('scripts.video_to_model.subprocess.run') as run:
                main()
            self.assertEqual(run.call_count, 2)
            agent = run.call_args_list[0].args[0]
            viewer = run.call_args_list[1].args[0]
            self.assertIn('run_agent.py', agent[1])
            self.assertIn('--appearance', agent)
            self.assertIn('export_viewer.py', viewer[1])
            self.assertNotIn('video_to_capture.py', ' '.join(agent + viewer))

    def test_planar_quality_flags_low_part_replay_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'joints.json').write_text('{"joints": []}')
            np.savez(path / 'planar_replay.npz', root_poses=np.eye(3)[None],
                     iou_arm=np.array([.5, np.nan, .6]), iou_base=np.array([.9, .8]))
            warnings = assess(path)['warnings']
            self.assertTrue(any('arm: mean planar replay mask IoU is 0.55 over 2/3' in w for w in warnings))
            self.assertFalse(any('base: mean planar replay' in w for w in warnings))

    def test_reusing_completed_planar_appearance_keeps_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'joints.json').write_text('{"joints": []}')
            (path / 'parts.json').write_text(json.dumps({'0': {'name': 'base'}}))
            (path / 'appearance.json').write_text(json.dumps({'object': 'articulated stand', 'category': 'other'}))
            original = {'active': True, 'object': 'articulated stand', 'inferred_visuals': [{'link': 'base'}],
                        'shape': 'previously checked', 'visual_coverage_check': {'links': {}}}
            (path / 'shape_inference.json').write_text(json.dumps(original))
            with patch('video2urdf.planar_shape.complete_stand_base') as complete:
                infer_appearance(path, path / 'video.mp4', reuse=True, texture=False)
                complete.assert_not_called()
            saved = json.loads((path / 'shape_inference.json').read_text())
            self.assertTrue(saved['active'])
            self.assertEqual(saved['visual_coverage_check'], original['visual_coverage_check'])

    def test_depth_agent_only_allows_explicit_multijoint_planar_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = object.__new__(Agent)
            agent.tracked, agent.capture, agent.track_dir = True, Path(tmp), Path(tmp)
            agent.video, agent.size, agent.px_per_m = Path(tmp) / 'video.mp4', (640, 480), None
            agent.last_fit = None
            agent.show = lambda *args, **kwargs: None
            agent.replay_sheet = lambda result: np.zeros((2, 2, 3), dtype=np.uint8)
            fitted = SimpleNamespace(summary=lambda: 'planar result')
            with patch('video2urdf.agent.fit', return_value=fitted) as fit:
                self.assertIn('at least two', agent.tool_fit(parents={'lid': 'base'}))
                fit.assert_not_called()
                response = agent.tool_fit(parents={'arm': 'base', 'top': 'arm'})
                fit.assert_called_once()
            self.assertIn('ASSUMED', response)
            self.assertIs(agent.last_fit, fitted)


if __name__ == '__main__':
    unittest.main()
