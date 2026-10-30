import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from scripts.video_to_model import laptop_review_candidate


class LaptopReviewPipelineTest(unittest.TestCase):
    def test_candidate_only_for_overclosed_laptop_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'run'
            target.mkdir()
            (target / 'appearance.json').write_text(json.dumps({'category': 'laptop'}))
            (target / 'hinge.json').write_text(json.dumps({'opening_on_reference_deg': 90, 'q': [0.] * 20}))
            with patch('scripts.experimental_laptop_align.align_copy') as align:
                self.assertIsNone(laptop_review_candidate(target, Path('video.mp4')))
                align.assert_not_called()
                (target / 'hinge.json').write_text(json.dumps({'opening_on_reference_deg': 90, 'q': [-2.1] * 20}))
                target.with_name('run_laptop_review').mkdir()
                self.assertIsNone(laptop_review_candidate(target, Path('video.mp4')))
                align.assert_not_called()

    def test_overclosed_laptop_creates_separate_review_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'run'
            target.mkdir()
            (target / 'appearance.json').write_text(json.dumps({'category': 'laptop'}))
            (target / 'hinge.json').write_text(json.dumps({'opening_on_reference_deg': 90, 'q': [-2.1] * 20}))
            (target / 'model.xml').write_text('<mujoco><asset/></mujoco>')
            def create_candidate(source, dest):
                dest.mkdir()
                (dest / 'quality.json').write_text('{"warnings": []}')
            comparison = {'links': {'base': {'frames': 10, 'baseline': .8, 'inferred': .9}}}
            with patch('scripts.experimental_laptop_align.align_copy', side_effect=create_candidate) as align, \
                 patch('video2urdf.appearance_check.compare_visuals', return_value=comparison), \
                 patch('scripts.video_to_model.subprocess.run', return_value=SimpleNamespace(returncode=0)) as render:
                candidate = laptop_review_candidate(target, Path('video.mp4'))
            self.assertEqual(candidate, target.with_name('run_laptop_review'))
            align.assert_called_once_with(target, candidate)
            self.assertIn(str(candidate), render.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
