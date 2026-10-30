import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_agent import write_measurement_source


class RunProvenanceTest(unittest.TestCase):
    def test_estimated_depth_is_recorded_even_before_model_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture, result = root / 'capture', root / 'result'
            capture.mkdir()
            (capture / 'mapanything.json').write_text(json.dumps({'scale_applied': .5}))
            write_measurement_source(result, capture)
            source = json.loads((result / 'measurement_source.json').read_text())
            self.assertEqual(source['depth'], 'MapAnything estimated from RGB video')
            self.assertEqual(source['scale_applied'], .5)
            self.assertFalse((result / 'model3d.json').exists())

    def test_plain_video_is_not_mislabeled_as_rgbd(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / 'result'
            write_measurement_source(result, None)
            self.assertFalse(result.exists())
