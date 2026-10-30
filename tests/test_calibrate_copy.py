import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.calibrate_capture import calibrate_copy


class CalibrateCopyTest(unittest.TestCase):
    def test_copy_is_calibrated_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir(); dest = root / 'new'
            (source / 'mapanything.json').write_text(json.dumps({'scale_applied': 2.0}))
            (source / 'depth-marker').write_text('original')
            reference = {'scale_factor': .5}
            def rescale(path, factor):
                (path / 'depth-marker').write_text('scaled')
            with patch('scripts.calibrate_capture.reference_scale', return_value=reference), \
                 patch('scripts.calibrate_capture.check_references', return_value=[]), \
                 patch('scripts.calibrate_capture.rescale_capture', side_effect=rescale):
                calibrate_copy(source, dest, [0, 1, 1, 2, 2, 100])
            self.assertEqual((source / 'depth-marker').read_text(), 'original')
            self.assertEqual((dest / 'depth-marker').read_text(), 'scaled')
            self.assertEqual(json.loads((dest / 'mapanything.json').read_text())['scale_applied'], 1.)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                calibrate_copy(source, dest, [0, 1, 1, 2, 2, 100])

    def test_failed_copy_does_not_leave_a_partial_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir(); dest = root / 'new'
            (source / 'mapanything.json').write_text('{}')
            with patch('scripts.calibrate_capture.reference_scale', side_effect=ValueError('bad reference')):
                with self.assertRaisesRegex(ValueError, 'bad reference'):
                    calibrate_copy(source, dest, [0, 1, 1, 2, 2, 100])
            self.assertFalse(dest.exists())
            self.assertEqual((source / 'mapanything.json').read_text(), '{}')

if __name__ == '__main__': unittest.main()
