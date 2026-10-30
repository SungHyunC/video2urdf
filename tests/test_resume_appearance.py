import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.resume_appearance import resume


class ResumeAppearanceTest(unittest.TestCase):
    def test_resume_preserves_source_and_records_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'fit'
            source.mkdir()
            for name in ('model.xml', 'model3d.json', 'replay3d.npz', 'parts.json', 'tracks.sam.npz'):
                (source / name).write_text('original')
            (source / 'measurement_source.json').write_text('{}')
            video = root / 'video.mp4'
            video.write_bytes(b'video')
            appearance = root / 'appearance.json'
            appearance.write_text(json.dumps({'object': 'laptop', 'category': 'laptop'}))
            def infer(stage, *_args, **_kwargs):
                (stage / 'model.xml').write_text('changed')
            with patch('scripts.resume_appearance.infer_appearance', side_effect=infer), \
                 patch('scripts.resume_appearance.assess'):
                record = resume(source, video, root / 'new', appearance)
            self.assertEqual(record['classification'], 'reused')
            self.assertEqual((source / 'model.xml').read_text(), 'original')
            self.assertEqual((root / 'new' / 'model.xml').read_text(), 'changed')
            self.assertEqual(json.loads((root / 'new' / 'resume.json').read_text())['classification'], 'reused')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                resume(source, video, root / 'new', appearance)

    def test_failed_resume_does_not_publish_partial_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'fit'
            source.mkdir()
            for name in ('model.xml', 'model3d.json', 'replay3d.npz', 'parts.json', 'tracks.sam.npz', 'measurement_source.json'):
                (source / name).write_text('{}')
            video = root / 'video.mp4'
            video.write_bytes(b'video')
            with patch('scripts.resume_appearance.infer_appearance', side_effect=ValueError('failure')):
                with self.assertRaisesRegex(ValueError, 'failure'):
                    resume(source, video, root / 'new')
            self.assertFalse((root / 'new').exists())
