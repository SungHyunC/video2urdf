import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from video2urdf.agent import Agent, capture_depth_note

class AgentLookTest(unittest.TestCase):
    def test_depth_source_note_distinguishes_rgb_estimates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIn('RGB-D capture', capture_depth_note(root))
            (root / 'mapanything.json').write_text('{}')
            self.assertIn('not measured by LiDAR', capture_depth_note(root))
            self.assertIn('estimates', capture_depth_note(root))

    def test_long_request_samples_both_ends(self):
        agent = object.__new__(Agent)
        agent.n = 800
        agent.video = 'unused'
        agent.show = lambda *args, **kwargs: None
        seen = []
        def frame(video, t):
            seen.append(t)
            return np.zeros((10, 10, 3), dtype=np.uint8)
        with patch('video2urdf.agent.read_frame', side_effect=frame):
            response = agent.tool_look([120, 150, 180, 220, 550, 600, 650, 750])
        self.assertEqual(seen, [120, 180, 550, 750])
        self.assertIn('Sampled 4 of 8', response)

    def test_empty_request_has_actionable_response(self):
        agent = object.__new__(Agent)
        self.assertIn('at least one', agent.tool_look([]))
