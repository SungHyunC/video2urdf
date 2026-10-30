import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from video2urdf.pipeline import load_parts
from video2urdf.sam import Detection, parse, replace_tracked_part
from video2urdf.agent import Agent


class TrackUnionTest(unittest.TestCase):
    def test_failed_full_retrack_preserves_previous_fit_and_tracks(self):
        class EmptySam:
            def track_boxes(self, *args): return ""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'parts.json').write_text('{"previous": true}')
            (root / 'tracks.sam.txt').write_text('previous track')
            agent = object.__new__(Agent)
            agent.track_dir, agent.size, agent.n = root, (20, 20), 2
            agent.video, agent.sam, agent.last_fit = root / 'video.mp4', EmptySam(), {'valid': True}
            d = Detection(0, 0, (1, 1, 9, 9), np.ones((9, 9), bool))
            agent.candidates = {'s1': ('drawer', d, 0)}
            self.assertIn('no masks', agent.tool_track([{'name': 'drawer', 'candidate': 's1'}]))
            self.assertEqual((root / 'tracks.sam.txt').read_text(), 'previous track')
            self.assertEqual((root / 'parts.json').read_text(), '{"previous": true}')
            self.assertEqual(agent.last_fit, {'valid': True})
            tiny = Detection(0, 0, (1, 1, 1, 1), np.ones((1, 1), bool))
            agent.size = (1000, 1000)
            with patch('video2urdf.agent.parse', return_value=[tiny]):
                self.assertIn('usable size', agent.tool_track([{'name': 'drawer', 'candidate': 's1'}]))
            self.assertEqual((root / 'tracks.sam.txt').read_text(), 'previous track')
            self.assertEqual(agent.last_fit, {'valid': True})

    def test_agent_retracks_one_part_and_invalidates_previous_fit(self):
        def rec(oid, x):
            return f"{oid}<|box;x1={x};y1=0;x2={x};y2=0|><|mask;data=1,1,R0.1|>"
        class SamStub:
            def track_boxes(self, video, size, boxes):
                return f"<0f>{rec(0, 8)}\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "parts.json").write_text(json.dumps({"0": {"name": "base"}, "1": {"name": "screen"}}))
            (root / "tracks.sam.txt").write_text(f"<0f>{rec(0, 1)},{rec(1, 2)}\n")
            agent = object.__new__(Agent)
            agent.tracked, agent.track_dir, agent.size = True, root, (10, 10)
            agent.video, agent.sam, agent.last_fit = root / "video.mp4", SamStub(), {"old": True}
            candidate = Detection(0, 0, (8, 0, 8, 0), np.ones((1, 1), bool), .9)
            agent.candidates = {"s1": ("screen", candidate, 0)}
            self.assertIn("Re-tracked screen", agent.tool_retrack_part("screen", ["s1"]))
            result = {(d.frame, d.obj_id): d.box[0] for d in parse((root / "tracks.sam.txt").read_text())}
            self.assertEqual(result, {(0, 0): 1, (0, 1): 8})
            self.assertIsNone(agent.last_fit)

    def test_empty_retrack_keeps_original_tracks_and_fit(self):
        class SamStub:
            def track_boxes(self, video, size, boxes): return ""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = json.dumps({"0": {"name": "screen"}})
            raw = "<0f>0<|box;x1=0;y1=0;x2=0;y2=0|><|mask;data=1,1,R0.1|>\n"
            (root / "parts.json").write_text(metadata)
            (root / "tracks.sam.txt").write_text(raw)
            agent = object.__new__(Agent)
            agent.tracked, agent.track_dir, agent.size = True, root, (10, 10)
            agent.video, agent.sam, agent.last_fit = root / "video.mp4", SamStub(), {"valid": True}
            d = Detection(0, 0, (0, 0, 0, 0), np.ones((1, 1), bool), .9)
            agent.candidates = {"s1": ("screen", d, 0)}
            with self.assertRaisesRegex(ValueError, "no masks"):
                agent.tool_retrack_part("screen", ["s1"])
            self.assertEqual((root / "tracks.sam.txt").read_text(), raw)
            self.assertEqual((root / "parts.json").read_text(), metadata)
            self.assertEqual(agent.last_fit, {"valid": True})

    def test_retracking_replaces_only_requested_object(self):
        def rec(oid, x):
            return f"{oid}<|box;x1={x};y1=0;x2={x};y2=0|><|mask;data=1,1,R0.1|>"
        original = f"<0f>{rec(0, 1)},{rec(2, 2)}\n<1f>{rec(0, 3)},{rec(2, 4)}\n"
        replacement = f"<0f>{rec(0, 8)}\n<2f>{rec(0, 9)}\n"
        result = {(d.frame, d.obj_id): d.box[0] for d in parse(replace_tracked_part(original, replacement, 2))}
        self.assertEqual(result, {(0, 0): 1, (1, 0): 3, (0, 2): 8, (2, 2): 9})
        with self.assertRaisesRegex(ValueError, "no masks"):
            replace_tracked_part(original, "", 2)

    def test_retracking_original_anchors_restores_without_remote_call(self):
        def rec(oid, x):
            return f"{oid}<|box;x1={x};y1=0;x2={x};y2=0|><|mask;data=1,1,R0.1|>"
        class SamStub:
            calls = 0
            def track_boxes(self, video, size, boxes):
                self.calls += 1
                return f"<0f>{rec(0, 8)}\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'parts.json').write_text(json.dumps({'0': {'name': 'base'}, '1': {'name': 'screen', 'anchors': [[0, [2, 0, 2, 0]]]}}))
            (root / 'tracks.sam.txt').write_text(f"<0f>{rec(0, 1)},{rec(1, 2)}\n")
            agent = object.__new__(Agent)
            agent.tracked, agent.track_dir, agent.size = True, root, (10, 10)
            agent.video, agent.sam, agent.last_fit = root / 'video.mp4', SamStub(), None
            agent.candidates = {'new': ('screen', Detection(0, 0, (8, 0, 8, 0), np.ones((1, 1), bool), .9), 0),
                                'old': ('screen', Detection(0, 0, (2, 0, 2, 0), np.ones((1, 1), bool), .9), 0)}
            agent.tool_retrack_part('screen', ['new'])
            result = agent.tool_retrack_part('screen', ['old'])
            self.assertEqual(agent.sam.calls, 1)
            self.assertIn('Restored cached track', result)
            self.assertEqual({d.obj_id: d.box[0] for d in parse((root / 'tracks.sam.txt').read_text())}, {0: 1, 1: 2})

    def test_missing_previous_track_can_be_retracked(self):
        class SamStub:
            def track_boxes(self, video, size, boxes):
                return '<0f>0<|box;x1=8;y1=0;x2=8;y2=0|><|mask;data=1,1,R0.1|>\n'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'parts.json').write_text(json.dumps({'1': {'name': 'screen', 'anchors': [[0, [2, 0, 2, 0]]]}}))
            (root / 'tracks.sam.txt').write_text('')
            agent = object.__new__(Agent)
            agent.tracked, agent.track_dir, agent.size = True, root, (10, 10)
            agent.video, agent.sam, agent.last_fit = root / 'video.mp4', SamStub(), None
            agent.candidates = {'new': ('screen', Detection(0, 0, (8, 0, 8, 0), np.ones((1, 1), bool), .9), 0)}
            agent.tool_retrack_part('screen', ['new'])
            self.assertEqual(parse((root / 'tracks.sam.txt').read_text())[0].obj_id, 1)

    def test_failed_combination_restores_previous_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "parts.json").write_text(json.dumps({"0": {"name": "keyboard"}, "1": {"name": "touchpad"}}))
            previous = '{"existing": {"parts": ["keyboard", "touchpad"]}}'
            (root / "track_unions.json").write_text(previous)
            agent = object.__new__(Agent)
            agent.tracked, agent.track_dir, agent.size = True, root, (10, 10)
            with patch("video2urdf.agent.load_parts", side_effect=RuntimeError("read failed")):
                with self.assertRaisesRegex(RuntimeError, "read failed"):
                    agent.tool_combine_tracks("base", ["keyboard", "touchpad"], "keyboard")
            self.assertEqual((root / "track_unions.json").read_text(), previous)

    def test_combines_distinct_regions_only_when_gate_is_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "parts.json").write_text(json.dumps({"0": {"name": "keyboard"}, "1": {"name": "touchpad"}}))
            (root / "track_unions.json").write_text(json.dumps({"base": {"parts": ["keyboard", "touchpad"],
                                                                          "gate": "keyboard", "frame_size": [10, 10]}}))
            keyboard = Detection(0, 0, (1, 1, 3, 3), np.ones((3, 3), bool))
            touchpad = Detection(0, 1, (5, 5, 7, 7), np.ones((3, 3), bool))
            only_touchpad = Detection(1, 1, (5, 5, 7, 7), np.ones((3, 3), bool))
            with patch("video2urdf.pipeline.load_tracks", return_value={0: {0: keyboard}, 1: {0: touchpad, 1: only_touchpad}}):
                names, tracks = load_parts(root)
            self.assertIn("base", names)
            self.assertEqual(list(tracks["base"]), [0])
            mask = tracks["base"][0].full_mask(10, 10)
            self.assertEqual(mask.sum(), 18)
            self.assertFalse(mask[4, 4])

