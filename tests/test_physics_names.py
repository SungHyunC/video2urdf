import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from video2urdf.agent import Agent


class PhysicsNamesTest(unittest.TestCase):
    def test_static_track_name_maps_to_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model.xml').write_text('<mujoco/>')
            (root / 'static.json').write_text(json.dumps({'part': 'box'}))
            a = object.__new__(Agent)
            a.track_dir, a.last_fit, a.fps = root, {'valid': True}, 30
            parts = [{'link': 'box', 'material': 'paper', 'mass_kg': .2, 'friction': .5}]
            with patch('video2urdf.agent.physics.apply', return_value={'links': {}}) as apply:
                self.assertIn('box to model link base', a.tool_set_physics(parts, []))
            self.assertEqual(set(apply.call_args.args[1]), {'base'})

    def test_drawer_track_name_maps_to_generated_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model.xml').write_text('<mujoco/>')
            (root / 'slides.json').write_text(json.dumps({'drawers': [{'part': 'drawer_moving'}]}))
            a = object.__new__(Agent)
            a.track_dir, a.last_fit, a.fps = root, {'valid': True}, 30
            parts = [{'link': 'base', 'material': 'wood', 'mass_kg': 18, 'friction': .5},
                     {'link': 'drawer_moving', 'material': 'wood', 'mass_kg': 2.5, 'friction': .4}]
            with patch('video2urdf.agent.physics.apply', return_value={'links': {}}) as apply:
                message = a.tool_set_physics(parts, [])
            self.assertEqual(set(apply.call_args.args[1]), {'base', 'drawer1'})
            self.assertIn('drawer_moving to model link drawer1', message)
            parts.append(dict(parts[1], link='drawer1'))
            self.assertIn('Multiple physics inputs', a.tool_set_physics(parts, []))

    def test_no_released_spans_in_any_spelling(self):
        # Spark sent "released": "{}" for a jointless AirPods case; the tool raised KeyError 'joint'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model.xml').write_text('<mujoco/>')
            a = object.__new__(Agent)
            a.track_dir, a.last_fit, a.fps = root, {'valid': True}, 30
            parts = [{'link': 'base', 'material': 'plastic', 'mass_kg': .08, 'friction': .4}]
            for released in ("{}", "[]", {}, [{}], [{"joint": "lid", "start_s": 1.0}], None):
                with patch('video2urdf.agent.physics.apply', return_value={'links': {}}) as apply:
                    a.tool_set_physics(parts, released)
                self.assertEqual(apply.call_args.args[2], {}, released)
            with patch('video2urdf.agent.physics.apply', return_value={'links': {}}) as apply:
                a.tool_set_physics(parts, [{"joint": "lid", "start_s": 1.0, "end_s": 2.0}])
            self.assertEqual(apply.call_args.args[2], {"lid_joint": [(30, 60)]})

if __name__ == '__main__': unittest.main()
