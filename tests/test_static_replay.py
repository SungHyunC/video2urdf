"""Small RGB-D integration test for a stationary cylinder seen by a moving camera."""
import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from video2urdf.lidar import open_capture
from video2urdf.sam import encode_rle
from video2urdf.static_shape import build_static_model


class StaticReplayTest(unittest.TestCase):
    def test_moving_camera_keeps_upright_cylinder_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); capture = root / 'capture'; out = root / 'model'
            capture.mkdir(); out.mkdir()
            w, h, f, n = 320, 240, 280., 12
            cx, cy = w / 2, h / 2
            yy, xx = np.mgrid[:h, :w]
            dx, dy = (xx + .5 - cx) / f, (yy + .5 - cy) / f
            positions = np.linspace(-.05, .05, n)
            world = np.tile(np.eye(4), (n, 1, 1)); world[:, 0, 3] = -positions
            depths, records = [], []
            video = capture / 'cylinder.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
            self.assertTrue(writer.isOpened())
            for frame, pos in enumerate(positions):
                a = dx * dx + 1; b = 2 * (pos * dx - .7); c = pos * pos + .7 ** 2 - .06 ** 2
                discriminant = b * b - 4 * a * c
                near = (-b - np.sqrt(np.maximum(discriminant, 0))) / (2 * a)
                obj = (discriminant >= 0) & (np.abs(dy * near) < .1)
                floor = (dy > .08) & ~obj
                depth = np.zeros((h, w), np.float32); depth[obj] = near[obj]; depth[floor] = .12 / dy[floor]
                depths.append(depth)
                image = np.full((h, w, 3), 30, np.uint8); image[floor] = 130; image[obj] = [40, 90, 210]
                writer.write(image)
                def rec(i, mask):
                    return f'{i}<|box;x1=0;y1=0;x2={w-1};y2={h-1}|><|mask;data={h},{w},{encode_rle(mask)}|>'
                records.append(f'<{frame}f>{rec(0, obj)},{rec(1, floor)}\n')
            writer.release()
            np.savez_compressed(capture / 'depth.npz', depth=np.asarray(depths))
            (capture / 'intrinsics.json').write_text(json.dumps({'f': f, 'cx': cx, 'cy': cy}))
            (out / 'tracks.sam.txt').write_text(''.join(records))
            (out / 'parts.json').write_text(json.dumps({'0': {'name': 'object'}, '1': {'name': 'floor'}}))
            cap = open_capture(video, capture, out); cap.world = world; cap.moving_camera = True
            result = build_static_model(cap, 'object', 'floor', shape='cylinder', log=lambda _: None)
            self.assertTrue(result['cylinder_curve_fit'])
            self.assertAlmostEqual(result['generated_size_mm'][0], 120, delta=5)
            self.assertGreater(result['mask_coverage']['base'], .85)
            self.assertLess(result['observed_centroid_shift_p90_mm'], 50)


if __name__ == '__main__': unittest.main()
