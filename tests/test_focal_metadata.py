"""The camera's focal length read from a phone video's QuickTime metadata."""

import struct
import tempfile
import unittest
from pathlib import Path

from video2urdf.ingest import FOCAL_KEY, camera_focal_35mm


def atom(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def quicktime(focal) -> bytes:
    """ftyp + mdat + a moov whose video trak carries QuickTime keys/ilst metadata."""
    keys = [b"com.apple.quicktime.camera.lens_model", FOCAL_KEY.encode()]
    keys_atom = atom(b"keys", struct.pack(">I", 0) + struct.pack(">I", len(keys)) +
                     b"".join(struct.pack(">I", 8 + len(k)) + b"mdta" + k for k in keys))
    items = [atom(struct.pack(">I", 1), atom(b"data", struct.pack(">II", 1, 0) + b"back camera 6.86mm f/1.78"))]
    if focal is not None:
        items.append(atom(struct.pack(">I", 2), atom(b"data", struct.pack(">II", 21, 0) + struct.pack(">h", focal))))
    meta = atom(b"meta", atom(b"hdlr", b"\0" * 8 + b"mdta" + b"\0" * 13) + keys_atom + atom(b"ilst", b"".join(items)))
    trak = atom(b"trak", atom(b"tkhd", b"\0" * 84) + meta)
    return atom(b"ftyp", b"qt  \0\0\0\0qt  ") + atom(b"mdat", b"\0" * 64) + atom(b"moov", atom(b"mvhd", b"\0" * 100) + trak)


class FocalMetadataTest(unittest.TestCase):
    def test_reads_the_35mm_equivalent_from_the_video_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mov"
            path.write_bytes(quicktime(26))
            self.assertEqual(camera_focal_35mm(path), 26.0)
            path.write_bytes(quicktime(None))
            self.assertIsNone(camera_focal_35mm(path))
            path.write_bytes(b"not a video at all")
            self.assertIsNone(camera_focal_35mm(path))


if __name__ == "__main__":
    unittest.main()
