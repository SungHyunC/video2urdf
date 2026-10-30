"""SAM 3.1 open weights on Modal: a drop-in for sam.SamClient (same methods, same raw text).

The model runs in modal_apps/sam3_open.py (deploy it first). Masks come back with the `R`
run-length payload, which sam.parse reads like the hosted encodings.
"""

from pathlib import Path

import cv2
import modal
import numpy as np

APP = "video2urdf-sam3-open"
CLS = "Sam3"


def merge_passes(fwd: dict, bwd: dict, anchors: dict[int, list]) -> dict:
    """As modal_apps/sam3_open.merge_passes: frames before an object's first anchor from the
    backward pass, the rest forward first; prompted frames missing an object take its neighbour's."""
    anchors = {int(o): sorted((int(f), tuple(b)) for f, b in fb) for o, fb in anchors.items()}
    first = {o: fb[0][0] for o, fb in anchors.items()}
    records: dict[int, dict[int, str]] = {}
    for f in sorted(set(fwd) | set(bwd)):
        for oid in anchors:
            r = bwd.get(f, {}).get(oid) if f < first[oid] else None
            r = r or fwd.get(f, {}).get(oid) or bwd.get(f, {}).get(oid)
            if r:
                records.setdefault(f, {})[oid] = r
    prompted = {f for fb in anchors.values() for f, _ in fb}
    for oid in anchors:
        for f in sorted(prompted):
            if oid in records.get(f, {}):
                continue
            for g in (f + 1, f - 1):
                if oid in records.get(g, {}):
                    records.setdefault(f, {})[oid] = records[g][oid]
                    break
    return records


class OpenSamClient:
    def __init__(self, app: str = APP, cls: str = CLS):
        self.remote = modal.Cls.from_name(app, cls)()

    def segment_image(self, image_bgr: np.ndarray, prompt: str) -> str:
        """One concept prompt on one image (frame 0). Returns raw output text."""
        return self.remote.segment_image.remote(cv2.imencode(".png", image_bgr)[1].tobytes(), prompt)

    def track_boxes(self, video: Path, size: tuple[int, int], boxes: dict[int, tuple | list],
                    stride: int | None = None) -> str:
        """Track up to 16 objects through the whole video; ``boxes`` as in SamClient.track_boxes.
        Local weights propagate both ways from the anchors, so no reversed clip is needed; the
        forward and backward passes run on two GPU containers at once."""
        from .sam import _records_of, _text_of, strided_track, track_stride
        anchors = {oid: [fb] if isinstance(fb, tuple) and isinstance(fb[0], int) else [tuple(x) for x in fb]
                   for oid, fb in boxes.items()}
        if stride is None:
            stride = track_stride(video)
        if stride > 1:
            return strided_track(lambda v, sz, a: self.track_boxes(v, sz, a, stride=1), video, size, anchors, stride)
        mp4 = Path(video).read_bytes()
        calls = [self.remote.track_pass.spawn(mp4, tuple(size), anchors, reverse) for reverse in (False, True)]
        fwd, bwd = (_records_of(c.get()) for c in calls)
        return _text_of(merge_passes(fwd, bwd, anchors))
