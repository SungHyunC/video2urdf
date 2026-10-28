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


class OpenSamClient:
    def __init__(self, app: str = APP, cls: str = CLS):
        self.remote = modal.Cls.from_name(app, cls)()

    def segment_image(self, image_bgr: np.ndarray, prompt: str) -> str:
        """One concept prompt on one image (frame 0). Returns raw output text."""
        return self.remote.segment_image.remote(cv2.imencode(".png", image_bgr)[1].tobytes(), prompt)

    def track_boxes(self, video: Path, size: tuple[int, int], boxes: dict[int, tuple | list]) -> str:
        """Track up to 16 objects through the whole video; ``boxes`` as in SamClient.track_boxes.
        Local weights propagate both ways from the anchors, so no reversed clip is needed."""
        anchors = {oid: [fb] if isinstance(fb, tuple) and isinstance(fb[0], int) else [tuple(x) for x in fb]
                   for oid, fb in boxes.items()}
        return self.remote.track_boxes.remote(Path(video).read_bytes(), tuple(size), anchors)
