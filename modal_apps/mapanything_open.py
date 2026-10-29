"""MapAnything (facebook/map-anything-apache, Meta, Apache-2.0) on a Modal GPU: metric depth,
intrinsics and camera poses for a chunk of video frames, from RGB alone.

Client: video2urdf/pseudo_depth.py, which splits a video into chunks that share a few anchor
frames, sends each chunk here, and stitches the chunks into one Stray-Scanner-style capture.

Setup (once):
    modal secret create huggingface HF_TOKEN=...   (already there for sam3_open.py)
    modal deploy modal_apps/mapanything_open.py
The 4.9 GB checkpoint and the DINOv2 hub code download into the `mapanything-weights`
volume on the first call.

GPU: ~1.2B parameters under bf16 autocast; a chunk of ~48 frames at 504x378 fits a 24 GB A10G
with memory-efficient inference.
"""

import io
from pathlib import Path

import modal

APP = "video2urdf-mapanything"
MAPANYTHING_COMMIT = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"  # facebookresearch/map-anything, 2026-08-07
MODEL_ID = "facebook/map-anything-apache"
WEIGHTS = Path("/weights")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision==0.25.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(f"mapanything @ git+https://github.com/facebookresearch/map-anything.git@{MAPANYTHING_COMMIT}",
                 "numpy<2.4")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "HF_HOME": str(WEIGHTS / "hf"), "TORCH_HOME": str(WEIGHTS / "torch")})
)
volume = modal.Volume.from_name("mapanything-weights", create_if_missing=True)
app = modal.App(APP, image=image)


@app.cls(gpu=["A10G", "L4"], volumes={WEIGHTS: volume}, secrets=[modal.Secret.from_name("huggingface")],
         scaledown_window=120, timeout=1800, max_containers=1)
class MapAnything:
    @modal.enter()
    def load(self):
        import torch
        from mapanything.models import MapAnything as Model

        self.model = Model.from_pretrained(MODEL_ID).to("cuda").eval()
        volume.commit()
        self.torch = torch

    @modal.method()
    def infer(self, jpegs: list[bytes], focal: float | None = None) -> bytes:
        """One chunk of frames (all the same size, sides multiples of 14) -> npz bytes with
        depth (V, H, W) metres along z, conf (V, H, W), mask (V, H, W), intrinsics (V, 3, 3)
        at the input size, and poses (V, 4, 4) camera-to-world, OpenCV axes, world = view 0.
        ``focal`` (px at the input size, principal point at the centre) conditions the model on
        a known camera; without it the model estimates the intrinsics too."""
        import cv2
        import numpy as np
        from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

        torch = self.torch
        norm = IMAGE_NORMALIZATION_DICT["dinov2"]
        views = []
        for i, b in enumerate(jpegs):
            rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
            x = (x - norm.mean.view(3, 1, 1)) / norm.std.view(3, 1, 1)
            view = {"img": x[None], "true_shape": np.int32([rgb.shape[:2]]), "idx": i, "instance": str(i),
                    "data_norm_type": ["dinov2"]}
            if focal:
                h, w = rgb.shape[:2]
                view["intrinsics"] = torch.tensor([[[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]]], dtype=torch.float32)
            views.append(view)
        with torch.no_grad():
            preds = self.model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                                     apply_mask=True, mask_edges=True)

        def stack(key):
            return np.concatenate([p[key].float().cpu().numpy() for p in preds])

        out = io.BytesIO()
        np.savez_compressed(out, depth=stack("depth_z")[..., 0].astype(np.float16),
                            conf=stack("conf").astype(np.float16), mask=stack("mask")[..., 0] > 0.5,
                            intrinsics=stack("intrinsics"), poses=stack("camera_poses"),
                            metric_scale=stack("metric_scaling_factor"))
        return out.getvalue()
