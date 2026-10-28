"""SAM 3.1 open weights (facebook/sam3.1) on a Modal GPU, answering in the hosted API's wire format.

The open-weights twin of video2urdf.sam.SamClient (client: video2urdf/sam_open.py). Masks come
back with the `R` run-length payload that video2urdf.sam decodes.

Setup (once): accept the SAM License on huggingface.co/facebook/sam3.1, then
    modal secret create huggingface HF_TOKEN=hf_...
    modal deploy modal_apps/sam3_open.py
The 3.5 GB checkpoint downloads into the `sam3-weights` volume on the first call.

GPU: the checkpoint is 3.5 GB of fp32 (~0.9B parameters) and runs under bf16 autocast; video
frames are kept on the CPU at 1008 px, so a 24 GB A10G (or L4) has ample room for <= 16 objects.
FlashAttention 3 is Hopper-only and off.
"""

import os
import tempfile
from pathlib import Path

import modal

APP = "video2urdf-sam3-open"
SAM3_COMMIT = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"  # facebookresearch/sam3, 2026-10-23
WEIGHTS = Path("/weights")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision==0.25.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        f"sam3 @ git+https://github.com/facebookresearch/sam3.git@{SAM3_COMMIT}",
        "opencv-python-headless==4.10.0.84",
        "einops==0.8.1",
        "scipy==1.15.3",
        "scikit-image==0.25.2",
        "psutil==7.0.0",
        "pycocotools==2.0.10",  # imported by sam3/__init__ via sam3.train
        "setuptools==80.9.0",  # sam3 imports pkg_resources
        "huggingface_hub==0.36.0",
    )
    .env({"HF_HUB_DISABLE_TELEMETRY": "1"})
)
volume = modal.Volume.from_name("sam3-weights", create_if_missing=True)
app = modal.App(APP, image=image)


def encode_rle(mask) -> str:
    """Same as video2urdf.sam.encode_rle (the container doesn't ship the package)."""
    import numpy as np

    flat = np.asarray(mask, bool).ravel()
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.concatenate([[0], edges, [flat.size]]))
    if flat.size and flat[0]:
        runs = np.concatenate([[0], runs])
    return "R" + ".".join(map(str, runs.tolist()))


def record(obj_id: int, mask, conf: float, width: int, height: int) -> str | None:
    """One `{id}<|box;..|><|mask;..|>` record: the mask cropped to its tight box (x2, y2 inclusive)."""
    import numpy as np

    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if not len(rows):
        return None
    y1, y2, x1, x2 = rows[0], rows[-1], cols[0], cols[-1]
    crop = mask[y1:y2 + 1, x1:x2 + 1]
    return (f"{obj_id}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={width};h={height};c={conf}|>"
            f"<|mask;x=0;y=0;data={crop.shape[0]},{crop.shape[1]},{encode_rle(crop)}|>")


def frame_records(outputs: dict, width: int, height: int) -> dict[int, str]:
    out = {}
    for oid, prob, mask in zip(outputs["out_obj_ids"], outputs["out_probs"], outputs["out_binary_masks"]):
        r = record(int(oid), mask, float(prob), width, height)
        if r:
            out[int(oid)] = r
    return out


def to_text(records: dict[int, dict[int, str]]) -> str:
    return "\n".join(f"<{f}f>" + ",".join(records[f][o] for o in sorted(records[f]))
                     for f in sorted(records) if records[f]) + "\n"


def load_predictor(checkpoint: str):
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    return build_sam3_multiplex_video_predictor(checkpoint_path=checkpoint, use_fa3=False, async_loading_frames=False)


def segment_image(predictor, png: bytes, prompt: str) -> str:
    """Concept prompt on one image, run as a one-frame video session with a text prompt."""
    import torch

    with tempfile.TemporaryDirectory() as tmp, torch.autocast("cuda", dtype=torch.bfloat16):
        path = Path(tmp) / "image.png"
        path.write_bytes(png)
        sid = predictor.handle_request({"type": "start_session", "resource_path": str(path)})["session_id"]
        try:
            state = predictor._all_inference_states[sid]["state"]
            w, h = state["orig_width"], state["orig_height"]
            out = predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                                            "text": prompt})["outputs"]
            return to_text({0: frame_records(out, w, h)})
        finally:
            predictor.handle_request({"type": "close_session", "session_id": sid})


def track_boxes(predictor, mp4: bytes, size: tuple[int, int], anchors: dict[int, list]) -> str:
    """Track box-anchored objects through every frame. A box is the tracker's two corner points
    (labels 2 and 3). Forward from the earliest anchor, then backward from the latest first
    anchor; frames before an object's first anchor take the backward result."""
    import torch

    anchors = {int(o): sorted((int(f), tuple(b)) for f, b in fb) for o, fb in anchors.items()}
    first = {o: fb[0][0] for o, fb in anchors.items()}
    with tempfile.TemporaryDirectory() as tmp, torch.autocast("cuda", dtype=torch.bfloat16):
        path = Path(tmp) / "video.mp4"
        path.write_bytes(mp4)
        sid = predictor.handle_request({"type": "start_session", "resource_path": str(path),
                                        "offload_video_to_cpu": True})["session_id"]
        try:
            state = predictor._all_inference_states[sid]["state"]
            w, h = state["orig_width"], state["orig_height"]
            if (w, h) != tuple(size):
                raise ValueError(f"decoded frames are {w}x{h}, caller expects {size[0]}x{size[1]}")
            for oid, fb in anchors.items():
                for frame, (x1, y1, x2, y2) in fb:
                    predictor.handle_request({
                        "type": "add_prompt", "session_id": sid, "frame_index": frame, "obj_id": oid,
                        "points": [[x1 / w, y1 / h], [x2 / w, y2 / h]], "point_labels": [2, 3]})
            fwd: dict[int, dict[int, str]] = {}
            for r in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                                                      "propagation_direction": "forward",
                                                      "start_frame_index": min(first.values())}):
                fwd[r["frame_index"]] = frame_records(r["outputs"], w, h)
            bwd: dict[int, dict[int, str]] = {}
            last = max(first.values())
            if last > 0:
                # Propagation replays the action history: a forward pass that started on frame 0
                # makes the next call a cache fetch, so mark the objects as refined to force a
                # real backward pass.
                late = [o for o, f in first.items() if f > 0]
                predictor.model.add_action_history(state, "refine", frame_idx=last, obj_ids=late)
                for r in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                                                          "propagation_direction": "backward",
                                                          "start_frame_index": last}):
                    bwd[r["frame_index"]] = frame_records(r["outputs"], w, h)
        finally:
            predictor.handle_request({"type": "close_session", "session_id": sid})
    records: dict[int, dict[int, str]] = {}
    for f in sorted(set(fwd) | set(bwd)):
        for oid in anchors:
            r = bwd.get(f, {}).get(oid) if f < first[oid] else None
            r = r or fwd.get(f, {}).get(oid) or bwd.get(f, {}).get(oid)
            if r:
                records.setdefault(f, {})[oid] = r
    return to_text(records)


@app.cls(gpu=["A10G", "L4"], volumes={WEIGHTS: volume}, secrets=[modal.Secret.from_name("huggingface")],
         scaledown_window=120, timeout=1800)
class Sam3:
    @modal.enter()
    def load(self):
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download("facebook/sam3.1", "sam3.1_multiplex.pt", cache_dir=WEIGHTS / "hf",
                               token=os.environ["HF_TOKEN"])
        volume.commit()
        self.predictor = load_predictor(ckpt)

    @modal.method()
    def segment_image(self, png: bytes, prompt: str) -> str:
        return segment_image(self.predictor, png, prompt)

    @modal.method()
    def track_boxes(self, mp4: bytes, size: tuple[int, int], anchors: dict[int, list]) -> str:
        return track_boxes(self.predictor, mp4, size, anchors)
