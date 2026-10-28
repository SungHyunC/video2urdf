"""Check the open-weights SAM 3.1 backend.

1. Round trip of the `R` mask payload (video2urdf.sam and the Modal app's copy of the encoder),
   through the full wire format and sam.parse. Runs locally, no GPU.
2. With --e2e (needs `modal deploy modal_apps/sam3_open.py` and the `huggingface` secret):
   track the synthetic lamp with the same anchors as the hosted run and compare masks with the
   hosted data/tracks/lamp/tracks.sam.txt by IoU per object; also one concept prompt per part.

Usage: .venv/bin/python scripts/check_sam_open.py [--e2e] [--video data/synth/lamp.mp4]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modal_apps import sam3_open  # noqa: E402
from video2urdf.sam import decode_rle, encode_rle, load_tracks, parse, tracks  # noqa: E402


def check_rle():
    rng = np.random.default_rng(0)
    cases = [np.zeros((1, 1), bool), np.ones((1, 1), bool), np.zeros((3, 5), bool), np.ones((4, 2), bool),
             np.array([[0, 1, 1], [1, 0, 0]], bool)]
    cases += [rng.random((rng.integers(1, 80), rng.integers(1, 80))) < p for p in (0.02, 0.5, 0.98) for _ in range(50)]
    yy, xx = np.mgrid[:600, :900]
    cases.append((xx - 450) ** 2 / 300 ** 2 + (yy - 300) ** 2 / 200 ** 2 < 1)  # big blob, like a real mask
    assert encode_rle(cases[4]) == "R1.3.2", encode_rle(cases[4])
    for m in cases:
        for enc in (encode_rle, sam3_open.encode_rle):
            p = enc(m)
            assert p[0] == "R" and "|" not in p and "," not in p
            assert np.array_equal(decode_rle(p, m.shape[1], m.shape[0]), m)
    try:
        decode_rle("R1.2", 2, 2)
        raise AssertionError("short RLE accepted")
    except ValueError:
        pass

    # Whole wire format: records from the Modal app, parsed by the pipeline.
    full = np.zeros((480, 640), bool)
    full[100:200, 300:350] = True
    full[150, 300:420] = True
    other = np.zeros_like(full)
    other[0:5, 630:640] = True
    out = {"out_obj_ids": np.array([3, 7]), "out_probs": np.array([0.9, 0.5]),
           "out_binary_masks": np.stack([full, other])}
    text = sam3_open.to_text({0: sam3_open.frame_records(out, 640, 480), 5: sam3_open.frame_records(out, 640, 480)})
    objs = tracks(parse(text))
    assert sorted(objs) == [3, 7] and sorted(objs[3]) == [0, 5]
    d = objs[3][5]
    assert d.box == (300, 100, 419, 199) and d.confidence == 0.9
    assert np.array_equal(d.full_mask(480, 640), full)
    assert np.array_equal(objs[7][0].full_mask(480, 640), other)
    big = cases[-1]
    print(f"RLE ok: {len(cases)} rasters x 2 encoders, wire format through sam.parse; "
          f"600x900 blob -> {len(encode_rle(big))} chars")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def check_e2e(video: Path, track_dir: Path):
    from video2urdf.sam_open import OpenSamClient

    parts = {int(k): v for k, v in json.loads((track_dir / "parts.json").read_text()).items()}
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, parts[min(parts)]["frame"])
    key = cap.read()[1]
    cap.release()
    client = OpenSamClient()

    for oid, p in parts.items():
        t0 = time.time()
        dets = parse(client.segment_image(key, p["prompt"]))
        best = max(dets, key=lambda d: d.confidence or 0, default=None)
        print(f"segment_image {p['prompt']!r}: {len(dets)} found"
              + (f", best box {best.box} c={best.confidence:.2f} (hosted {p['box']} c={p['confidence']:.2f})"
                 if best else "") + f", {time.time() - t0:.1f} s")

    anchors = {oid: [(f, tuple(b)) for f, b in p.get("anchors", [[p["frame"], p["box"]]])] for oid, p in parts.items()}
    t0 = time.time()
    text = client.track_boxes(video, (w, h), anchors)
    print(f"track_boxes: {time.time() - t0:.1f} s for {n} frames")
    out = track_dir / "tracks.open.txt"
    out.write_text(text)
    ours, hosted = tracks(parse(text)), load_tracks(track_dir / "tracks.sam.txt")
    for oid, p in parts.items():
        frames = sorted(set(hosted.get(oid, {})) | set(ours.get(oid, {})))
        ious = [iou(ours[oid][f].full_mask(h, w) if f in ours.get(oid, {}) else np.zeros((h, w), bool),
                    hosted[oid][f].full_mask(h, w) if f in hosted.get(oid, {}) else np.zeros((h, w), bool))
                for f in frames]
        print(f"  {p['prompt']!r} (id {oid}): {len(ours.get(oid, {}))}/{n} frames, IoU vs hosted "
              f"mean {np.mean(ious):.3f}, min {np.min(ious):.3f}")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--video", type=Path, default=ROOT / "data" / "synth" / "lamp.mp4")
    args = ap.parse_args()
    check_rle()
    if args.e2e:
        check_e2e(args.video, ROOT / "data" / "tracks" / args.video.stem)


if __name__ == "__main__":
    main()
