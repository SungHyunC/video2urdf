"""Spark alone: the baseline for "why not just ask an AI for the model?".

The same Muse Spark that drives Video2URDF gets the same video (what the agent sees first: 480 px,
10 fps, plus three frames at 1024 px) and one plain request, the kind a user would type. No tools:
no SAM, no depth, no fitting, no replay. Its URDF is then scored with the measuring tools:

- hinged objects: URDF Doctor registers the URDF onto the measured model by geometry (base and
  moving part at seven openings, the best joint value chosen for each), then reports its axis
  direction and position, its limits against the motion in the video and its shape; on RBO
  sequences the registered hinge is also scored against motion capture, exactly as the agent's
  fit is (rbo.hinge_errors);
- the drawer: the prismatic joint's travel and the drawer front against the measured ones.

Several hinges or slides in one URDF: the one that fits the measured motion best is scored,
which favours the baseline.
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import doctor, rbo
from .agent import BASE_URL, MODEL, jpeg_url, mp4_url
from .pipeline import read_frame, video_info

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "spark_alone"
FRAME_WIDTH = 1024
FRAMES_AT = (0.25, 0.5, 0.75)  # fractions of the video, the stills sent with it
MAX_TOKENS = 16000  # Spark reasons before it answers

PROMPT = """I filmed this object being moved. Write a URDF of it that I can load into MuJoCo to simulate it.

- Links: simple box or cylinder geometry, sized in metres to match the real object.
- Joints: the right type, axis, origin and limits, so the model moves the way the object moves in the video.

Make it as close to the real object as you can tell from the video. Reply with the complete URDF in one ```xml code block."""


@dataclass
class Case:
    video: str  # paths relative to the repository
    measured: str  # the Video2URDF run it is scored against
    kind: str  # "hinge" or "slide"
    rbo: str | None = None  # RBO sequence: also scored against motion capture
    truth: dict = field(default_factory=dict)
    label: str = ""


# Not here: the microwave. Its measured body is only the control-panel strip beside the door
# (the cabinet behind the door wasn't measured), so a whole-microwave URDF can't be placed on it:
# its five answers came out 55-89 deg off, an artefact of the placement, not Spark's door.
# RBO sequences whose measured model can place another model: the baseline's URDF is matched to
# the measured base and lid, so they must have the object's shape. Left out: laptop04 and book03
# (the replayed base covers 27% / 5% of its mask) and book05 (its lid mesh is 525 mm long: the
# outline voted over the video is wrong although the hinge is right). Their hinges are scored in
# README's RBO table.
RBO = ("laptop01", "laptop02", "laptop05")
CASES = {
    "laptop_lidar": Case("data/real/laptop_lidar/laptop_lidar.mp4", "data/tracks_agent/laptop_lidar", "hinge",
                         truth={"moving_mm": (304, 212), "source": "13-inch MacBook Pro display, spec"},
                         label="13\" MacBook Pro, iPhone LiDAR"),
    "drawer2_lidar": Case("data/real/drawer2_lidar/drawer2_lidar.mp4", "data/tracks_agent/drawer2_lidar", "slide",
                          label="office drawer, iPhone LiDAR"),
    **{s: Case(f"data/rbo/prep/{s}/{s}.mp4", f"data/spark_alone/measured_rbo/{s}", "hinge", rbo=s,
               label=f"RBO {s} (motion capture)") for s in RBO},
}


def extract_urdf(text: str) -> str | None:
    """The URDF in Spark's answer: the code block holding <robot>, else the bare <robot> element."""
    for block in re.findall(r"```[a-zA-Z]*\s*\n(.*?)```", text, re.S):
        if "<robot" in block:
            return block.strip() + "\n"
    m = re.search(r"<robot\b.*?</robot>", text, re.S)
    return m.group(0) + "\n" if m else None


def _chat(client, **kw):
    from openai import APIConnectionError, APIStatusError
    for attempt in range(6):
        try:
            return client.chat.completions.create(model=MODEL, max_tokens=MAX_TOKENS, **kw)
        except (APIConnectionError, APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if status is not None and status < 500 and status != 429 or attempt == 5:
                raise
            time.sleep(min(120, 15 * 2 ** attempt))


def ask(case: str, run_dir: Path, client) -> dict:
    """One plain request to Spark for the case's URDF; writes response.json and spark.urdf."""
    c = CASES[case]
    video = ROOT / c.video
    _, n, _ = video_info(video)
    frames = [int(n * f) for f in FRAMES_AT]
    content = ([{"type": "video_url", "video_url": {"url": mp4_url(video)}}]
               + [{"type": "image_url", "image_url": {"url": jpeg_url(read_frame(video, t), width=FRAME_WIDTH)}}
                  for t in frames]
               + [{"type": "text", "text": PROMPT}])
    t0 = time.time()
    resp = _chat(client, messages=[{"role": "user", "content": content}])
    text = resp.choices[0].message.content or ""
    run_dir.mkdir(parents=True, exist_ok=True)
    urdf = extract_urdf(text)
    record = {"case": case, "model": MODEL, "base_url": BASE_URL, "video": c.video, "frames": frames,
              "prompt": PROMPT, "answer": text, "seconds": round(time.time() - t0, 1),
              "usage": resp.usage.model_dump() if getattr(resp, "usage", None) else None,
              "urdf_found": urdf is not None}
    (run_dir / "response.json").write_text(json.dumps(record, indent=2))
    if urdf:
        (run_dir / "spark.urdf").write_text(urdf)
    return record


# --- scoring ----------------------------------------------------------------------------------

def _pose0(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_kinematics(model, data)
    return data


def extents_mm(model: mujoco.MjModel, bodies: list[int], along: np.ndarray | None = None) -> list[float]:
    """Sizes (mm) of the bodies' geometry along its principal axes, largest first; with ``along``
    (a unit vector) the size along it comes first and the two across it follow."""
    pts = doctor.body_points(model, _pose0(model), bodies)
    pts = pts - pts.mean(axis=0)
    if along is None:
        return sorted((1000 * np.ptp(pts @ np.linalg.svd(pts, full_matrices=False)[2].T, axis=0)).tolist(),
                      reverse=True)
    along = along / np.linalg.norm(along)
    across = pts - np.outer(pts @ along, along)
    rest = np.ptp(across @ np.linalg.svd(across, full_matrices=False)[2][:2].T, axis=0)
    return [float(1000 * np.ptp(pts @ along))] + sorted((1000 * rest).tolist(), reverse=True)


def _load(urdf: Path) -> tuple[mujoco.MjModel | None, str | None]:
    try:
        return doctor.load(urdf), None
    except Exception as e:  # what a user would hit loading it into MuJoCo
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"


def score_hinge(urdf: Path, case: str) -> dict:
    c = CASES[case]
    measured = ROOT / c.measured
    model, error = _load(urdf)
    if model is None:
        return {"usable": False, "why": f"doesn't load in MuJoCo ({error})"}
    names = doctor.hinge_names(model)
    if not names:
        return {"usable": False, "why": "no revolute joint"}
    tried = []
    for name in names:
        try:
            tried.append((name, doctor.diagnose(urdf, measured, joint=name)))
        except Exception as e:
            tried.append((name, e))
    fits = [(n, d) for n, d in tried if isinstance(d, doctor.Diagnosis)]
    if not fits:
        return {"usable": False, "why": f"doctor failed: {tried[0][1]}"}
    name, d = min(fits, key=lambda f: f[1].numbers["shape_mismatch_mm"])
    num = d.numbers
    uj = doctor.hinge(model, name)
    moving, _ = doctor.moving_and_fixed(model, uj)
    axis = _pose0(model).xaxis[uj.jid].copy()
    lo_need, hi_need = num["q_needed_deg"]
    lim = num["urdf_limits_deg"]
    out = {"usable": True, "joint": name, "revolute_joints": len(names), "findings": d.findings,
           "axis_direction_error_deg": num["axis_direction_error_deg"], "axis_offset_mm": num["axis_offset_mm"],
           "shape_mismatch_mm": num["shape_mismatch_mm"], "q_per_opening": num["q_per_opening"],
           "opening_seen_deg": num["opening_seen_deg"], "q_needed_deg": num["q_needed_deg"], "urdf_limits_deg": lim,
           "limits_cover_motion": bool(lim and lim[0] <= lo_need + doctor.LIMIT_TOL_DEG
                                       and hi_need <= lim[1] + doctor.LIMIT_TOL_DEG),
           "moving_extents_mm": extents_mm(model, moving, along=axis),  # along the hinge, across, thick
           # where URDF Doctor placed it, and its joint value for each real opening (for overlay())
           "registration": num["registration"], "q_when_closed_deg": num["q_when_closed_deg"]}
    if c.rbo:
        out["mocap"] = mocap_scores(case, np.array(num["urdf_axis_in_measured"]),
                                    np.array(num["urdf_anchor_in_measured"]))
    return out


def _to_camera(measured: Path, axis: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A line in the measured model's world back into the reference camera's coordinates:
    the model's world is rotation @ (p_camera - hinge_point) + root_pos (lidar.build_model)."""
    ctm = json.loads((measured / "model3d.json").read_text())["camera_to_model"]
    r, p0, r0 = np.array(ctm["rotation"]), np.array(ctm["hinge_point"]), np.array(ctm["root_pos"])
    return r.T @ axis, r.T @ (point - r0) + p0


_TRUTH: dict = {}


def mocap_truth(case: str) -> dict:
    c = CASES[case]
    if case not in _TRUTH:
        h = json.loads((ROOT / c.measured / "hinge.json").read_text())
        _TRUTH[case] = rbo.mocap_hinge(ROOT / "data" / "rbo" / "x" / c.rbo, len(h["q"]), h["reference_frame"])
    return _TRUTH[case]


def mocap_scores(case: str, axis_model: np.ndarray, point_model: np.ndarray) -> dict:
    measured = ROOT / CASES[case].measured
    gt = mocap_truth(case)
    axis, point = _to_camera(measured, axis_model, point_model)
    err, gap = rbo.hinge_errors(axis, point, gt)
    return {"axis_direction_error_deg": err, "axis_position_error_mm": 1000 * gap,
            "true_range_deg": float(np.rad2deg(np.nanmax(gt["q"]) - np.nanmin(gt["q"])))}


def measured_scores(case: str) -> dict:
    """The Video2URDF model of the same case, for the table's other column."""
    c = CASES[case]
    measured = ROOT / c.measured
    model = mujoco.MjModel.from_xml_path(str(measured / "model.xml"))
    meta = json.loads((measured / "model3d.json").read_text())
    if c.kind == "slide":
        slides = json.loads((measured / "slides.json").read_text())
        j = slides["joints"][0]
        return {"travel_mm": j["travel_mm"], "box_mm": slides.get("box_sizes_mm", {})}
    mj = doctor.hinge(model)
    axis = _pose0(model).xaxis[mj.jid].copy()
    lo, hi = meta["hinge_range_deg"]  # the model's limits: the range measured in the video
    out = {"moving_extents_mm": extents_mm(model, doctor.subtree(model, mj.child), along=axis),
           "plate_size_mm": meta.get("plate_size_mm"), "mask_coverage": meta.get("mask_coverage"),
           "limit_span_deg": hi - lo}
    if c.rbo:
        h = lidar_hinge(measured)
        gt = mocap_truth(case)
        err, gap = rbo.hinge_errors(np.array(h["axis"]), np.array(h["point"]), gt)
        out["mocap"] = {"axis_direction_error_deg": err, "axis_position_error_mm": 1000 * gap,
                        "true_range_deg": float(np.rad2deg(np.nanmax(gt["q"]) - np.nanmin(gt["q"]))),
                        "measured_range_deg": h["opening_range_deg"]}
    return out


def lidar_hinge(run: Path) -> dict:
    from .lidar import read_hinge
    return read_hinge(run)


def score_slide(urdf: Path, case: str) -> dict:
    c = CASES[case]
    model, error = _load(urdf)
    if model is None:
        return {"usable": False, "why": f"doesn't load in MuJoCo ({error})"}
    slides = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE]
    if not slides:
        return {"usable": False, "why": "no prismatic joint"}
    measured = json.loads((ROOT / c.measured / "slides.json").read_text())["joints"][0]["travel_mm"]
    data = _pose0(model)
    rows = []
    for j in slides:
        travel = 1000 * float(model.jnt_range[j][1] - model.jnt_range[j][0]) if model.jnt_limited[j] else None
        body = int(model.jnt_bodyid[j])
        rows.append({"joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j), "travel_mm": travel,
                     "moving_extents_mm": extents_mm(model, doctor.subtree(model, body), along=data.xaxis[j].copy())})
    # The slide closest to the measured travel stands for the drawer that moved (favours the baseline).
    best = min(rows, key=lambda r: abs(r["travel_mm"] - measured) if r["travel_mm"] is not None else np.inf)
    return {"usable": True, "prismatic_joints": len(slides), **best, "all": rows}


def score(urdf: Path, case: str) -> dict:
    return (score_slide if CASES[case].kind == "slide" else score_hinge)(urdf, case)


# --- overlay: both models over the video from the recorded camera -----------------------------

def _with_camera(urdf: Path) -> mujoco.MjModel:
    """The URDF as MuJoCo compiles it, with a world camera named "video" and any free joint
    dropped (the body stays where the joint would start it; the camera moves instead)."""
    import tempfile
    model = doctor.load(urdf)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.xml"
        mujoco.mj_saveLastXML(str(path), model)
        text = path.read_text()
    text = re.sub(r"<freejoint[^>]*/>", "", text)
    text = re.sub(r"<joint[^>]*type=\"free\"[^>]*/>", "", text)
    text = text.replace("<worldbody>", '<worldbody>\n    <camera name="video" pos="0 0 0"/>', 1)
    return mujoco.MjModel.from_xml_string(text)


def _segment(model, data, renderer, cam_to_world: np.ndarray, flip: np.ndarray) -> np.ndarray:
    """Body id per pixel (-1: background) seen from cam_to_world (OpenCV axes)."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (cam_to_world[:3, :3] @ flip).flatten())
    cam = model.camera("video").id
    model.cam_pos[cam], model.cam_quat[cam] = cam_to_world[:3, 3], quat
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="video")
    seg = renderer.render()
    geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
    return np.where(geom >= 0, model.geom_bodyid[np.maximum(geom, 0)], -1)


def _iou(pred: np.ndarray, mask: np.ndarray) -> float:
    union = (pred | mask).sum()
    return float((pred & mask).sum() / union) if union else np.nan


def overlay(case: str, run: str, out: Path, stride: int = 2, out_width: int = 1920) -> dict:
    """Video2URDF's model and Spark's URDF (placed by URDF Doctor, joint set from the measured
    opening through Doctor's fit) rendered over the video from the phone's recorded camera, side
    by side; scores both against the SAM masks (IoU per part). Writes out (mp4) and out.json."""
    from .lidar import read_hinge
    from .pipeline import load_parts
    from .replay import COLORS
    c = CASES[case]
    measured = ROOT / c.measured
    run_dir = OUT / case / run
    score = json.loads((run_dir / "score.json").read_text())
    rec = np.load(measured / "replay3d.npz")
    cams, f = rec["cam_to_world"], float(rec["f"])
    q_ours = rec["q_lid_joint"]
    n = len(cams)

    # The real opening per frame (as URDF Doctor reads it) -> Spark's joint value.
    h = read_hinge(measured)
    q = np.array([np.nan if v is None else v for v in h["q"]], float)[:n]
    known = np.flatnonzero(~np.isnan(q))
    q = np.interp(np.arange(n), known, q[known])
    shut = np.deg2rad(h["opening_on_reference_deg"])
    toward = 1.0 if abs(q.max() - shut) < abs(q.min() + shut) else -1.0
    opening = shut - toward * q
    slope, q0 = score["q_per_opening"], np.deg2rad(score["q_when_closed_deg"])
    q_spark = slope * opening + q0
    if score["urdf_limits_deg"]:  # MuJoCo holds the joint inside the URDF's limits
        q_spark = np.clip(q_spark, *np.deg2rad(score["urdf_limits_deg"]))
    opening_spark = (q_spark - q0) / slope
    to_spark = np.linalg.inv(np.array(score["registration"]))  # measured world -> Spark's world

    roles = json.loads((measured / "roles.json").read_text())
    _, tracks = load_parts(measured)
    masks = {"lid": tracks[roles["plate"]], "base": tracks[roles["base"]]}

    video = ROOT / c.video
    cap = cv2_capture(video)
    w, hgt = int(cap.get(3)), int(cap.get(4))
    fovy = np.rad2deg(2 * np.arctan(hgt / 2 / f))
    mm = mujoco.MjModel.from_xml_path(str(measured / "model.xml"))
    sm = _with_camera(run_dir / "spark.urdf")
    for m in (mm, sm):
        m.vis.global_.offwidth, m.vis.global_.offheight = max(w, 640), max(hgt, 480)
        m.cam_fovy[m.camera("video").id] = fovy
    md, sd = mujoco.MjData(mm), mujoco.MjData(sm)
    rm, rs = mujoco.Renderer(mm, hgt, w), mujoco.Renderer(sm, hgt, w)
    for r in (rm, rs):
        r.enable_segmentation_rendering()
    mj = doctor.hinge(mm)
    sj = doctor.hinge(sm, score["joint"])
    parts = {"ours": {"lid": set(doctor.subtree(mm, mj.child))},
             "spark": {"lid": set(doctor.subtree(sm, sj.child))}}
    parts["ours"]["base"] = {b for b in range(1, mm.nbody) if b not in parts["ours"]["lid"]}
    parts["spark"]["base"] = {b for b in range(1, sm.nbody) if b not in parts["spark"]["lid"]}
    flip = np.diag([1.0, -1.0, -1.0])  # camera axes: OpenCV -> MuJoCo
    scores = {k: {p: [] for p in masks} for k in parts}
    import cv2
    half = min(out_width // 2, w)  # never upscale a small (RBO 640 x 480) video
    size = (half, round(hgt * half / w) // 2 * 2)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), (cap.get(5) or 30) / stride,
                             (2 * size[0], size[1]))
    stills = {}
    for t in range(n):
        ok, img = cap.read()
        if not ok:
            break
        if t % stride:
            continue
        md.qpos[:] = mm.qpos0
        md.joint(mj.jid).qpos[0] = q_ours[t]
        sd.qpos[:] = sm.qpos0
        sd.qpos[sm.jnt_qposadr[sj.jid]] = q_spark[t]
        seen = {"ours": _segment(mm, md, rm, cams[t], flip), "spark": _segment(sm, sd, rs, to_spark @ cams[t], flip)}
        panels = []
        for k, body in seen.items():
            panel = img.copy()
            for i, (p, bodies) in enumerate(parts[k].items()):
                pred = np.isin(body, list(bodies))
                det = masks[p].get(t)
                if det is not None:
                    m = det.full_mask(hgt, w)
                    if m.any():
                        scores[k][p].append(_iou(pred, m))
                color = np.array(COLORS[i % len(COLORS)])
                panel[pred] = (0.5 * panel[pred] + 0.5 * color).astype(np.uint8)
                contours, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel, contours, -1, color.tolist(), max(2, w // 400))
            panel = cv2.resize(panel, size, interpolation=cv2.INTER_AREA)
            if k == "ours":
                lines = ["Video2URDF (measured)", f"opens {np.rad2deg(opening[t]):.0f} deg"]
            else:
                lim = score["urdf_limits_deg"]
                lines = ["Muse Spark alone (placed where it fits best)",
                         f"opens {np.rad2deg(opening_spark[t]):.0f} deg"
                         + (f" (its limits {lim[0]:.0f}..{lim[1]:.0f} deg)" if lim else "")]
            sc = size[0] / 960
            for row, text in enumerate(lines):
                for col, th in (((0, 0, 0), 5), ((255, 255, 255), 2)):
                    cv2.putText(panel, text, (int(16 * sc), int((40 + 38 * row) * sc)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9 * sc, col, max(1, int(th * sc)))
            panels.append(panel)
        frame = np.hstack(panels)
        writer.write(frame)
        stills[t] = frame
    writer.release()
    for r in (rm, rs):
        r.close()
    summary = {k: {p: float(np.nanmean(v)) if v else None for p, v in d.items()} for k, d in scores.items()}
    summary.update({"case": case, "run": run, "frames_scored": {p: len(v) for p, v in scores["ours"].items()},
                    "stride": stride})
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "stills": stills}


def cv2_capture(video: Path):
    import cv2
    return cv2.VideoCapture(str(video))


def slide_limit_still(case: str, travel_m: float, out: Path, label: str) -> dict:
    """The measured drawer drawn at the frame where it is pulled out furthest: at its measured
    travel (left) and at the travel Spark alone claims (right), same geometry and camera, so the
    only difference is the claimed limit."""
    import cv2
    from .replay import COLORS
    c = CASES[case]
    measured = ROOT / c.measured
    rec = np.load(measured / "replay3d.npz")
    joint = next(k[2:] for k in rec.files if k.startswith("q_"))
    q = rec[f"q_{joint}"]
    t = int(np.nanargmax(q))
    cap = cv2_capture(ROOT / c.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    ok, img = cap.read()
    hgt, w = img.shape[:2]
    model = mujoco.MjModel.from_xml_path(str(measured / "model.xml"))
    model.vis.global_.offwidth, model.vis.global_.offheight = max(w, 640), max(hgt, 480)
    model.cam_fovy[model.camera("video").id] = np.rad2deg(2 * np.arctan(hgt / 2 / float(rec["f"])))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, hgt, w)
    renderer.enable_segmentation_rendering()
    jid = model.joint(joint).id
    moving = set(doctor.subtree(model, int(model.jnt_bodyid[jid])))
    flip = np.diag([1.0, -1.0, -1.0])
    reported = json.loads((measured / "slides.json").read_text())["joints"][0]["travel_mm"]
    panels = []
    for value, text in ((float(q[t]), f"Video2URDF: pulls out {reported:.0f} mm (measured)"),
                        (travel_m, f"{label}: {1000 * travel_m:.0f} mm")):
        data.qpos[:] = model.qpos0
        data.qpos[model.jnt_qposadr[jid]] = value
        body = _segment(model, data, renderer, rec["cam_to_world"][t], flip)
        pred = np.isin(body, list(moving))
        panel = img.copy()
        color = np.array(COLORS[0])
        panel[pred] = (0.5 * panel[pred] + 0.5 * color).astype(np.uint8)
        contours, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, color.tolist(), max(2, w // 400))
        k = w / 1920
        for col, th in (((0, 0, 0), 7), ((255, 255, 255), 3)):
            cv2.putText(panel, text, (int(30 * k), int(70 * k)), cv2.FONT_HERSHEY_SIMPLEX, 1.6 * k, col, max(1, int(th * k)))
        panels.append(panel)
    renderer.close()
    cv2.imwrite(str(out), np.hstack(panels))
    return {"frame": t, "measured_m": float(q[t]), "claimed_m": travel_m}
