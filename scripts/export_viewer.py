"""Export fitted runs for the web viewer (docs/, served as a static page).

For each run: the MJCF's body tree (positions, joints, meshes, colours) as JSON, the joint
values per video frame, the STL meshes, URDF/MJCF downloads, and web-sized videos (phone
video, replay over it, robot demo).

Usage:
  .venv/bin/python scripts/export_viewer.py \\
      laptop_lidar:data/tracks_agent/laptop_lidar:data/real/laptop_lidar/laptop_lidar.mp4 \\
      notebook:data/tracks/notebook_2:data/real/notebook_2.MOV [--out docs]
"""

import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf.pipeline import video_info  # noqa: E402

WEB_HEIGHT = 720  # px, long videos are scaled so their shorter side is at most this


def floats(s: str | None, default):
    return [float(v) for v in s.split()] if s else default


def body_tree(el: ET.Element) -> dict:
    joint = el.find("joint")
    out = {
        "name": el.get("name"),
        "pos": floats(el.get("pos"), [0, 0, 0]),
        "joint": None if joint is None else {
            "name": joint.get("name"), "type": joint.get("type", "hinge"),
            "axis": floats(joint.get("axis"), [0, 0, 1]), "range": floats(joint.get("range"), [0, 0])},
        "geoms": [],
        "children": [body_tree(c) for c in el.findall("body")],
    }
    for g in el.findall("geom"):
        out["geoms"].append({
            "mesh": g.get("mesh"), "type": g.get("type"), "rgba": floats(g.get("rgba"), [0.7, 0.7, 0.7, 1]),
            "pos": floats(g.get("pos"), [0, 0, 0]), "size": floats(g.get("size"), []),
            "axisangle": floats(g.get("axisangle"), None),
            "material": g.get("material"),  # the video texture (texture.py), resolved in export
            "realistic": (g.get("mesh") or "").endswith("_sam3d"),  # SAM 3D's mesh (scripts/sam3d_meshes.py)
            "measured": g.get("group") == "3"})  # hidden under a realistic visual in MuJoCo
    return out


def joint_tracks(run: Path) -> dict[str, list]:
    """Joint name -> value per video frame (None where unmeasured)."""
    if (run / "slides.json").exists():
        return {f"drawer{i + 1}_joint": d["q"] for i, d in enumerate(json.loads((run / "slides.json").read_text())["drawers"])}
    if (run / "hinge.json").exists() and (run / "model3d.json").exists():
        return {"lid_joint": json.loads((run / "hinge.json").read_text())["q"]}
    joints = json.loads((run / "joints.json").read_text())["joints"]
    return {f"{j['child']}_joint": j["q"] for j in joints if j["type"] != "fixed"}


def web_video(src: Path, dst: Path, crf: int = 30):
    """H.264 for browsers: shorter side at most WEB_HEIGHT, no audio, index up front."""
    vf = f"scale='if(gt(iw,ih),-2,min({WEB_HEIGHT},iw))':'if(gt(iw,ih),min({WEB_HEIGHT},ih),-2)'"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-an", "-c:v", "libx264",
                    "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(dst)], check=True)


def front(model_xml: Path) -> list[float] | None:
    """Where to look from, in the model's frame (horizontal): a hinged part's inner face looks
    toward its parent (a screen toward the keyboard), so from the child's centre toward the
    root's; a slide pulls out toward the viewer."""
    import mujoco
    import numpy as np
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if model.njnt == 0:
        return None
    j = 0
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
        v = data.xaxis[j].copy()
    else:
        v = data.subtree_com[1] - data.subtree_com[model.jnt_bodyid[j]]
    v[2] = 0
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 1e-6 else None


def export(name: str, run: Path, video: Path, out: Path) -> dict:
    dst = out / "models" / name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meshes").mkdir(parents=True)
    root = ET.parse(run / "model.xml").getroot()
    meshes = {m.get("name"): m.get("file") for m in root.iter("mesh")}
    for name_, file in meshes.items():
        if file.endswith(".obj") and (run / file).with_suffix(".ply").exists():  # the web gets vertex colours
            file = meshes[name_] = str(Path(file).with_suffix(".ply"))
        shutil.copy(run / file, dst / file)
    tree = body_tree(root.find("worldbody").find("body"))

    textures = {t.get("name"): t.get("file") for t in root.iter("texture") if t.get("file")}
    materials = {m.get("name"): textures.get(m.get("texture")) for m in root.iter("material")}
    for file in {f for f in materials.values() if f}:
        shutil.copy(run / file, dst / file)

    def fix_mesh_paths(b):
        for g in b["geoms"]:
            if g["mesh"]:
                g["mesh"] = meshes[g["mesh"]]
            g["texture"] = materials.get(g.pop("material"))
        for c in b["children"]:
            fix_mesh_paths(c)

    fix_mesh_paths(tree)
    for f in ("model.urdf", "model.xml"):
        shutil.copy(run / f, dst / f)
    _, n, fps = video_info(video)
    videos = {}
    for key, src in (("phone", video), ("replay", run / "replay3d.mp4"), ("replay", run / "replay.mp4"),
                     ("robot", run / "robot_demo.mp4"), ("physics", run / "physics_demo.mp4")):
        if key not in videos and src.exists():
            web_video(src, dst / f"{key}.mp4")
            videos[key] = f"{key}.mp4"
    meta = {}
    for f in ("model3d.json", "hinge.json"):
        if (run / f).exists():
            m = json.loads((run / f).read_text())
            meta.update({k: v for k, v in m.items() if k not in ("q", "camera_to_model")})
    report = run / "agent" / "report.md"
    phys = None
    if (run / "physics.json").exists():  # masses and friction (physics.apply)
        p = json.loads((run / "physics.json").read_text())
        phys = {"links": {k: {f: v[f] for f in ("material", "mass_kg")} for k, v in p["links"].items()},
                "joints": {k: {f: v[f] for f in ("type", "friction", "friction_measured", "held_range",
                                                    "drift_with_deg", "drift_without_deg") if f in v}
                           for k, v in p["joints"].items()}}
    spec = {"name": name, "fps": fps, "frames": n, "tree": tree, "joints": joint_tracks(run), "videos": videos,
            "measured": meta, "agent_report": report.read_text() if report.exists() else None,
            "three_d": (run / "model3d.json").exists(), "physics": phys,
            "front": front(run / "model.xml") if (run / "model3d.json").exists() else None}
    (dst / "model.json").write_text(json.dumps(spec))
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    print(f"{name}: {size / 1e6:.1f} MB -> {dst}")
    return {"name": name, "path": f"models/{name}/model.json", "three_d": spec["three_d"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name:run_dir:video")
    ap.add_argument("--out", type=Path, default=ROOT / "docs")
    args = ap.parse_args()
    index = [export(name, Path(run), Path(video), args.out) for name, run, video in (r.split(":") for r in args.runs)]
    (args.out / "models" / "index.json").write_text(json.dumps(index))


if __name__ == "__main__":
    main()
