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
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf.pipeline import video_info  # noqa: E402

WEB_HEIGHT = 720  # px, long videos are scaled so their shorter side is at most this
POLICY_DIR = ROOT / "data" / "policies" / "drawer_ppo_honest"  # scripts/run_drawer_policy.py's default --policy


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
        if g.get("group") == "4":  # a visual superseded by another (design under a SAM 3D mesh)
            continue
        out["geoms"].append({
            "mesh": g.get("mesh"), "type": g.get("type"), "rgba": floats(g.get("rgba"), [0.7, 0.7, 0.7, 1]),
            "pos": floats(g.get("pos"), [0, 0, 0]), "size": floats(g.get("size"), []),
            "axisangle": floats(g.get("axisangle"), None), "quat": floats(g.get("quat"), None),
            "material": g.get("material"),  # the video texture (texture.py), resolved in export
            "realistic": ((g.get("mesh") or "").endswith(("_sam3d", "_infer"))
                          or g.get("name", "").startswith(("inferred_", "design_"))),  # generated visual
            "measured": g.get("group") == "3"})  # hidden under a realistic visual in MuJoCo
    return out


def export_trace(log: Path, out: Path, width: int = 1200) -> str:
    """A log.md and the images it shows, as JPEGs at most ``width`` px wide. Returns the log's path
    relative to the model folder."""
    import re
    out.mkdir(parents=True, exist_ok=True)
    text = log.read_text()

    def copy(m):
        src = log.parent / m.group(2)
        if not src.exists():
            return m.group(0)
        img = cv2.imread(str(src))
        if img is None:
            return m.group(0)
        if img.shape[1] > width:
            img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        name = Path(m.group(2)).with_suffix(".jpg").name
        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return f"![{m.group(1)}]({name})"
    (out / "log.md").write_text(re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", copy, text))
    return f"{out.name}/log.md"


def joint_tracks(run: Path) -> dict[str, list]:
    """Joint name -> value per video frame (None where unmeasured)."""
    if (run / "static.json").exists():
        return {}
    if (run / "chain.json").exists():  # chain3d: rods on hinges
        return {f"{j['child']}_joint": j["q"] for j in json.loads((run / "chain.json").read_text())["joints"]
                if j["type"] != "fixed"}
    if (run / "slides.json").exists():
        return {f"drawer{i + 1}_joint": d["q"] for i, d in enumerate(json.loads((run / "slides.json").read_text())["drawers"])}
    if (run / "slide.json").exists():
        return {"drawer_joint": json.loads((run / "slide.json").read_text())["q"]}
    if (run / "hinge.json").exists() and (run / "model3d.json").exists():
        from video2urdf.physics import hinge_q  # in the model's sense: a door's model can negate it
        return {"lid_joint": [None if v != v else float(v) for v in hinge_q(run)]}
    joints = json.loads((run / "joints.json").read_text())["joints"]
    return {f"{j['child']}_joint": j["q"] for j in joints if j["type"] != "fixed"}


def web_video(src: Path, dst: Path, crf: int = 30):
    """H.264 for browsers: shorter side at most WEB_HEIGHT, no audio, index up front."""
    vf = f"scale='if(gt(iw,ih),-2,min({WEB_HEIGHT},iw))':'if(gt(iw,ih),min({WEB_HEIGHT},ih),-2)'"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-an", "-c:v", "libx264",
                    "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(dst)], check=True)


def policy_note(policy_dir: Path = POLICY_DIR) -> str | None:
    """Caption for robot_policy.mp4 (scripts/run_drawer_policy.py), from the policy's own tests:
    held-out placements in the training simulation (result.json) and unseen start poses on the
    full arm (arm_eval_cartesian.json)."""
    try:
        sim = json.loads((policy_dir / "result.json").read_text())
        arm = json.loads((policy_dir / "arm_eval_cartesian.json").read_text())["summary"]
        test = sim["test_policy"]
        n_sim, n_arm = test["episodes"], arm["episodes"]
        return (f"A PPO policy trained in MuJoCo on this measured drawer ({sim['steps'] / 1e6:.0f}M steps) opens it "
                f"(out at least 150 mm while pinching the handle) on {round(test['success_rate'] * n_sim)}/{n_sim} "
                f"held-out placements in simulation and {round(arm['success_rate'] * n_arm)}/{n_arm} unseen start "
                f"poses on the full Panda arm under Cartesian impedance control. "
                f"{100 * arm['held_share_of_travel']:.0f}% of the drawer's travel happens while the fingers pinch "
                "the handle. Simulation only: it has not run on a real robot.")
    except (OSError, KeyError, TypeError, ValueError):
        return None


def front(model_xml: Path) -> list[float] | None:
    """Where to look from, in the model's frame (horizontal): a hinged part's inner face looks
    toward its parent (a screen toward the keyboard), so from the child's centre toward the
    root's; a slide pulls out toward the viewer; a door on a vertical hinge is seen face-on from
    the side the phone filmed it from (replay3d.npz)."""
    import mujoco
    import numpy as np
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if model.njnt == 0:
        shape_file = model_xml.parent / 'shape_inference.json'
        if shape_file.exists():
            shape = json.loads(shape_file.read_text())
            if shape.get('static_drawer_side_assumed') and shape.get('active') is True:
                panel = next((v for v in shape.get('inferred_visuals', [])
                              if v.get('name', '').startswith('inferred_drawer_front_')), None)
                if panel and panel.get('front_axis') in (0, 1) and panel.get('front_sign') in (-1, 1):
                    direction = [0., 0., 0.]
                    direction[panel['front_axis']] = float(panel['front_sign'])
                    return direction
        return None
    j = 0
    replay = model_xml.parent / "replay3d.npz"
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
        v = data.xaxis[j].copy()
    elif abs(data.xaxis[j][2]) > 0.9 and replay.exists():  # a door: its thin direction, toward the phone
        body = model.jnt_bodyid[j]
        pts = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == body and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                m = model.geom_dataid[g]
                vert = model.mesh_vert[model.mesh_vertadr[m]: model.mesh_vertadr[m] + model.mesh_vertnum[m]]
                pts.append(data.geom_xpos[g] + vert @ data.geom_xmat[g].reshape(3, 3).T)
        if not pts:
            return None
        pts = np.vstack(pts)
        c = pts.mean(axis=0)
        v = np.linalg.svd(pts - c, full_matrices=False)[2][2]
        phone = np.load(replay)["cam_to_world"][:, :3, 3].mean(axis=0)  # cameras in the model's frame
        v = v if v @ (phone - c) > 0 else -v
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
        # MJCF/URDF still reference OBJ even when the web viewer prefers a coloured PLY.
        shutil.copy(run / file, dst / file)
        if file.endswith(".obj"):
            for line in (run / file).read_text().splitlines():
                if line.startswith("mtllib "):
                    mtl = Path(file).parent / line.split(maxsplit=1)[1]
                    if (run / mtl).exists():
                        shutil.copy(run / mtl, dst / mtl)
        if file.endswith(".obj") and (run / file).with_suffix(".ply").exists():  # the web gets vertex colours
            file = meshes[name_] = str(Path(file).with_suffix(".ply"))
            shutil.copy(run / file, dst / file)
    tree = body_tree(root.find("worldbody").find("body"))

    textures = {t.get("name"): t.get("file") for t in root.iter("texture") if t.get("file")}
    materials = {m.get("name"): textures.get(m.get("texture")) for m in root.iter("material")}
    for file in {f for f in materials.values() if f}:
        shutil.copy(run / file, dst / file)

    # URDF collision meshes may be different from MJCF's visual assets.
    urdf_root = ET.parse(run / "model.urdf").getroot()
    for element in list(urdf_root.iter("mesh")) + list(urdf_root.iter("texture")):
        file = element.get("filename")
        if not file:
            continue
        source, target = run / file, dst / file
        if not source.is_file():
            raise FileNotFoundError(f"URDF asset missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy(source, target)

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
    experimental_replay = run / "experimental_replay.mp4"
    replay_sources = ([experimental_replay] if (run / "experimental_alignment.json").exists() else []) + [run / "replay3d.mp4", run / "replay.mp4"]
    replay_experimental = False
    for key, src in (("phone", video), *(("replay", p) for p in replay_sources),
                     ("robot", run / "robot_demo.mp4"), ("policy", run / "robot_policy.mp4"),
                     ("physics", run / "physics_demo.mp4")):
        if key not in videos and src.exists():
            web_video(src, dst / f"{key}.mp4")
            videos[key] = f"{key}.mp4"
            if key == "replay":
                replay_experimental = src == experimental_replay
    note = policy_note() if "policy" in videos else None
    video_notes = {"policy": note} if note else {}  # key -> caption shown under that video
    meta = {}
    for f in ("model3d.json", "hinge.json"):
        if (run / f).exists():
            m = json.loads((run / f).read_text())
            meta.update({k: v for k, v in m.items() if k not in ("q", "camera_to_model")})
    report = run / "agent" / "report.md"
    traces = {}
    for folder in ("agent", "design"):  # how the model was made: every turn and image a Meta model saw
        log = run / folder / "log.md"
        full = run / f"{folder}_full"  # the whole agent run, when a later refit overwrote this run's log
        if (full / "log.md").exists():
            log = full / "log.md"
        if log.exists():
            traces[folder] = export_trace(log, dst / folder)
            source = log.parent / "SOURCE.txt"
            if source.exists():
                traces[folder + "_note"] = source.read_text().strip()
    phys = None
    if (run / "physics.json").exists():  # masses and friction (physics.apply)
        p = json.loads((run / "physics.json").read_text())
        phys = {"links": {k: {f: v[f] for f in ("material", "mass_kg")} for k, v in p["links"].items()},
                "joints": {k: {f: v[f] for f in ("type", "friction", "friction_measured", "held_range",
                                                    "drift_with_deg", "drift_without_deg") if f in v}
                           for k, v in p["joints"].items()}}
    def read_optional(filename):
        p = run / filename
        return json.loads(p.read_text()) if p.exists() else None

    # Keep the intermediate inferred mesh referenced by the provenance metadata,
    # even if texturing changed the active visual asset to an OBJ.
    inference = read_optional("shape_inference.json") or {}
    for item in inference.get("inferred_visuals", []):
        file = item.get("generated")
        if not file:
            continue
        rel = Path(file)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Invalid inferred mesh path: {file}")
        source, target = run / rel, dst / rel
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, target)

    spec = {"name": name, "fps": fps, "frames": n, "tree": tree, "joints": joint_tracks(run), "videos": videos,
            "video_notes": video_notes, "bundle": "model.zip",
            "measured": meta, "agent_report": report.read_text() if report.exists() else None,
            # how the model shown relates to the agent's run, when a person refit it or a log was replaced
            "provenance": (run / "provenance.txt").read_text().strip() if (run / "provenance.txt").exists() else None,
            # what the capture didn't measure, in plain words (written by hand per run)
            "unmeasured": (run / "unmeasured.txt").read_text().strip() if (run / "unmeasured.txt").exists() else None,
            "traces": traces,
            "three_d": (run / "model3d.json").exists(), "physics": phys,
            "appearance": read_optional("appearance.json"), "shape_inference": read_optional("shape_inference.json"),
            "texture": read_optional("texture.json"), "design": read_optional("design.json"),
            "measurement_source": read_optional("measurement_source.json"),
            "experimental_alignment": read_optional("experimental_alignment.json"),
            "experimental_comparison": read_optional("experimental_comparison.json"),
            "replay_experimental": replay_experimental,
            "physics_input_provenance": read_optional("physics_input_provenance.json"),
            "sam3d": read_optional("sam3d.json"), "sam3d_attempts": read_optional("sam3d_attempts.json"),
            "quality": read_optional("quality.json"),
            "front": front(run / "model.xml") if (run / "model3d.json").exists() else None}
    (dst / "model.json").write_text(json.dumps(spec))
    # Individual XML downloads reference relative meshes; bundle them for actual use.
    with zipfile.ZipFile(dst / "model.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in [dst / "model.urdf", dst / "model.xml", *sorted((dst / "meshes").rglob("*"))]:
            if p.is_file():
                archive.write(p, p.relative_to(dst))
        for name_ in ("model3d.json", "static.json", "appearance.json", "shape_inference.json", "texture.json", "design.json",
                      "measurement_source.json", "sam3d.json", "sam3d_attempts.json", "sam3d_status.json", "quality.json", "physics.json",
                      "track_unions.json", "repair.json", "manual_selection.json", "physics_input_provenance.json", "experimental_alignment.json", "experimental_comparison.json", "experimental_drawer_collision.json", "physics.original.json", "resume.json"):
            p = run / name_
            if p.exists():
                archive.write(p, name_)
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    print(f"{name}: {size / 1e6:.1f} MB -> {dst}")
    label = (spec.get('design') or {}).get('object') if (spec.get('design') or {}).get('applied') else None
    label = label or (spec.get('appearance') or {}).get('object')
    if isinstance(label, str):
        label = label.split(" (")[0].split(" - ")[0].strip()  # a tab name, not the model's whole description
        if len(label) > 40:
            label = label.split(" with ")[0]
    return {"name": name, "path": f"models/{name}/model.json", "three_d": spec["three_d"],
            **({'label': label} if isinstance(label, str) and label.strip() else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name:run_dir:video")
    ap.add_argument("--out", type=Path, default=ROOT / "docs")
    args = ap.parse_args()
    index = [export(name, Path(run), Path(video), args.out) for name, run, video in (r.split(":") for r in args.runs)]
    (args.out / "models" / "index.json").write_text(json.dumps(index))


if __name__ == "__main__":
    main()
