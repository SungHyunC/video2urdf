"""Rerun a published LiDAR measurement offline from its saved model decisions (scripts/reproduce.py).

A run's expensive, model-driven steps are Muse Spark's decisions (which parts to track, with
which SAM prompts on which frames, which part plays which role in the fit, the materials and
masses) and SAM 3.1's tracks. A sample bundle (``build_bundle``) saves exactly those for one
published capture. Everything after them runs here on the CPU with no model call: the agent's
own fit_lidar tool (hinge, link geometry from depth, URDF and MJCF, the MuJoCo replay from the
phone's camera poses with how much of each SAM mask the model covers) and the physics step on
Spark's saved estimate (joint friction from the hand-free still frames, the release check in
MuJoCo). ``offline`` makes sure of it: once the downloads are done, opening a network
connection raises, the API keys are out of the environment and the model clients refuse use.
"""

import contextlib
import hashlib
import json
import os
import re
import shutil
import socket
import ssl
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RELEASES = "https://github.com/SungHyunC/video2urdf/releases/download"
CAPTURE_URL = f"{RELEASES}/sample-data-v1/laptop_lidar_stray.zip"
CAPTURE_SHA256 = "0ee98f444e7287a00cf9a70355312cb4b5e6e494d9aeba9d4630b6b77d008996"
BUNDLE_NAME = "laptop_lidar_tracks"
BUNDLE_URL = f"{RELEASES}/sample-data-v2/{BUNDLE_NAME}.zip"
BUNDLE_SHA256 = "962b9b4a15534bb9330b532ba5876f3f1040141b24bcda216cc4f5df4dd198b5"  # scripts/make_reproduce_bundle.py
CASE = "laptop_lidar"  # the published model it reproduces: docs/models/<CASE>/model.json
BUNDLE_FILES = {
    "tracks.sam.txt": "SAM 3.1's masks for the five tracked parts on all 923 frames, in the Model API's text "
                      "format (decoded locally)",
    "parts.json": "the parts Muse Spark tracked: names, SAM prompts, and the frames and boxes it anchored them on",
    "roles.json": "the role Muse Spark gave each tracked part in fit_lidar (floor, base, plate, plate_closed, "
                  "open_indicator)",
    "physics_estimate.json": "Muse Spark's material, mass and contact friction per link, and the spans where the lid "
                             "stood still with no hand on it",
}
CAPTURE_CHECKED = ("rgb.mp4", "odometry.csv", "camera_matrix.csv")  # hashed into the bundle's manifest
API_KEYS = ("MODEL_API_KEY", "BASETEN_API_KEY", "FIREWORKS_API_KEY", "GLIMMER_API_KEY", "GLIMMER_URL",
            "MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET",
            "OPENAI_API_KEY", "HF_TOKEN")
ZIP_TIME = (1980, 1, 1, 0, 0, 0)  # fixed entry times: the same files make the same zip


class ReproduceError(Exception):
    """Something the rerun needs is missing or isn't what it should be; the message says what to do."""


class DownloadError(ReproduceError):
    def __init__(self, url: str, status: int | None, reason: str):
        super().__init__(f"{url}: " + (f"HTTP {status}" if status else reason))
        self.url, self.status, self.reason = url, status, reason


class NetworkRefused(RuntimeError):
    """Not an OSError on purpose: HTTP clients must not catch it and retry."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- downloads

def ssl_context() -> ssl.SSLContext:
    """Certificate authorities from certifi (installed with the openai client): the python.org
    macOS build has none of its own until its 'Install Certificates' step is run."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def download(url: str, dest: Path, log=print) -> Path:
    """``url`` into ``dest`` (through a .part file); raises DownloadError with the HTTP status."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "video2urdf-reproduce"})
    try:
        with urllib.request.urlopen(request, timeout=60, context=ssl_context()) as response, open(part, "wb") as f:
            total, done, shown = int(response.headers.get("Content-Length") or 0), 0, 0
            for block in iter(lambda: response.read(1 << 20), b""):
                f.write(block)
                done += len(block)
                if total and done - shown >= total / 4:
                    shown = done
                    log(f"  {done / 2 ** 20:.0f} of {total / 2 ** 20:.0f} MB")
    except urllib.error.HTTPError as error:
        part.unlink(missing_ok=True)
        raise DownloadError(url, error.code, str(error.reason)) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        part.unlink(missing_ok=True)
        raise DownloadError(url, None, str(getattr(error, "reason", error))) from error
    part.replace(dest)
    return dest


def fetch(source: str, cache: Path, label: str, expected_sha256: str | None = None, log=print) -> Path:
    """A local file or folder as given, or a URL downloaded into ``cache`` (kept: a rerun with
    the same file doesn't download again)."""
    if not re.match(r"https?://", source):
        path = Path(source).expanduser()
        if not path.exists():
            raise ReproduceError(f"{label}: {path} doesn't exist")
        return path
    dest = cache / source.rsplit("/", 1)[-1]
    if dest.is_file() and (expected_sha256 is None or sha256(dest) == expected_sha256):
        log(f"{label}: {dest} (downloaded before)")
        return dest
    log(f"{label}: downloading {source}")
    return download(source, dest, log=log)


# ----------------------------------------------------------------------------- the bundle

def capture_record(capture_zip: Path, url: str) -> dict:
    """What the bundle's manifest says about the capture its tracks were made on."""
    record = {"url": url, "sha256": sha256(capture_zip), "files": {}}
    with zipfile.ZipFile(capture_zip) as z:
        names = [n for n in z.namelist() if "__MACOSX" not in n]
        for want in CAPTURE_CHECKED:
            member = next((n for n in names if n.rsplit("/", 1)[-1] == want), None)
            if member is None:
                raise ReproduceError(f"{capture_zip} has no {want}: not a Stray Scanner capture")
            h = hashlib.sha256()
            with z.open(member) as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            record["files"][want] = h.hexdigest()
        record["depth_frames"] = sum(1 for n in names if re.search(r"(^|/)depth/\d+\.png$", n))
    return record


def build_bundle(run: Path, capture_zip: Path, out_zip: Path, source: str, capture_url: str = CAPTURE_URL) -> dict:
    """Zip the saved model decisions of ``run`` (a track folder) with a manifest of their hashes
    and the capture they belong to. Fixed entry times and order: the same files give the same
    zip. Returns the manifest."""
    missing = [name for name in BUNDLE_FILES if not (run / name).is_file()]
    if missing:
        raise ReproduceError(f"{run} lacks {missing}")
    manifest = {
        "bundle": BUNDLE_NAME, "format": 1, "case": CASE,
        "what": "The saved model decisions for the published iPhone LiDAR laptop: everything scripts/reproduce.py "
                "needs on top of the capture to rerun the measurement without calling a model.",
        "published_model": f"docs/models/{CASE}/model.json",
        "made_from": source,
        "capture": capture_record(capture_zip, capture_url),
        "files": {name: {"sha256": sha256(run / name), "bytes": (run / name).stat().st_size, "what": what}
                  for name, what in BUNDLE_FILES.items()},
    }
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w") as z:
        entries = [("MANIFEST.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())]
        entries += [(name, (run / name).read_bytes()) for name in sorted(BUNDLE_FILES)]
        for name, data in entries:
            info = zipfile.ZipInfo(f"{BUNDLE_NAME}/{name}", date_time=ZIP_TIME)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data, compresslevel=9)
    return manifest


def open_bundle(path: Path, work: Path) -> tuple[Path, dict]:
    """The bundle's folder (unpacked into ``work`` from its zip, or as given) and its manifest,
    with every file checked against the manifest's hashes."""
    from .ingest import unpack
    if path.is_file():
        if not zipfile.is_zipfile(path):
            raise ReproduceError(f"{path} isn't a zip file")
        dest = work / "bundle"
        shutil.rmtree(dest, ignore_errors=True)
        unpack(path, dest, log=lambda _: None)
        path = dest
    found = sorted(path.rglob("MANIFEST.json"), key=lambda p: len(p.parts))
    if not found:
        raise ReproduceError(f"{path} has no MANIFEST.json: not a bundle made by scripts/make_reproduce_bundle.py")
    root = found[0].parent
    manifest = json.loads(found[0].read_text())
    absent = [name for name in BUNDLE_FILES if name not in manifest.get("files", {})]
    if absent:
        raise ReproduceError(f"the bundle doesn't hold {absent}; this rerun needs them and won't ask a model instead")
    for name, info in manifest["files"].items():
        if not (root / name).is_file():
            raise ReproduceError(f"the bundle's {name} is missing; unpack the whole zip again")
        if sha256(root / name) != info["sha256"]:
            raise ReproduceError(f"the bundle's {name} doesn't match its manifest: the file is damaged or edited")
    return root, manifest


def open_capture(path: Path, cache: Path, manifest: dict) -> Path:
    """The Stray Scanner folder (rgb.mp4, depth/, confidence/, odometry.csv, camera_matrix.csv),
    unpacked from the release zip into ``cache`` (once) or as given, and checked against the
    capture the bundle's tracks were made on."""
    from .ingest import unpack
    from .stray import find_capture
    expected = manifest["capture"]
    if path.is_file():
        got = sha256(path)
        if got != expected["sha256"]:
            raise ReproduceError(f"{path} isn't the capture these tracks were made on (sha256 {got[:12]}..., "
                                 f"expected {expected['sha256'][:12]}...): download {expected['url']}")
        dest = cache / f"capture_{got[:12]}"
        if not (dest / ".unpacked").is_file():
            shutil.rmtree(dest, ignore_errors=True)
            unpack(path, dest, log=lambda _: None)
            (dest / ".unpacked").write_text(got + "\n")
        path = dest
    try:
        root = find_capture(path)
    except FileNotFoundError as error:
        raise ReproduceError(f"{path}: {error}") from error
    for name, digest in expected["files"].items():
        if not (root / name).is_file():
            raise ReproduceError(f"the capture at {root} has no {name}")
        if sha256(root / name) != digest:
            raise ReproduceError(f"the capture's {name} at {root} isn't the one these tracks were made on")
    depth = len(list((root / "depth").glob("*.png"))) if (root / "depth").is_dir() else 0
    confidence = len(list((root / "confidence").glob("*.png"))) if (root / "confidence").is_dir() else 0
    if depth < expected["depth_frames"] or confidence < expected["depth_frames"]:
        raise ReproduceError(f"the capture at {root} has {depth} depth and {confidence} confidence images; "
                             f"{expected['depth_frames']} of each expected: unpack the whole zip again")
    return root


def preflight(work: Path) -> None:
    """What the replay needs from this machine, checked before the long part: MuJoCo's offscreen
    renderer (a GL context) and an H.264 ('avc1') video writer in OpenCV."""
    import cv2
    import mujoco
    try:
        renderer = mujoco.Renderer(mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><geom size='0.1'/></worldbody></mujoco>"), 32, 32)
        renderer.render()
        renderer.close()
    except Exception as error:  # mujoco raises plain exceptions when there is no GL
        raise ReproduceError(f"MuJoCo can't render here ({error}); on a machine without a display, set MUJOCO_GL=egl "
                             "(or osmesa) and run again") from error
    work.mkdir(parents=True, exist_ok=True)
    probe = work / "h264_probe.mp4"
    writer = cv2.VideoWriter(str(probe), cv2.VideoWriter_fourcc(*"avc1"), 30, (64, 64))
    opened = writer.isOpened()
    if opened:
        writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    probe.unlink(missing_ok=True)
    if not opened:
        raise ReproduceError("this OpenCV build can't write H.264 ('avc1') video, which the replay is written as "
                             f"(OpenCV {cv2.__version__}); use an OpenCV with an H.264 encoder")


# ----------------------------------------------------------------------------- no model, no network

@dataclass
class NoModel:
    """Stands in for a model client (Muse Spark, SAM 3.1): this mode makes no model calls."""
    name: str

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        raise RuntimeError(f"reproduce mode makes no model calls, but the fit tried to use {self.name} ({attr})")


@contextlib.contextmanager
def offline():
    """Inside: no API keys in the environment, and any internet connection or name lookup raises
    NetworkRefused (local sockets and pipes, which the decoder's worker processes use, still work)."""
    removed = {k: os.environ.pop(k) for k in API_KEYS if k in os.environ}
    saved = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo, socket.create_connection

    def refuse(what):
        raise NetworkRefused(f"reproduce mode is offline after its downloads, but something tried to open a network "
                             f"connection ({what}); nothing in this mode should")

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            refuse(address)
        return saved[0](sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            refuse(address)
        return saved[1](sock, address)

    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex
    socket.getaddrinfo = lambda host, *a, **k: refuse(host)
    socket.create_connection = lambda address, *a, **k: refuse(address)
    try:
        yield removed
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo, socket.create_connection = saved
        os.environ.update(removed)


# ----------------------------------------------------------------------------- the rerun

def rerun(bundle: Path, capture: Path, out: Path, reference: int | None, log=print) -> dict:
    """The agent's fit_lidar tool on the bundle's tracks and roles, then the physics step on its
    saved estimate, into ``out``. ``reference`` pins the reference frame (None: the code picks).
    Returns the fit tool's text and the physics log."""
    from . import physics
    from .agent import Agent
    from .pipeline import video_info
    out.mkdir(parents=True)
    for name in BUNDLE_FILES:
        shutil.copy(bundle / name, out / name)
    video = out / f"{CASE}.mp4"  # tracks are filed under the video's name, and the model is named after it
    shutil.copy(capture / "rgb.mp4", video)
    roles = {k: v for k, v in json.loads((out / "roles.json").read_text()).items() if v}
    agent = Agent(video, out, capture=capture, client=NoModel("Muse Spark"), sam=NoModel("SAM 3.1"))
    agent.tracked = True  # the bundle's tracks stand in for the agent's track tool
    try:
        fit_text = agent.tool_fit_lidar(**roles, reference=reference)
    except ValueError as error:  # the fit's own refusals (e.g. a given reference frame it can't use)
        raise ReproduceError(f"fit_lidar stopped: {error}") from error
    (out / "fit_lidar.txt").write_text(fit_text + "\n")
    if not (out / "model.xml").is_file():
        raise ReproduceError(f"fit_lidar wrote no model:\n{fit_text}")
    _, _, fps = video_info(video)
    lines = []
    physics.apply_estimate(out, json.loads((out / "physics_estimate.json").read_text()), fps, log=lines.append)
    (out / "physics.txt").write_text("\n".join(lines) + "\n")
    return {"fit": fit_text, "physics": lines}


# ----------------------------------------------------------------------------- the comparison

def coverage_from_text(text: str | None) -> dict | None:
    """The replay coverage a provenance note states ('... replay 95% / 94%': lid / base)."""
    m = re.search(r"replay (\d+(?:\.\d+)?)% / (\d+(?:\.\d+)?)%", text or "")
    return {"lid": float(m.group(1)), "base": float(m.group(2))} if m else None


def _opening(q: list, shut_deg: float) -> np.ndarray:
    """Per-frame opening in degrees (0 = shut) from the joint values (radians, None where
    unmeasured): q = 0 is the reference pose, opened ``shut_deg``; q's sign follows the fitted
    axis, so the shut end is the end of q's range that sits that far from the reference."""
    q = np.array([np.nan if v is None else v for v in q], float)
    shut = np.deg2rad(shut_deg)
    toward = 1.0 if abs(np.nanmax(q) - shut) < abs(np.nanmin(q) + shut) else -1.0
    return np.rad2deg(shut - toward * q)


def values(measured: dict, q: list, physics_report: dict | None, coverage: dict | None) -> dict:
    """The compared quantities, alike for the published model.json and a rerun's files."""
    joint = (physics_report or {}).get("joints", {}).get("lid_joint", {})
    return {
        "reference_frame": measured["reference_frame"],
        "lid_mm": tuple(measured["plate_size_mm"]),
        "lid_thickness_mm": measured["plate_thickness_mm"],
        "base_thickness_mm": measured["base_thickness_mm"],
        "limits_deg": tuple(measured["hinge_range_deg"]),
        "opening_deg": tuple(measured["opening_range_deg"]),
        "opening_on_reference_deg": measured["opening_on_reference_deg"],
        "plate_frames": (measured["frames_with_plate"], measured["frames"]),
        "from_closed": (measured.get("from_closed_part"), measured.get("from_open_object")),
        "residual_mm": 1000 * measured["residual_m"],
        "span_mm": measured["plate_span_along_hinge_mm"],
        "axis": np.array(measured["axis"], float), "point": np.array(measured["point"], float),
        "opening_per_frame": _opening(q, measured["opening_on_reference_deg"]),
        "coverage": coverage,
        "friction_nm": joint.get("friction"),
        "drift_with_deg": max(map(abs, joint["drift_with_deg"])) if joint.get("drift_with_deg") else None,
        "drift_without_deg": max(map(abs, joint["drift_without_deg"])) if joint.get("drift_without_deg") else None,
    }


def published_values(model_json: dict) -> dict:
    return values(model_json["measured"], model_json["joints"]["lid_joint"], model_json.get("physics"),
                  coverage_from_text(model_json.get("provenance")))


def rerun_values(out: Path) -> dict:
    meta = json.loads((out / "model3d.json").read_text())
    hinge = json.loads((out / "hinge.json").read_text())
    report = json.loads((out / "physics.json").read_text()) if (out / "physics.json").exists() else None
    cover = {k: 100 * v for k, v in meta.get("mask_coverage", {}).items() if v is not None} or None
    return values({**meta, **{k: v for k, v in hinge.items() if k != "q"}}, hinge["q"], report, cover)


@dataclass
class Row:
    quantity: str
    published: str
    reproduced: str
    difference: str


def _f(x, digits: int) -> str:
    return f"{x:.{digits}f}"


def _pair(x, digits: int, sep: str) -> str:
    return f"{x[0]:.{digits}f}{sep}{x[1]:.{digits}f}"


def _delta(a, b, digits: int) -> str:
    """b - a, signed, for a number or a pair; '0' when equal at that precision."""
    if isinstance(a, tuple):
        return " / ".join(_delta(x, y, digits) for x, y in zip(a, b))
    d = round(b - a, digits)
    return "0" if d == 0 else f"{d:+.{digits}f}"


def compare(pub: dict, rep: dict) -> list[Row]:
    same_ref = pub["reference_frame"] == rep["reference_frame"]
    zero = "" if same_ref else " (zero at each fit's reference frame)"
    rows = [
        Row("reference frame (the joint's zero)", str(pub["reference_frame"]), str(rep["reference_frame"]),
            "same" if same_ref else "other frame"),
        Row("lid size, mm", _pair(pub["lid_mm"], 1, " x "), _pair(rep["lid_mm"], 1, " x "),
            _delta(pub["lid_mm"], rep["lid_mm"], 3)),
        Row("lid thickness, mm", _f(pub["lid_thickness_mm"], 2), _f(rep["lid_thickness_mm"], 2),
            _delta(pub["lid_thickness_mm"], rep["lid_thickness_mm"], 3)),
        Row("base thickness, mm", _f(pub["base_thickness_mm"], 2), _f(rep["base_thickness_mm"], 2),
            _delta(pub["base_thickness_mm"], rep["base_thickness_mm"], 3)),
        Row("hinge range (joint limits), deg" + zero, _pair(pub["limits_deg"], 1, " .. "),
            _pair(rep["limits_deg"], 1, " .. "), _delta(pub["limits_deg"], rep["limits_deg"], 2)),
        Row("opening seen, deg (0 = shut)", _pair(pub["opening_deg"], 1, " .. "), _pair(rep["opening_deg"], 1, " .. "),
            _delta(pub["opening_deg"], rep["opening_deg"], 2)),
        Row("opening on the reference frame, deg", _f(pub["opening_on_reference_deg"], 1),
            _f(rep["opening_on_reference_deg"], 1),
            _delta(pub["opening_on_reference_deg"], rep["opening_on_reference_deg"], 2)),
        Row("frames with a measured lid plane", "%d of %d" % pub["plate_frames"], "%d of %d" % rep["plate_frames"],
            _delta(float(pub["plate_frames"][0]), float(rep["plate_frames"][0]), 0)),
    ]
    if None not in pub["from_closed"] + rep["from_closed"]:
        rows.append(Row("  of them from plate_closed's mask: shut / raised", "%d / %d" % pub["from_closed"],
                        "%d / %d" % rep["from_closed"], _delta(tuple(map(float, pub["from_closed"])),
                                                               tuple(map(float, rep["from_closed"])), 0)))
    rows += [
        Row("hinge fit residual, mm", _f(pub["residual_mm"], 2), _f(rep["residual_mm"], 2),
            _delta(pub["residual_mm"], rep["residual_mm"], 3)),
        Row("lid span along the hinge, mm", _f(pub["span_mm"], 1), _f(rep["span_mm"], 1),
            _delta(pub["span_mm"], rep["span_mm"], 3)),
    ]
    if same_ref:  # axis and point are in the reference camera's coordinates: comparable on the same frame only
        turn = np.rad2deg(np.arccos(np.clip(abs(pub["axis"] @ rep["axis"]) /
                                            (np.linalg.norm(pub["axis"]) * np.linalg.norm(rep["axis"])), -1, 1)))
        off = pub["point"] - rep["point"]
        a = pub["axis"] / np.linalg.norm(pub["axis"])
        shift = 1000 * np.linalg.norm(off - (off @ a) * a)  # between the two lines, across the axis
        rows.append(Row("hinge line vs the published one", "", "", f"{turn:.3f} deg, {shift:.2f} mm apart"))
    n = min(len(pub["opening_per_frame"]), len(rep["opening_per_frame"]))
    both = np.isfinite(pub["opening_per_frame"][:n]) & np.isfinite(rep["opening_per_frame"][:n])
    if both.any():
        d = np.abs(pub["opening_per_frame"][:n][both] - rep["opening_per_frame"][:n][both])
        rows.append(Row(f"opening on each frame both measured ({both.sum()}), deg", "", "",
                        f"median {np.median(d):.2f}, max {d.max():.2f}"))
    for link in ("lid", "base"):
        p, r = (pub["coverage"] or {}).get(link), (rep["coverage"] or {}).get(link)
        if p is None and r is None:
            continue
        diff = "" if p is None or r is None else (
            ("same to the whole %" if round(r) == round(p) else f"{round(r) - round(p):+d} pts")
            if p == round(p) else _delta(p, r, 1) + " pts")
        rows.append(Row(f"replay covers the {link}'s SAM masks, %", "-" if p is None else f"{p:g}",
                        "-" if r is None else _f(r, 1), diff))
    for key, label, digits in (("friction_nm", "hinge friction, N m", 3),
                               ("drift_with_deg", "released for 2 s with it, worst drift, deg", 2),
                               ("drift_without_deg", "released without it, worst drift, deg", 1)):
        p, r = pub[key], rep[key]
        if p is None and r is None:
            continue
        rows.append(Row(label, "-" if p is None else _f(p, digits), "-" if r is None else _f(r, digits),
                        "" if p is None or r is None else _delta(p, r, digits + 1)))
    return rows


def text_table(rows: list[Row]) -> str:
    head = Row("quantity", "published", "reproduced", "difference")
    cols = [max(len(getattr(r, f)) for r in [head, *rows]) for f in ("quantity", "published", "reproduced")]
    lines = []
    for r in [head, None, *rows]:
        if r is None:
            lines.append("  ".join("-" * c for c in cols) + "  " + "-" * 10)
            continue
        lines.append(f"{r.quantity:<{cols[0]}}  {r.published:<{cols[1]}}  {r.reproduced:<{cols[2]}}  {r.difference}")
    return "\n".join(lines)


def markdown_table(rows: list[Row]) -> str:
    lines = ["| quantity | published | reproduced | difference |", "|---|---|---|---|"]
    lines += [f"| {r.quantity} | {r.published} | {r.reproduced} | {r.difference} |" for r in rows]
    return "\n".join(lines)
