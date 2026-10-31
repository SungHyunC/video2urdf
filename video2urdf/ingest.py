"""Accept whatever a user films: find out what the input is from its content, then normalise it.

Kinds (``detect``):
- ``stray``: an iPhone LiDAR capture from the Stray Scanner app (rgb.mp4, depth/, confidence/,
  camera_matrix.csv, odometry.csv), as a folder at any depth or inside a zip/tar;
- ``mapanything``: the same layout with depth estimated from RGB (mapanything.json);
- ``rgbd``: a fixed-camera RGB-D folder (depth.npz in metres, intrinsics.json, one video), as
  scripts/rbo_eval.py prepares;
- ``video``: an ordinary video (iPhone .MOV with HEVC/HDR/rotation, Android .mp4, ...);
- ``images``: a folder of photos or frames, in natural name order;
- ``image`` (one photo) and ``unknown``: unusable, with a reason.

Normalising (``normalise``) writes everything the pipeline reads into the run's input folder;
the user's files are never modified. A plain video becomes H.264 yuv420p at a constant frame
rate, upright (rotation metadata applied), SDR (HLG/PQ tone-mapped), at most MAX_FPS and
MAX_SIDE px, without audio or metadata (iPhone files carry GPS). A depth capture is copied
frame for frame: its RGB frames are never resampled apart from its depth and poses; a very long
capture is decimated with all of them together.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

MAX_FPS = 30.0  # plain video: higher frame rates are decimated (60 -> 30, 120 -> 30, 50 -> 25)
MIN_FPS = 10.0  # long videos lower their frame rate down to this before being trimmed
MAX_FRAMES = 1800  # plain video: frames after normalisation (60 s at 30 fps)
MAX_SIDE = 1920  # px, long side of a normalised plain video
MIN_SIDE = 120  # px, short side below which a video is unusable
MAX_CAPTURE_FRAMES = 3600  # depth capture: longer ones are decimated with depth and poses together
MIN_FRAMES = 8  # fewer frames than this can't show a joint move
MIN_IMAGES = 8  # a folder needs at least this many photos to be an image sequence
IMAGE_FPS = 10.0  # frame rate given to an image sequence
MAX_UNPACKED = 40 * 2 ** 30  # bytes; refuse archives that would unpack to more (zip bombs)
DEPTH_KINDS = ("stray", "mapanything", "rgbd")
JUNK = ("__MACOSX", ".DS_Store", "Thumbs.db")


# ----------------------------------------------------------------------------- sniffing

def _head(path: Path, n: int = 64) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read(n)
    except OSError:
        return b""


ISO_IMAGE_BRANDS = {b"heic", b"heix", b"heim", b"heis", b"mif1", b"msf1", b"avif", b"avis"}


def image_magic(head: bytes) -> str | None:
    """The image format from a file's first bytes (None if it isn't a still image)."""
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head.startswith(b"BM") and len(head) > 14:
        return "bmp"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp"
    if head[4:8] == b"ftyp" and head[8:12] in ISO_IMAGE_BRANDS:
        return "heic"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return None


def media_magic(head: bytes) -> bool:
    """A video container's first bytes: ISO BMFF (mp4/mov/3gp), Matroska/WebM, AVI, MPEG-TS/PS, FLV, ASF."""
    return (head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot")
            or head.startswith(b"\x1a\x45\xdf\xa3") or (head.startswith(b"RIFF") and head[8:12] == b"AVI ")
            or head[:1] == b"\x47" or head.startswith(b"\x00\x00\x01\xba") or head.startswith(b"FLV")
            or head.startswith(b"\x30\x26\xb2\x75"))


def looks_textual(head: bytes) -> bool:
    return all(32 <= b < 127 or b in (9, 10, 13) for b in head)


def sniff(path: Path) -> str:
    """What a file is, from its content: 'image', 'zip', 'tar', 'video' or 'unknown'."""
    head = _head(path)
    if not head:
        return "unknown"
    if image_magic(head) == "gif":  # an animated GIF is a (tiny) video
        probe = ffprobe(path)
        return "video" if probe and video_stream(probe) is not None else "image"
    if image_magic(head):
        return "image"
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        return "zip" if zipfile.is_zipfile(path) else "unknown"
    if media_magic(head) or (not looks_textual(head) and path.stat().st_size > 100_000):
        probe = ffprobe(path)
        if probe and video_stream(probe) is not None:
            return "video"
    if head[:2] in (b"\x1f\x8b", b"BZ") or head[:6] == b"\xfd7zXZ\x00" or head[257:262] == b"ustar" or \
            _head(path, 512)[257:262] == b"ustar":
        try:
            if tarfile.is_tarfile(path):
                return "tar"
        except (OSError, tarfile.TarError):
            pass
    return "unknown"


def ffprobe(path: Path) -> dict | None:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode:
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return None


def video_stream(probe: dict) -> dict | None:
    """The first real video stream (not cover art, not a single still frame)."""
    for s in probe.get("streams", []):
        if s.get("codec_type") != "video" or (s.get("disposition") or {}).get("attached_pic"):
            continue
        frames = _int(s.get("nb_frames"))
        duration = _float(s.get("duration")) or _float((probe.get("format") or {}).get("duration"))
        if frames == 1 or (frames is None and not duration):
            continue
        return s
    return None


def _int(x) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _float(x) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _rate(x) -> float | None:
    try:
        f = Fraction(str(x))
        return float(f) if f > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


# ----------------------------------------------------------------------------- detection

@dataclass
class Found:
    kind: str  # stray / mapanything / rgbd / video / images / image / unknown
    path: Path  # the capture folder, video file, image folder or file
    reason: str  # how it was recognised, or why it is unusable
    images: list[Path] = field(default_factory=list)  # for kind 'images'
    archive: Path | None = None  # the archive it came out of


def _skip(p: Path) -> bool:
    return any(part in JUNK or part.startswith("._") for part in p.parts)


def is_stray(folder: Path) -> bool:
    return ((folder / "rgb.mp4").is_file() and (folder / "depth").is_dir() and (folder / "odometry.csv").is_file()
            and (folder / "camera_matrix.csv").is_file())


def is_rgbd(folder: Path) -> bool:
    return (folder / "intrinsics.json").is_file() and (folder / "depth.npz").is_file()


def _walk_dirs(root: Path, max_depth: int = 6):
    """Folders under root, shallowest first (junk skipped)."""
    level = [root]
    for _ in range(max_depth + 1):
        nxt = []
        for d in level:
            if d != root and _skip(d.relative_to(root)):
                continue
            yield d
            try:
                nxt += sorted(p for p in d.iterdir() if p.is_dir() and not p.is_symlink())
            except OSError:
                pass
        level = nxt
        if not level:
            break


def _files(folder: Path) -> list[Path]:
    try:
        return sorted(p for p in folder.iterdir() if p.is_file() and not _skip(Path(p.name)))
    except OSError:
        return []


def natural_key(p: Path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)]


def detect(path: Path) -> Found:
    """Recognise an input from its content (names and extensions are only used for ordering)."""
    path = Path(path)
    if not path.exists():
        return Found("unknown", path, f"{path} does not exist")
    if path.is_file():
        kind = sniff(path)
        if kind == "image":
            return Found("image", path, f"a single {image_magic(_head(path))} photo: a photo can't show a part move; "
                                        "film a video (or give a folder of photos taken while it moves)")
        if kind == "video":
            return Found("video", path, "a video file")
        if kind in ("zip", "tar"):
            return Found(kind, path, f"a {kind} archive")
        return Found("unknown", path, "not a video, image or archive that ffmpeg/zip/tar can read")
    return detect_folder(path)


def detect_folder(root: Path) -> Found:
    dirs = list(_walk_dirs(root))
    for d in dirs:  # a depth capture anywhere inside wins over loose videos next to it
        if is_stray(d):
            if (d / "mapanything.json").is_file():
                return Found("mapanything", d, "a capture in the Stray layout whose depth MapAnything estimated")
            return Found("stray", d, "a Stray Scanner LiDAR capture (rgb.mp4, depth/, odometry.csv, camera_matrix.csv)")
        if is_rgbd(d):
            videos = [f for f in _files(d) if sniff(f) == "video"]
            if videos:
                return Found("rgbd", d, "a fixed-camera RGB-D folder (depth.npz, intrinsics.json, video)")
    archives, videos, images_by_dir = [], [], {}
    for d in dirs:
        for f in _files(d):
            kind = sniff(f)
            if kind in ("zip", "tar"):
                archives.append(f)
            elif kind == "video":
                videos.append(f)
            elif kind == "image":
                images_by_dir.setdefault(d, []).append(f)
    if archives:  # look inside: a zipped capture next to its own preview video is still a capture
        for a in archives:
            if archive_has_capture(a):
                return Found("zip" if sniff(a) == "zip" else "tar", a, f"an archive holding a depth capture ({a.name})")
    if videos:
        best = max(videos, key=_frames_estimate)
        note = "" if len(videos) == 1 else f" (the longest of {len(videos)} videos found; pass the file to choose another)"
        return Found("video", best, "a video file" + note)
    if images_by_dir:
        d, imgs = max(images_by_dir.items(), key=lambda kv: len(kv[1]))
        if len(imgs) >= MIN_IMAGES:
            return Found("images", d, f"a folder of {len(imgs)} photos, taken in name order", sorted(imgs, key=natural_key))
        if len(imgs) == 1:
            return Found("image", imgs[0], "a single photo: a photo can't show a part move; film a video")
    if archives:
        return Found("zip" if sniff(archives[0]) == "zip" else "tar", archives[0], f"an archive ({archives[0].name})")
    n = sum(len(v) for v in images_by_dir.values())
    if n:
        return Found("unknown", root, f"only {n} photos: an image sequence needs at least {MIN_IMAGES} "
                                      "(a video is better)")
    return Found("unknown", root, "no video, depth capture, photos or archive found in this folder")


def _frames_estimate(video: Path) -> float:
    p = ffprobe(video)
    s = video_stream(p) if p else None
    if not s:
        return 0.0
    n = _int(s.get("nb_frames"))
    if n:
        return float(n)
    d = _float(s.get("duration")) or _float((p.get("format") or {}).get("duration")) or 0.0
    return d * (_rate(s.get("avg_frame_rate")) or 30.0)


def archive_members(archive: Path) -> list[str]:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            return z.namelist()
    with tarfile.open(archive) as t:
        return t.getnames()


def archive_has_capture(archive: Path) -> bool:
    try:
        names = [n for n in archive_members(archive) if not _skip(Path(n))]
    except (OSError, zipfile.BadZipFile, tarfile.TarError):
        return False
    dirs = {}
    for n in names:
        p = Path(n)
        dirs.setdefault(str(p.parent), set()).add(p.name)
        if len(p.parts) >= 2:
            dirs.setdefault(str(p.parent.parent), set()).add(p.parent.name + "/")
    for content in dirs.values():
        if {"rgb.mp4", "odometry.csv", "camera_matrix.csv"} <= content and "depth/" in content:
            return True
        if {"intrinsics.json", "depth.npz"} <= content:
            return True
    return False


# ----------------------------------------------------------------------------- archives

def _inside(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def unpack(archive: Path, dest: Path, log=print) -> Path:
    """Extract a zip or tar safely into dest: no absolute paths, no '..', no links out of dest,
    no macOS resource-fork junk, and a size limit against zip bombs."""
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.infolist() if not _skip(Path(m.filename))]
            total = sum(m.file_size for m in members)
            if total > MAX_UNPACKED:
                raise ValueError(f"{archive.name} would unpack to {total / 2 ** 30:.1f} GB; refusing")
            for m in members:
                name = m.filename.replace("\\", "/")
                if name.startswith("/") or ".." in Path(name).parts:
                    log(f"  skipped unsafe archive path {name!r}")
                    continue
                target = dest / name
                if not _inside(dest, target):
                    log(f"  skipped unsafe archive path {name!r}")
                    continue
                if m.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if (m.external_attr >> 16) & 0o170000 == 0o120000:  # a symlink stored in the zip
                    log(f"  skipped link {name!r}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
    else:
        with tarfile.open(archive) as t:
            members = [m for m in t.getmembers() if not _skip(Path(m.name))]
            total = sum(m.size for m in members)
            if total > MAX_UNPACKED:
                raise ValueError(f"{archive.name} would unpack to {total / 2 ** 30:.1f} GB; refusing")
            t.extractall(dest, members=members, filter="data")  # 'data' blocks unsafe paths and links
    return dest


# ----------------------------------------------------------------------------- video plans

@dataclass
class VideoInfo:
    codec: str
    width: int  # as displayed (rotation applied)
    height: int
    rotation: int  # degrees the player turns the stored frames (0, 90, 180, 270)
    fps: float  # average frame rate
    nominal_fps: float  # the stream's declared rate
    vfr: bool
    frames: int | None
    duration: float
    pix_fmt: str
    bit_depth: int
    transfer: str
    primaries: str
    hdr: str | None  # 'hlg' / 'pq' / None
    dolby_vision: int | None  # profile, when the stream carries Dolby Vision
    has_audio: bool
    container: str
    make: str | None  # e.g. Apple
    model: str | None  # e.g. iPhone 18 Pro Max
    has_location: bool  # GPS in the file's metadata


def video_info(probe: dict) -> VideoInfo:
    s = video_stream(probe)
    if s is None:
        raise ValueError("no video stream")
    fmt = probe.get("format") or {}
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    rot = 0
    dv = None
    for sd in s.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"]))) % 360
        if "dv_profile" in sd:
            dv = _int(sd.get("dv_profile"))
    if not rot and "rotate" in (s.get("tags") or {}):  # older muxers
        rot = int(s["tags"]["rotate"]) % 360
    rot = (360 - rot) % 360  # ffprobe gives the matrix's counter-clockwise angle; store clockwise
    w, h = int(s.get("width") or 0), int(s.get("height") or 0)
    if rot in (90, 270):
        w, h = h, w
    avg, nominal = _rate(s.get("avg_frame_rate")), _rate(s.get("r_frame_rate"))
    duration = _float(s.get("duration")) or _float(fmt.get("duration")) or 0.0
    frames = _int(s.get("nb_frames"))
    if not avg and frames and duration:
        avg = frames / duration
    avg = avg or nominal or 30.0
    nominal = nominal or avg
    vfr = abs(avg - nominal) / nominal > 0.02 if nominal else False
    pix = s.get("pix_fmt") or ""
    depth = 10 if "10" in pix else 12 if "12" in pix else 8
    transfer = s.get("color_transfer") or ""
    hdr = "hlg" if transfer == "arib-std-b67" else "pq" if transfer == "smpte2084" else None
    loc = any("location" in k or k == "location" for k in tags)
    return VideoInfo(
        codec=s.get("codec_name") or "", width=w, height=h, rotation=rot, fps=avg, nominal_fps=nominal, vfr=vfr,
        frames=frames, duration=duration, pix_fmt=pix, bit_depth=depth, transfer=transfer,
        primaries=s.get("color_primaries") or "", hdr=hdr, dolby_vision=dv,
        has_audio=any(x.get("codec_type") == "audio" for x in probe.get("streams", [])),
        container=fmt.get("format_name") or "", make=tags.get("com.apple.quicktime.make") or tags.get("make"),
        model=tags.get("com.apple.quicktime.model") or tags.get("model"), has_location=loc)


@dataclass
class VideoPlan:
    fps: str  # output frame rate (ffmpeg rational, e.g. "30" or "30000/1001")
    fps_value: float
    width: int
    height: int
    scale: bool
    tonemap: str | None  # 'hlg' / 'pq'
    turn: int  # extra clockwise turn of the content (upright fix), degrees
    trim_s: float | None  # keep only the first trim_s seconds
    frames_expected: int
    notes: list[str] = field(default_factory=list)  # what changes, in plain words
    warnings: list[str] = field(default_factory=list)


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def plan_video(info: VideoInfo, turn: int = 0, max_fps: float = MAX_FPS, max_frames: int = MAX_FRAMES,
               max_side: int = MAX_SIDE, min_fps: float = MIN_FPS) -> VideoPlan:
    """How to normalise a plain video (pure: decisions only, see ffmpeg_command)."""
    notes, warnings = [], []
    if info.rotation:
        notes.append(f"stored turned by {info.rotation} deg (rotation metadata): frames are turned upright")
    if turn % 360:
        notes.append(f"content turned {turn % 360} deg clockwise so the world's up is the image's up")
    # Frame rate: a whole divisor of the source rate (60 -> 30, 50 -> 25, 120 -> 30) keeps motion
    # evenly sampled; variable-rate video is resampled at its average rate.
    src = info.fps
    if info.vfr:
        out = Fraction(min(max_fps, max(1.0, round(src)))).limit_denominator(1001)
        notes.append(f"variable frame rate (average {src:.2f} fps, declared {info.nominal_fps:.2f}): "
                     f"resampled to a constant {float(out):.2f} fps")
    else:
        rational = Fraction(src).limit_denominator(1001)
        k = 1 if float(rational) <= max_fps * 1.05 else math.ceil(float(rational) / max_fps - 1e-6)
        out = rational / k
        if k > 1:
            notes.append(f"{float(rational):.2f} fps -> {float(out):.2f} fps (1 frame in {k} kept; "
                         "faster rates only add tracking cost)")
    duration = info.duration or ((info.frames or 0) / src if src else 0.0)
    trim = None
    if duration * float(out) > max_frames:
        lowered = max(min_fps, max_frames / duration)
        if lowered < float(out):
            notes.append(f"{duration:.0f} s is long: frame rate lowered to {lowered:.1f} fps "
                         f"to keep about {max_frames} frames")
            out = Fraction(lowered).limit_denominator(100)
        if duration * float(out) > max_frames * 1.001:
            trim = max_frames / float(out)
            warnings.append(f"only the first {trim:.0f} s of {duration:.0f} s are used; trim the video to the part "
                            "where the object moves")
    w, h = info.width, info.height
    if turn % 180 == 90:
        w, h = h, w
    scale = max(w, h) > max_side
    if scale:
        k = max_side / max(w, h)
        notes.append(f"{w}x{h} -> {_even(w * k)}x{_even(h * k)} px")
        w, h = _even(w * k), _even(h * k)
    elif w % 2 or h % 2:
        scale = True
        w, h = _even(w), _even(h)
    if info.hdr:
        notes.append(f"HDR ({'HLG' if info.hdr == 'hlg' else 'PQ'}, {info.bit_depth}-bit) tone-mapped to SDR BT.709")
        if info.dolby_vision == 5:
            warnings.append("Dolby Vision profile 5 has no standard base layer: colours may be off; "
                            "export the video as SDR (H.264, 'Most Compatible') if masks look wrong")
    elif info.bit_depth > 8:
        notes.append(f"{info.bit_depth}-bit -> 8-bit")
    if info.codec not in ("h264",):
        notes.append(f"{info.codec or 'unknown codec'} -> H.264")
    if info.has_audio:
        notes.append("audio dropped")
    if info.has_location:
        notes.append("metadata dropped (the file carries a GPS location)")
    short = min(w, h)
    if short < 480:
        warnings.append(f"low resolution ({w}x{h}): film at 1080p for clean masks")
    frames = int(round(min(duration, trim or duration) * float(out)))
    return VideoPlan(fps=str(out.numerator if out.denominator == 1 else f"{out.numerator}/{out.denominator}"),
                     fps_value=float(out), width=w, height=h, scale=scale, tonemap=info.hdr, turn=turn % 360,
                     trim_s=trim, frames_expected=frames, notes=notes, warnings=warnings)


TURN_FILTERS = {90: ["transpose=1"], 180: ["hflip", "vflip"], 270: ["transpose=2"]}


def ffmpeg_command(src: Path, dst: Path, plan: VideoPlan) -> list[str]:
    """The ffmpeg call for a plan. ffmpeg turns frames by their rotation metadata itself before
    these filters (and writes none to the output)."""
    vf = []
    if plan.tonemap:
        tin = "arib-std-b67" if plan.tonemap == "hlg" else "smpte2084"
        vf += [f"zscale=tin={tin}:pin=bt2020:min=bt2020nc:t=linear:npl=100", "format=gbrpf32le",
               "zscale=p=bt709", "tonemap=tonemap=mobius:desat=0", "zscale=t=bt709:m=bt709:r=tv", "format=yuv420p"]
    vf += TURN_FILTERS.get(plan.turn, [])
    if plan.scale:
        vf.append(f"scale={plan.width}:{plan.height}:flags=area")
    vf += [f"fps={plan.fps}", "format=yuv420p"]
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-map_metadata", "-1",
           "-map_chapters", "-1", "-an", "-sn", "-dn", "-vf", ",".join(vf), "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if plan.tonemap:
        cmd += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]
    if plan.trim_s:
        cmd += ["-t", f"{plan.trim_s:.3f}"]
    return cmd + [str(dst)]


def count_frames(video: Path) -> int:
    """Frames by decoding (headers are off by one on variable-rate phone video)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def normalise_video(src: Path, dst: Path, turn: int = 0, log=print, **limits) -> dict:
    probe = ffprobe(src)
    if not probe or video_stream(probe) is None:
        raise ValueError(f"{src.name}: no decodable video stream")
    info = video_info(probe)
    plan = plan_video(info, turn=turn, **limits)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    run = subprocess.run(ffmpeg_command(src, tmp, plan), capture_output=True, text=True)
    if run.returncode:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"ffmpeg could not convert {src.name}: {run.stderr.strip()[-600:]}")
    tmp.replace(dst)
    out = video_info(ffprobe(dst))
    n = count_frames(dst)
    for line in plan.notes:
        log(f"  {line}")
    for line in plan.warnings:
        log(f"  WARNING: {line}")
    return {"source": asdict(info), "plan": asdict(plan), "output": {**asdict(out), "frames": n}}


def images_to_video(images: list[Path], dst: Path, fps: float = IMAGE_FPS, log=print) -> dict:
    """A photo sequence as a video: EXIF orientation applied, every frame at the first one's size."""
    import cv2
    import numpy as np
    frames, skipped, size = 0, [], None
    with tempfile.TemporaryDirectory() as tmp:
        for f in images:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)  # applies EXIF orientation
            if img is None and image_magic(_head(f)) == "heic" and shutil.which("sips"):
                conv = Path(tmp) / "heic.jpg"
                subprocess.run(["sips", "-s", "format", "jpeg", str(f), "--out", str(conv)], capture_output=True)
                img = cv2.imread(str(conv), cv2.IMREAD_COLOR)
            if img is None:
                skipped.append(f.name)
                continue
            h, w = img.shape[:2]
            if size is None:
                k = min(1.0, MAX_SIDE / max(w, h))
                size = (_even(w * k), _even(h * k))
            elif abs(w / h - size[0] / size[1]) > 0.02 * size[0] / size[1]:
                skipped.append(f.name)  # turned or cropped differently: can't be a frame of the same shot
                continue
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(Path(tmp) / f"{frames:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames += 1
        if frames < MIN_FRAMES:
            raise ValueError(f"only {frames} readable photos of the same shape (skipped {len(skipped)})")
        dst.parent.mkdir(parents=True, exist_ok=True)
        run = subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", f"{fps:g}", "-i", str(Path(tmp) / "%06d.jpg"),
                              "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p",
                              "-movflags", "+faststart", str(dst)], capture_output=True, text=True)
        if run.returncode:
            raise ValueError(f"ffmpeg could not join the photos: {run.stderr.strip()[-400:]}")
    if skipped:
        log(f"  WARNING: skipped {len(skipped)} photos that couldn't be read or have another shape: {skipped[:5]}")
    log(f"  {frames} photos -> {size[0]}x{size[1]} video at {fps:g} fps (photo timing is unknown)")
    return {"photos": len(images), "frames": frames, "skipped": skipped, "fps": fps, "size": list(size),
            "output": {**asdict(video_info(ffprobe(dst))), "frames": count_frames(dst)}}


# ----------------------------------------------------------------------------- depth captures

def frame_files(folder: Path) -> list[Path]:
    """A capture's per-frame files in name order: PNG, or .npy as older Stray Scanner versions wrote."""
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.png")) or sorted(folder.glob("*.npy"))


def capture_counts(root: Path) -> dict:
    """Frame counts of a Stray-layout capture's parts (they must line up frame for frame)."""
    with open(root / "odometry.csv") as fh:
        rows = sum(1 for line in fh if line.strip()) - 1
    return {"depth": len(frame_files(root / "depth")), "confidence": len(frame_files(root / "confidence")),
            "odometry": rows}


def _write_frame(src: Path, dst: Path, depth: bool) -> None:
    """A depth (uint16 mm) or confidence (uint8, 0-2) frame as the PNG the pipeline reads; .npy
    frames are converted (float depth is taken as metres)."""
    import cv2
    import numpy as np
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst)
        return
    a = np.asarray(np.load(src))
    if depth:
        if a.dtype.kind == "f":
            a = np.round(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0)
        a = np.clip(a, 0, 65535).astype(np.uint16)
    else:
        a = np.clip(a, 0, 255).astype(np.uint8)
    cv2.imwrite(str(dst), a)


def plan_capture(frames: int, max_frames: int = MAX_CAPTURE_FRAMES) -> int:
    """Keep every k-th frame of a depth capture (video, depth, confidence and poses together)."""
    return max(1, math.ceil(frames / max_frames)) if frames > max_frames else 1


def check_capture(root: Path, video_frames: int, rotation: int) -> list[str]:
    """Warnings about a Stray-layout capture whose parts don't line up."""
    c = capture_counts(root)
    out = []
    if c["depth"] == 0:
        out.append("the capture has no depth frames")
    elif abs(c["depth"] - video_frames) > 2:
        out.append(f"rgb.mp4 has {video_frames} frames but depth/ has {c['depth']}: only the first "
                   f"{min(c['depth'], video_frames)} are used, and if frames were dropped from one of them "
                   "the depth may not belong to its video frame")
    if c["confidence"] and c["confidence"] != c["depth"]:
        out.append(f"confidence/ has {c['confidence']} frames, depth/ {c['depth']}")
    if c["odometry"] < min(c["depth"], video_frames):
        out.append(f"odometry.csv has {c['odometry']} poses for {min(c['depth'], video_frames)} frames")
    if rotation:
        out.append(f"rgb.mp4 carries a {rotation} deg rotation tag: the video is shown turned but depth isn't; "
                   "frames and depth may not line up")
    return out


def copy_stray(src: Path, dst: Path, keep_every: int = 1, log=print) -> dict:
    """The capture's files into dst (flat layout). With keep_every > 1, every k-th frame of the
    video, depth (with its confidence) and odometry, renumbered together."""
    import cv2
    import numpy as np
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("camera_matrix.csv", "imu.csv", "mapanything.json"):
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    (dst / "depth").mkdir(exist_ok=True)
    (dst / "confidence").mkdir(exist_ok=True)
    conf_dir = src / "confidence"
    synthesized = 0
    for i, f in enumerate(frame_files(src / "depth")[::keep_every]):
        name = f"{i:06d}.png" if keep_every > 1 else f.stem + ".png"  # as recorded unless frames are dropped
        _write_frame(f, dst / "depth" / name, depth=True)
        conf = next((conf_dir / (f.stem + ext) for ext in (".png", ".npy") if (conf_dir / (f.stem + ext)).is_file()),
                    None)  # stray.load pairs a confidence with its depth frame by name
        if conf is not None:
            _write_frame(conf, dst / "confidence" / name, depth=False)
        else:
            d = cv2.imread(str(dst / "depth" / name), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(str(dst / "confidence" / name), np.where(d > 0, 2, 0).astype(np.uint8))
            synthesized += 1
    if synthesized:
        log(f"  WARNING: {synthesized} depth frames have no confidence map: all their depth is trusted")
    if keep_every == 1:  # as recorded: rows and video untouched
        shutil.copy2(src / "odometry.csv", dst / "odometry.csv")
        shutil.copy2(src / "rgb.mp4", dst / "rgb.mp4")
        return capture_counts(dst)
    with open(src / "odometry.csv") as fh:
        lines = [line for line in fh.read().splitlines() if line.strip()]
    out = [lines[0]]
    for i, row in enumerate(lines[1:][::keep_every]):
        cells = row.split(",")
        if len(cells) > 1 and cells[1].strip().isdigit():
            cells[1] = f" {i:06d}"
        out.append(",".join(cells))
    (dst / "odometry.csv").write_text("\n".join(out) + "\n")
    rate = Fraction(video_info(ffprobe(src / "rgb.mp4")).fps).limit_denominator(1001) / keep_every
    rate_s = f"{rate.numerator}/{rate.denominator}"
    # Select by frame index (not time): the n-th kept video frame is the n-th kept depth frame. The
    # kept frames get evenly spaced timestamps at exactly the output rate, so CFR adds or drops none.
    run = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src / "rgb.mp4"), "-map", "0:v:0", "-an",
                          "-vf", f"select=not(mod(n\\,{keep_every})),setpts=N/({rate_s}*TB)",
                          "-fps_mode", "cfr", "-r", rate_s, "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
                          "-pix_fmt", "yuv420p", str(dst / "rgb.mp4")],
                         capture_output=True, text=True)
    if run.returncode:
        raise ValueError(f"ffmpeg could not decimate rgb.mp4: {run.stderr.strip()[-400:]}")
    log(f"  kept 1 frame in {keep_every} of video, depth, confidence and poses together")
    return capture_counts(dst)


def copy_rgbd(src: Path, dst: Path) -> Path:
    """A fixed-camera RGB-D folder (depth.npz, intrinsics.json, its video) into dst. Returns the video."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("depth.npz", "intrinsics.json"):
        shutil.copy2(src / name, dst / name)
    videos = [f for f in _files(src) if sniff(f) == "video"]
    video = max(videos, key=_frames_estimate)
    shutil.copy2(video, dst / video.name)
    return dst / video.name


# ----------------------------------------------------------------------------- top level

@dataclass
class Ingested:
    kind: str  # stray / mapanything / rgbd / video / images
    source: str  # what the user gave
    found: str  # what was used inside it
    video: str  # the video the pipeline reads (frames 1:1 with depth for captures)
    capture: str | None  # the capture folder in the run (Stray layout or RGB-D), if any
    frames: int
    fps: float
    size: tuple[int, int]
    details: dict
    notes: list[str]
    warnings: list[str]
    usable: bool = True
    reason: str = ""

    @property
    def has_depth(self) -> bool:
        return self.kind in DEPTH_KINDS


def ingest(source: Path, input_dir: Path, name: str, turn: int = 0, image_fps: float = IMAGE_FPS,
           max_capture_frames: int = MAX_CAPTURE_FRAMES, log=print, **limits) -> Ingested:
    """Detect what ``source`` is and write the pipeline's inputs into ``input_dir``."""
    source = Path(source)
    input_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    found = detect(source)
    depth = 0
    while found.kind in ("zip", "tar"):  # unpack and look again (an archive may hold another)
        depth += 1
        if depth > 3:
            return _unusable(source, found, "archives nested more than 3 deep")
        dest = input_dir / f"unpacked{depth if depth > 1 else ''}"
        log(f"unpacking {found.path.name} ...")
        try:
            unpack(found.path, dest, log=log)
        except (ValueError, OSError, zipfile.BadZipFile, tarfile.TarError) as error:
            return _unusable(source, found, f"could not unpack {found.path.name}: {error}")
        notes.append(f"unpacked {found.path.name}")
        archive = found.path
        found = detect(dest)
        found.archive = archive
    log(f"input: {found.reason} ({found.path})")
    if found.kind in ("image", "unknown"):
        return _unusable(source, found, found.reason)
    video = input_dir / f"{name}.mp4"
    if found.kind == "video":
        try:
            details = normalise_video(found.path, video, turn=turn, log=log, **limits)
        except ValueError as error:
            return _unusable(source, found, str(error))
        out = details["output"]
        return Ingested("video", str(source), str(found.path), str(video), None, out["frames"], out["fps"],
                        (out["width"], out["height"]), details, notes + details["plan"]["notes"],
                        list(details["plan"]["warnings"]))
    if found.kind == "images":
        try:
            details = images_to_video(found.images, video, fps=image_fps, log=log)
        except ValueError as error:
            return _unusable(source, found, str(error))
        if turn % 360:  # an upright fix: turn the joined video (no other change)
            details = {**details, "upright": normalise_video(video, video.with_name(video.stem + ".upright.mp4"),
                                                               turn=turn, log=log, **limits)}
            video.with_name(video.stem + ".upright.mp4").replace(video)
        out = details["output"] if not turn % 360 else details["upright"]["output"]
        warns = [f"photos joined at {image_fps:g} fps: their real timing is unknown, so physics and speeds are "
                 "approximate"]
        return Ingested("images", str(source), str(found.path), str(video), None, out["frames"], out["fps"],
                        (out["width"], out["height"]), details, notes + [f"{details['frames']} photos joined"], warns)
    # depth captures: frames stay aligned with their depth and poses
    capture = input_dir / "capture"
    if found.kind == "rgbd":
        if capture.exists():
            shutil.rmtree(capture)
        src_video = copy_rgbd(found.path, capture)
        shutil.copy2(src_video, video)
        info = video_info(ffprobe(video))
        n = count_frames(video)
        warns = []
        import numpy as np
        with np.load(capture / "depth.npz") as d:
            nd = len(d["depth"])
        if abs(nd - n) > 2:
            warns.append(f"the video has {n} frames but depth.npz {nd}: only the first {min(n, nd)} are used")
        if info.rotation:
            warns.append(f"the video carries a {info.rotation} deg rotation tag that its depth doesn't have")
        return Ingested("rgbd", str(source), str(found.path), str(video), str(capture), n, info.fps,
                        (info.width, info.height), {"video": asdict(info), "depth_frames": nd}, notes, warns)
    src = found.path
    probe = ffprobe(src / "rgb.mp4")
    if not probe or video_stream(probe) is None:
        return _unusable(source, found, "the capture's rgb.mp4 has no decodable video")
    info = video_info(probe)
    n = count_frames(src / "rgb.mp4")
    warns = check_capture(src, n, info.rotation)
    if capture_counts(src)["depth"] == 0:
        return _unusable(source, found, "the capture has no depth frames")
    keep = plan_capture(min(n, capture_counts(src)["depth"]), max_capture_frames)
    if keep > 1:
        notes.append(f"{n} frames is long: kept 1 frame in {keep} of video, depth and poses together")
    if capture.exists():
        shutil.rmtree(capture)
    where = str(src)
    if found.archive is not None and _inside(input_dir, src):
        where = f"{found.archive} -> {src.relative_to(input_dir)}"
    c0 = capture_counts(src)
    ready = (frame_files(src / "depth")[0].suffix == ".png" and c0["confidence"] == c0["depth"]
             and all(f.suffix == ".png" for f in frame_files(src / "confidence")[:1]))
    if found.archive is not None and _inside(input_dir, src) and keep == 1 and ready:
        src.rename(capture)  # unpacked by us and already in the layout the pipeline reads: move it
        counts = capture_counts(capture)
    else:
        counts = copy_stray(src, capture, keep_every=keep, log=log)
    link_or_copy(capture / "rgb.mp4", video)
    nv = count_frames(video) if keep > 1 else n
    if keep > 1:
        warns = check_capture(capture, nv, 0)  # the decimated copy must still line up
    kind = found.kind
    log(f"  capture: {nv} video frames, {counts['depth']} depth frames, {counts['odometry']} poses, "
        f"{info.width}x{info.height} at {info.fps / keep:.1f} fps ({info.codec})")
    return Ingested(kind, str(source), where, str(video), str(capture), nv, info.fps / keep,
                    (info.width, info.height), {"video": asdict(info), "counts": counts, "keep_every": keep},
                    notes, warns)


def link_or_copy(src: Path, dst: Path) -> None:
    import os
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _unusable(source: Path, found: Found, reason: str) -> Ingested:
    return Ingested(found.kind, str(source), str(found.path), "", None, 0, 0.0, (0, 0), {}, [], [], False, reason)
