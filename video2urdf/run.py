"""Whatever was filmed -> a checked URDF/MJCF, a viewer folder and a summary, in one call.

Steps (each timed and logged; a failing later step never discards the model):
 1. ingest       detect the input from its content, normalise it (video2urdf/ingest.py)
 2. check        cheap local capture check: length, rate, size, light, camera motion, motion
 3. look         one Muse Spark call on a few frames: object, view, which way is up, problems
                 (a sideways plain video is turned upright here)
 4. route        3D (depth, or MapAnything for a moving/oblique camera) or planar (route.py)
 5. depth        plain video on the 3D path: MapAnything on Modal + automatic scale
 6. agent        Muse Spark builds the model (video2urdf/agent.py)
 7. fallback     no model -> a rigid model of the whole object (fallback.py), needs_review
 8. physics      if the agent didn't set it (scripts/add_physics.py)
 9. texture      3D models: the video painted on the measured meshes (scripts/texture_model.py)
10. design       Spark's clean visual, kept only if it matches the video (scripts/design_shape.py)
11. completion   the unseen back of a partly filmed body, as estimates (scripts/complete_body.py,
                 when present)
12. robot        optional: a Panda uses the model in MuJoCo (scripts/robot_tasks.py)
13. viewer       a standalone web viewer folder (docs/index.html + scripts/export_viewer.py)
Texture runs before design because design moves the measured meshes into the hidden collision
group, where texturing no longer sees them; completion runs after design so it can use the
object's identity, and last, so nothing later overwrites it.

Everything lands in <out>/<name>/: input/ (normalised media), model/ (the model and the
agent's log), viewer/, logs/, summary.json and summary.md.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import ingest as ING
from .capture_check import assess, check
from .route import choose_path, spark_look, upright_turn

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
STATUSES = ("checks_passed", "needs_review", "failed")


@dataclass
class Options:
    out: Path = ROOT / "data" / "runs"
    name: str | None = None
    path: str = "auto"  # auto / 3d / planar
    rotate: str = "auto"  # auto (Spark's look) / 0 / 90 / 180 / 270 (clockwise)
    look: bool = True
    open_weights: bool = False
    glimmer: bool = False
    sam: str | None = None  # api / open
    design: bool = True
    texture: bool = True
    robot: bool = False
    viewer_out: Path | None = None
    check_only: bool = False
    focal_35mm: float | None = None
    scale: float | None = None
    auto_scale: bool = True
    max_fps: float = ING.MAX_FPS
    max_frames: int = ING.MAX_FRAMES
    max_side: int = ING.MAX_SIDE
    image_fps: float = ING.IMAGE_FPS


@dataclass
class Step:
    name: str
    status: str  # ok / skipped / failed
    seconds: float
    note: str = ""


def slug(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()[:60]


def unique_dir(base: Path) -> Path:
    """base, or base_2, base_3 ... : a run never overwrites an earlier one."""
    if not base.exists() or not any(base.iterdir()):
        return base
    k = 2
    while (base.parent / f"{base.name}_{k}").exists():
        k += 1
    return base.parent / f"{base.name}_{k}"


def backend_env(opt: Options) -> dict[str, str]:
    """Environment for the agent and every later step (read by video2urdf.agent on import)."""
    env = {}
    if opt.open_weights:
        sys.path.insert(0, str(SCRIPTS))
        from run_agent import OPEN_WEIGHTS
        env.update(OPEN_WEIGHTS)
    if opt.glimmer:
        env.update({"GLIMMER_API_KEY": f"{os.environ['MODAL_PROXY_TOKEN_ID']}.{os.environ['MODAL_PROXY_TOKEN_SECRET']}",
                    "AGENT_MODEL": "glimmer", "AGENT_BASE_URL": os.environ["GLIMMER_URL"].rstrip("/") + "/v1",
                    "AGENT_API_KEY_ENV": "GLIMMER_API_KEY", "AGENT_MAX_IMAGES": "8", "AGENT_VIDEO": "0"})
    if opt.sam:
        env["SAM_BACKEND"] = opt.sam
    return env


class Run:
    def __init__(self, source: Path, opt: Options):
        self.source, self.opt = Path(source), opt
        self.name = slug(opt.name or self.source.stem or self.source.name) or "capture"
        self.dir = unique_dir(opt.out / self.name)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        self.model_dir = self.dir / "model"
        self.viewer_dir = opt.viewer_out or self.dir / "viewer"
        self.steps: list[Step] = []
        self.summary: dict = {"name": self.name, "source": str(self.source), "run_dir": str(self.dir),
                              "status": "failed", "issues": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.env = backend_env(opt)
        os.environ.update(self.env)
        self.t0 = time.time()

    # ------------------------------------------------------------------ plumbing
    def log(self, text: str):
        print(text, flush=True)
        with open(self.dir / "pipeline.log", "a") as fh:
            fh.write(text + "\n")

    def step(self, name: str, fn, *args, **kw):
        """Run one step: time it, log it, and turn an exception into a failed step."""
        self.log(f"\n== {name}")
        t = time.time()
        try:
            result = fn(*args, **kw)
        except Exception as error:
            self.log(f"   {name} failed: {type(error).__name__}: {error}")
            with open(self.dir / "logs" / f"{name}.error.txt", "w") as fh:
                fh.write(traceback.format_exc())
            self.steps.append(Step(name, "failed", round(time.time() - t, 1), f"{type(error).__name__}: {error}"[:300]))
            return None
        note = result.pop("_note", "") if isinstance(result, dict) else ""
        self.steps.append(Step(name, "ok", round(time.time() - t, 1), note))
        return result

    def skip(self, name: str, why: str):
        self.log(f"\n== {name}: skipped ({why})")
        self.steps.append(Step(name, "skipped", 0.0, why))

    def script(self, name: str, args: list[str]) -> dict:
        """A repo script as a subprocess (own memory, same backend settings); output to logs/."""
        log_path = self.dir / "logs" / f"{name}.log"
        with open(log_path, "w") as fh:
            r = subprocess.run([sys.executable, *args], cwd=ROOT, env={**os.environ, **self.env},
                               stdout=fh, stderr=subprocess.STDOUT)
        tail = log_path.read_text(errors="replace").strip().splitlines()[-6:]
        for line in tail:
            self.log(f"   | {line[:220]}")
        if r.returncode:
            raise RuntimeError(f"{Path(args[0]).name} exited {r.returncode} (see {log_path.relative_to(self.dir)})")
        return {"_note": tail[-1][:200] if tail else ""}

    # ------------------------------------------------------------------ the run
    def go(self) -> dict:
        try:
            self._go()
        except KeyboardInterrupt:
            self.summary["issues"].append("interrupted")
            self.summary["status"] = "failed"
            raise
        except Exception as error:  # a bug in a step's glue: still write what we know
            self.summary["issues"].append(f"pipeline error: {type(error).__name__}: {error}")
            with open(self.dir / "logs" / "pipeline.error.txt", "w") as fh:
                fh.write(traceback.format_exc())
            self.log(traceback.format_exc())
        finally:
            self.finish()
        return self.summary

    def _go(self):
        opt = self.opt
        if opt.viewer_out and opt.viewer_out.exists() and any(opt.viewer_out.iterdir()):
            raise ValueError(f"--viewer-out {opt.viewer_out} is not empty")
        self.log(f"Video2URDF: {self.source} -> {self.dir}")
        limits = dict(max_fps=opt.max_fps, max_frames=opt.max_frames, max_side=opt.max_side)
        turn = 0 if opt.rotate == "auto" else int(opt.rotate) % 360
        ing = self.step("ingest", ING.ingest, self.source, self.dir / "input", self.name, turn=turn,
                        image_fps=opt.image_fps, log=self.log, **limits)
        if ing is None or not ing.usable:
            self.summary["input"] = asdict(ing) if ing else None
            self.summary["issues"].append(f"unusable input: {ing.reason if ing else 'ingest failed'}")
            return
        self.summary["input"] = self._input_facts(ing)

        metrics = self.step("check", check, Path(ing.video), ing.frames, ing.fps, ing.size, ing.has_depth,
                            Path(ing.capture) if ing.capture else None)
        photos = ing.kind == "images"
        verdict = (assess(metrics, ing.has_depth, assumed_fps=photos) if metrics else
                   {"usable": True, "warnings": [], "tips": []})
        if not verdict["usable"]:
            self.summary["check"] = {"metrics": metrics, **verdict}
            self.summary["issues"].append(f"unusable capture: {verdict['reason']}")
            self.log(f"   unusable: {verdict['reason']}")
            return

        look = None
        if opt.look:
            look = self.step("look", self._look, ing)
        else:
            self.skip("look", "--no-look")
        if look and not ing.has_depth and opt.rotate == "auto" and upright_turn(look):
            k = upright_turn(look)
            self.log(f"   Spark sees the world's up at the image's {look['up']}: turning the video {k} deg clockwise")
            again = self.step("upright", ING.ingest, self.source, self.dir / "input", self.name, turn=k,
                              image_fps=opt.image_fps, log=self.log, **limits)
            if again is not None and again.usable:
                ing = again
                self.summary["input"] = self._input_facts(ing)
                self.summary["input"]["turned_upright_deg"] = k
                metrics = check(Path(ing.video), ing.frames, ing.fps, ing.size, ing.has_depth, None) or metrics
        verdict = assess(metrics, ing.has_depth, look, assumed_fps=photos) if metrics else verdict
        self.summary["check"] = {"metrics": metrics, **verdict}
        self.summary["look"] = look
        for w in verdict["warnings"]:
            self.log(f"   WARNING: {w}")
        for tip in verdict["tips"]:
            self.log(f"   tip: {tip}")

        forced = None if opt.path == "auto" else opt.path
        path, reasons = choose_path(ing.kind, (metrics or {}).get("camera"), look, forced)
        self.summary["path"] = {"chosen": path, "reasons": reasons, "forced": forced is not None}
        self.log(f"\n== route: {path}\n   " + "\n   ".join(reasons))
        if opt.check_only:
            self.summary["status"] = "checked"
            return

        video = Path(ing.video)
        capture = Path(ing.capture) if ing.capture and path == "3d" else None
        if path == "3d" and not ing.has_depth:
            capture = self.step("depth", self._pseudo_depth, video)
            if capture is None:
                self.summary["issues"].append("MapAnything depth failed: trying the planar path instead")
                path = "planar"
                self.summary["path"].update(chosen="planar", fell_back_from="3d")
        else:
            self.skip("depth", "the capture has its own depth" if ing.has_depth else "planar path")
        self.summary["path"]["capture"] = str(capture) if capture else None

        agent = self.step("agent", self._agent, video, capture)
        if agent is not None:
            report = agent.get("report") or ""
            self.summary["agent"] = {"status": agent.get("status"), "error": agent.get("error"),
                                     "model_current": agent.get("last_fit"), "scale": agent.get("scale_note"),
                                     "report_head": report.strip().split("\n\n", 1)[-1][:600] if report else None}
        model_ok = (self.model_dir / "model.xml").exists() and (self.model_dir / "model.urdf").exists()
        crashed = agent is None or agent.get("error")
        if crashed and model_ok:  # keep the agent's model, flagged: it may predate its last re-track
            self.summary["issues"].append(f"the agent stopped with an error after building a model: "
                                          f"{(agent or {}).get('error') or 'see logs/agent.error.txt'}")
        if not model_ok:
            why = ("the agent stopped with an error" if crashed else "the agent built no model")
            fb = self.step("fallback", self._fallback, video, capture, look, agent, why)
            model_ok = bool(fb and fb.get("ok"))
            self.summary["fallback"] = {"used": True, "why": why, **(fb or {"ok": False})}
        else:
            self.skip("fallback", "the agent built a model")
            self.summary["fallback"] = {"used": False}
        if not model_ok:
            self.summary["issues"].append("no model could be built")
            return

        if self._has_physics():
            self.skip("physics", "set by the agent")
        else:
            self.step("physics", self.script, "physics",
                      [str(SCRIPTS / "add_physics.py"), str(self.model_dir), "--video", str(video)])
        if not opt.texture:
            self.skip("texture", "--no-texture")
        elif (self.model_dir / "replay3d.npz").exists():
            self.step("texture", self.script, "texture",
                      [str(SCRIPTS / "texture_model.py"), str(self.model_dir), str(video)])
        else:
            self.skip("texture", "planar model: no camera poses to project the video with")
        if opt.design:
            flags = ["--open-weights"] if opt.open_weights else []
            flags += ["--glimmer"] if opt.glimmer else []
            self.step("design", self.script, "design",
                      [str(SCRIPTS / "design_shape.py"), str(self.model_dir), str(video), *flags])
        else:
            self.skip("design", "--no-design")
        completion = SCRIPTS / "complete_body.py"
        if completion.exists():
            self.step("completion", self.script, "completion", [str(completion), str(self.model_dir), str(video)])
        else:
            self.skip("completion", "scripts/complete_body.py isn't in this checkout")
        if opt.robot:
            self.step("robot", self.script, "robot", [str(SCRIPTS / "robot_tasks.py"), str(self.model_dir),
                                                      "--out", str(self.model_dir / "robot_demo.mp4")])
            self.summary["robot"] = robot_results(self.dir / "logs" / "robot.log")
        else:
            self.skip("robot", "add --robot for a simulated robot using the model")
        self.step("viewer", self._viewer, video)

    # ------------------------------------------------------------------ steps
    def _input_facts(self, ing) -> dict:
        d = asdict(ing)
        d["size"] = list(ing.size)
        return d

    def _look(self, ing) -> dict:
        look = spark_look(Path(ing.video), ing.frames, ing.fps, log=self.log)
        if not look:  # recorded as a failed step; routing goes on with the measured numbers alone
            raise RuntimeError("no usable answer from Spark")
        self.log(f"   {look.get('object')!r}: moving parts {look.get('moving_parts')}, view {look.get('view')}, "
                 f"camera {look.get('camera')}, up at the {look.get('up')} ({look.get('up_confidence')})")
        return {**look, "_note": f"{look.get('object')}, {look.get('view')} view, up at the {look.get('up')}"}

    def _pseudo_depth(self, video: Path) -> Path:
        """Plain video -> a capture in the Stray layout with MapAnything depth, at an automatic scale."""
        from . import pseudo_depth
        capture = self.dir / "input" / "capture"
        focal = None
        if self.opt.focal_35mm:
            info = ING.video_info(ING.ffprobe(video))
            focal = pseudo_depth.focal_from_35mm(self.opt.focal_35mm, (info.width, info.height))
        self.log("   MapAnything on Modal estimates depth and camera motion (metric scale not guaranteed)")
        rec = pseudo_depth.reconstruct(video, focal=focal, scale=self.opt.scale, log=lambda s: self.log("   " + s))
        pseudo_depth.write_capture(rec, video, capture)
        meta = {"source": str(video), "frames": len(rec.depth), "depth_size": list(rec.depth.shape[2:0:-1]),
                "focal_px": float(rec.k[0, 0]), "focal_given": focal is not None,
                "scale_applied": self.opt.scale or 1.0, "chunk_scales": rec.chunk_scales}
        (capture / "mapanything.json").write_text(json.dumps(meta, indent=2))
        if self.opt.scale is None and self.opt.auto_scale:
            from .autoscale import auto_scale
            found = auto_scale(capture, capture / "rgb.mp4", log=lambda s: self.log("   " + s))
            if found is None:
                self.log("   no automatic scale: MapAnything's own scale is kept (it can be several times off)")
            else:
                pseudo_depth.rescale_capture(capture, found["scale_factor"])
                meta["scale_applied"] = meta["scale_applied"] * found["scale_factor"]
                meta["auto_scale"] = found
                (capture / "mapanything.json").write_text(json.dumps(meta, indent=2))
                self.log(f"   automatic scale ({found['method']}): x{found['scale_factor']:.3f}")
        return capture

    def _agent(self, video: Path, capture: Path | None) -> dict:
        from .agent import Agent  # reads the backend settings on import
        sys.path.insert(0, str(SCRIPTS))
        from run_agent import write_measurement_source
        self.model_dir.mkdir(parents=True, exist_ok=True)
        agent = Agent(video, self.model_dir, capture=capture)
        write_measurement_source(self.model_dir, capture)
        result = {"report": None, "error": None}
        try:
            result["report"] = agent.run()
        except Exception as error:  # keep what it built; the fallback decides what to do
            result["error"] = f"{type(error).__name__}: {error}"
            with open(self.dir / "logs" / "agent.error.txt", "w") as fh:
                fh.write(traceback.format_exc())
            self.log(f"   the agent stopped: {result['error']}")
        result["last_fit"] = agent.last_fit is not None
        result["scale_note"] = agent.scale_note
        result["px_per_m"] = agent.px_per_m
        status_path = self.model_dir / "agent" / "status.json"
        result["status"] = json.loads(status_path.read_text()) if status_path.exists() else None
        if capture is None:  # the planar path (also describes a fallback model, which uses the same scale)
            card = "from card" in (agent.scale_note or "")
            source = {"depth": "Planar side-view video (no depth)",
                      "metric": "measured from an ID-1 card in view" if card else "assumed, not measured",
                      "note": ("Joint motion comes from the parts' 2D motion; sizes in the motion plane use the "
                               f"scale {agent.scale_note}; depth across the plane is assumed.")}
            (self.model_dir / "measurement_source.json").write_text(json.dumps(source, indent=2) + "\n")
        result["_note"] = (result["status"] or {}).get("status", "no status") + (
            f"; error: {result['error']}" if result["error"] else "")
        return result

    def _fallback(self, video: Path, capture: Path | None, look: dict | None, agent: dict | None, why: str) -> dict:
        from .fallback import rigid_model
        self.log(f"   {why}: building a rigid model of the whole object")
        prompt = (look or {}).get("object") or "object"
        result = rigid_model(video, self.model_dir, capture, prompt, px_per_m=(agent or {}).get("px_per_m"),
                             log=lambda s: self.log("   " + s))
        if result.get("ok"):
            self.log(f"   rigid model: {result['how']} ({result.get('described')})")
            result["_note"] = result["how"]
        else:
            self.log(f"   rigid model failed: {result.get('error')}")
            result["_note"] = result.get("error", "")[:200]
        return result

    def _has_physics(self) -> bool:
        try:
            return ET.parse(self.model_dir / "model.xml").getroot().find(".//inertial") is not None
        except (ET.ParseError, OSError):
            return False

    def _viewer(self, video: Path) -> dict:
        out = self.viewer_dir
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "docs" / "index.html", out / "index.html")
        self.script("viewer", [str(SCRIPTS / "export_viewer.py"), f"{self.name}:{self.model_dir}:{video}",
                               "--out", str(out)])
        return {"_note": str(out)}

    # ------------------------------------------------------------------ summary
    def finish(self):
        from .summary import model_facts, write_summary
        s = self.summary
        s["steps"] = [asdict(x) for x in self.steps]
        s["seconds"] = round(time.time() - self.t0, 1)
        s["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if (self.model_dir / "model.xml").exists() and s.get("status") != "checked":
            try:
                s["model"] = model_facts(self.model_dir)
            except Exception as error:
                s["model"] = {"error": f"{type(error).__name__}: {error}"}
            self._status()
        s["outputs"] = {k: str(v) for k, v in {
            "urdf": self.model_dir / "model.urdf", "mjcf": self.model_dir / "model.xml",
            "agent_report": self.model_dir / "agent" / "report.md",
            "viewer": self.viewer_dir / "index.html", "robot_video": self.model_dir / "robot_demo.mp4",
            "replay": (self.model_dir / "replay3d.mp4") if (self.model_dir / "replay3d.mp4").exists()
            else self.model_dir / "replay.mp4"}.items() if Path(v).exists()}
        write_summary(self.dir, s)
        self.log(f"\n== {s['status'].replace('_', ' ').upper()} in {s['seconds'] / 60:.1f} min: "
                 f"{self.dir / 'summary.md'}")

    def _status(self):
        s = self.summary
        agent_status = ((s.get("agent") or {}).get("status") or {})
        status = agent_status.get("status") or "needs_review"
        issues = list(agent_status.get("issues") or [])
        if not agent_status:
            issues.insert(0, "the agent recorded no status (it stopped before finishing)")
        if (s.get("fallback") or {}).get("used"):
            status = "needs_review"
            issues.insert(0, f"rigid fallback: {s['fallback']['why']}; no joint was measured")
        if (s.get("path") or {}).get("fell_back_from"):
            status = "needs_review"
        failed = [x.name for x in self.steps if x.status == "failed"]
        if failed:
            issues.append(f"steps that failed (the model stands without them): {', '.join(failed)}")
        s["status"] = status if status in STATUSES else "needs_review"
        s["issues"] = issues + [i for i in s.get("issues", []) if i not in issues]


def robot_results(log_path: Path) -> dict | None:
    """What scripts/robot_tasks.py reports per task: the joint (or object) where each task left it,
    relative to the start. Its exit code only says the simulation ran, not that the task worked."""
    if not log_path.exists():
        return None
    tasks: dict[str, str] = {}
    kind = None
    for line in log_path.read_text(errors="replace").splitlines():
        m = re.match(r"\s*after '(.+)': (.+)$", line)
        if m:
            tasks[m.group(1)] = m.group(2).strip()  # the last line per task is where it ended
        elif re.match(r"^(plate|rod|drawer|static) \(", line):
            kind = line.split(";")[0]
    return {"kind": kind, "tasks": [{"task": k, "ended": v} for k, v in tasks.items()]} if tasks else None


def run(source: Path, opt: Options) -> dict:
    r = Run(source, opt)
    return r.go()
