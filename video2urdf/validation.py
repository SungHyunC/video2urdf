"""Do the final downloads load? scripts/check_exports.py run on the model a run hands out, kept as
validation.json next to summary.json, with the model files' hashes so a later change to the model
shows the validation as stale.

The agent's verdict (agent/status.json) is decided before physics, texture, shape design and
completion change the model, and says nothing about other simulators reading the files. This is the
second verdict, on the artifacts themselves:

  artifact status   checks_passed  nothing failed (warnings are listed)
                    needs_review   a check or a format failed (named); the others still load
                    failed         the model's files don't resolve, or no format loads at all
                    not_tested     no loader installed, the checker didn't finish, or the
                                   validation is stale / missing
  overall status    the worse of the agent's and the artifact's (not_tested counts as needs_review)

Per row: pass | warn | fail | not_tested (a loader that isn't installed is not_tested, never pass).
Local only: no network, no keys; MuJoCo is the one loader the project installs, PyBullet, yourdfpy
and Drake are checked when present.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKER = ROOT / "scripts" / "check_exports.py"
TIMEOUT_S = 90
ROW_STATUS = {"pass": "pass", "warn": "warn", "fail": "fail", "skip": "not_tested"}
LOADERS = ("mjcf:mujoco", "urdf:mujoco", "urdf:pybullet", "urdf:yourdfpy", "urdf:drake")
RANK = {"checks_passed": 0, "checked": 0, "needs_review": 1, "not_tested": 1, "failed": 2}


# ------------------------------------------------------------------ hashes

def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(path: Path) -> str | None:
    """A file's sha256; a zip by its members' names, CRCs and sizes (re-zipping the same files
    gives the same digest); a folder by its files' relative paths and hashes. None if missing."""
    if path.is_file() and path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as z:
                members = sorted((i.filename, i.CRC, i.file_size) for i in z.infolist() if not i.is_dir())
        except zipfile.BadZipFile:
            return _sha(path)
        return "zip:" + hashlib.sha256(json.dumps(members).encode()).hexdigest()
    if path.is_file():
        return _sha(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        return "dir:" + hashlib.sha256(json.dumps([[str(p.relative_to(path)), _sha(p)] for p in files]).encode()).hexdigest()
    return None


def fingerprint(base: Path, paths: list[Path]) -> dict[str, str | None]:
    """Relative path (to base) -> digest, for each path that makes up the model."""
    return {_rel(p, base): digest(p) for p in paths}


def _rel(p: Path, base: Path) -> str:
    try:
        return str(p.resolve().relative_to(base.resolve()))
    except ValueError:  # outside base: kept absolute so freshness() finds it again
        return str(p.resolve())


def freshness(base: Path, validation: dict | None = None) -> dict:
    """Is validation.json (in base) still about the model on disk? state current / stale / missing,
    and which files changed since it was written."""
    if validation is None:
        validation = read(base)
    if not validation:
        return {"state": "missing", "changed": []}
    changed = []
    for rel, was in (validation.get("files") or {}).items():
        p = Path(rel) if Path(rel).is_absolute() else base / rel
        now = digest(p)
        if now != was:
            changed.append(rel + (" (gone)" if now is None and was is not None else " (changed)"))
    return {"state": "stale" if changed else "current", "changed": changed}


def read(base: Path) -> dict | None:
    try:
        return json.loads((base / "validation.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ------------------------------------------------------------------ verdicts

def artifact_status(rows: list[dict]) -> tuple[str, list[str]]:
    """The artifact verdict from check rows ({check, status, detail}, status already mapped) and
    one short reason per failing check."""
    if not rows:
        return "not_tested", ["the export checks produced no rows"]
    fails = [r for r in rows if r["status"] == "fail"]
    reasons = [f"{r['check']} failed: {_first(r['detail'])}" for r in fails]
    tested = [r for r in rows if r["check"] in LOADERS and r["status"] != "not_tested"]
    if any(r["check"] == "files" for r in fails):
        return "failed", reasons
    if tested and all(r["status"] == "fail" for r in tested):
        return "failed", reasons + ["no format loads"]
    if not tested:
        return "not_tested", reasons + ["no loader installed (MuJoCo, PyBullet, yourdfpy, Drake)"]
    return ("needs_review" if fails else "checks_passed"), reasons


def _first(detail: str, n: int = 240) -> str:
    d = " ".join((detail or "").split())
    return d if len(d) <= n else d[: n - 3] + "..."


def combine(agent: str | None, artifact: str | None) -> str:
    """Overall status: the worse of the two; anything unknown or untested counts as needs_review."""
    a = agent if agent in RANK else "needs_review"
    b = artifact if artifact in RANK else "needs_review"
    worse = max(a, b, key=lambda s: RANK[s])
    return "needs_review" if worse in ("not_tested", "checked") else worse


def current_artifact_status(base: Path) -> tuple[str, list[str], dict]:
    """The artifact status as it stands now: validation.json's verdict if it is still about the
    model on disk, else not_tested with the reason (missing, or which files changed)."""
    v = read(base)
    fresh = freshness(base, v)
    if fresh["state"] == "missing":
        return "not_tested", ["no validation.json: the downloads were not checked"], fresh
    if fresh["state"] == "stale":
        return "not_tested", ["validation.json is stale: " + ", ".join(fresh["changed"]) +
                              " since it was written (rerun scripts/validate_run.py)"], fresh
    return v.get("artifact_status", "not_tested"), list(v.get("reasons") or []), fresh


# ------------------------------------------------------------------ the check

def run_checker(target: Path, records: Path | None = None, timeout: float = TIMEOUT_S) -> tuple[list[dict], str | None]:
    """scripts/check_exports.py on target (a viewer folder with model.zip, or a folder with
    model.urdf + model.xml) in a subprocess (a crashing loader can't take the pipeline down).
    Rows with statuses mapped, and an error note when it didn't finish."""
    with tempfile.TemporaryDirectory(prefix="validate_") as tmp:
        out = Path(tmp) / "rows.json"
        cmd = [sys.executable, str(CHECKER), str(target), "--json", str(out)]
        if records is not None:
            cmd += ["--run", f"{target.name}={records}"]
        try:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return [], f"check_exports.py took longer than {timeout:g} s"
        if not out.exists():
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or [f"exit {r.returncode}"]
            return [], f"check_exports.py wrote no rows: {tail[0][:200]}"
        rows = json.loads(out.read_text())
    return [{"check": x["check"], "status": ROW_STATUS.get(x["status"], "not_tested"), "detail": x["detail"]}
            for x in rows], None


def validate(target: Path, base: Path, hashed: list[Path], records: Path | None = None,
             timeout: float = TIMEOUT_S) -> dict:
    """Check target, write base/validation.json and return it. ``hashed`` are the files that make
    up the model (their digests mark the validation stale when any of them changes)."""
    t = time.time()
    files = fingerprint(base, hashed)  # before the check: a change during it shows as stale
    rows, error = run_checker(target, records, timeout)
    status, reasons = artifact_status(rows)
    if error:
        reasons.insert(0, error)
    v = {"artifact_status": status, "reasons": reasons, "target": _rel(target, base),
         "records": _rel(records, base) if records else None,
         "validated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t, 1),
         "rows": rows, "files": files,
         "counts": {s: sum(r["status"] == s for r in rows) for s in ("pass", "warn", "fail", "not_tested")}}
    (base / "validation.json").write_text(json.dumps(v, indent=1) + "\n")
    return v


def run_paths(run_dir: Path, name: str, model_dir: Path | None = None, viewer_dir: Path | None = None
              ) -> tuple[Path, list[Path]]:
    """What a pipeline run hands out and what makes it up: the viewer's model folder (its model.zip
    is the main download) when it was exported, else the agent's model folder; hashed are the
    model folder's URDF, MJCF and meshes, and the viewer's downloads."""
    model_dir = model_dir or run_dir / "model"
    viewer = (viewer_dir or run_dir / "viewer") / "models" / name
    hashed = [model_dir / "model.urdf", model_dir / "model.xml", model_dir / "meshes"]
    if (viewer / "model.zip").exists():
        return viewer, hashed + [viewer / "model.zip", viewer / "model.urdf", viewer / "model.xml"]
    return model_dir, hashed


def validate_run(run_dir: Path, name: str, model_dir: Path | None = None, viewer_dir: Path | None = None,
                 timeout: float = TIMEOUT_S) -> dict:
    model_dir = model_dir or run_dir / "model"
    target, hashed = run_paths(run_dir, name, model_dir, viewer_dir)
    return validate(target, run_dir, hashed, records=model_dir, timeout=timeout)


def problem_rows(v: dict) -> list[str]:
    """One line per failing or warning row, failures first."""
    rows = v.get("rows") or []
    return [f"{r['status'].upper()} {r['check']}: {_first(r['detail'])}"
            for s in ("fail", "warn") for r in rows if r["status"] == s]
