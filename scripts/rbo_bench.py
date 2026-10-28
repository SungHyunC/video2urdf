"""RBO benchmark: run the Muse Spark agent on each downloaded sequence, score it against
motion capture, and summarise.

Usage: .venv/bin/python scripts/rbo_bench.py [laptop02 book03 ...]
Without names, runs every sequence of a plate-hinge object (laptop, book, treasurebox,
cardboardbox) found in data/rbo/x/. Agent runs are cached (data/rbo/agent/<name>/agent/report.md);
results go to data/rbo/results/<name>.json and data/rbo/results/summary.{json,md}.
RBO dataset: TU Berlin, CC BY 4.0 (see video2urdf/rbo.py).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "rbo"
PLATES = ("laptop", "book", "treasurebox", "cardboardbox")
PY = str(ROOT / ".venv" / "bin" / "python")


def run(cmd: list[str], log: Path) -> int:
    with open(log, "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def main():
    names = sys.argv[1:] or sorted(p.name for p in (DATA / "x").iterdir()
                                   if p.is_dir() and p.name.rstrip("0123456789") in PLATES)
    rows = []
    for name in names:
        log = DATA / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        run([PY, "scripts/rbo_eval.py", name, "--prepare-only"], log)
        if not (DATA / "agent" / name / "agent" / "report.md").exists():
            run([PY, "scripts/run_agent.py", str(DATA / "prep" / name), "--out", str(DATA / "agent")], log)
        ok = (DATA / "agent" / name / "roles.json").exists() and \
            run([PY, "scripts/rbo_eval.py", name, "--tracks", str(DATA / "agent")], log) == 0
        result = DATA / "results" / f"{name}.json"
        rows.append(json.loads(result.read_text()) if ok and result.exists() else {"sequence": name, "failed": True})
        print(name, "ok" if ok else "FAILED (see " + str(log.relative_to(ROOT)) + ")", flush=True)

    done = [r for r in rows if not r.get("failed")]
    keys = ["axis_direction_error_deg", "axis_position_error_mm", "angle_error_deg_median", "angle_error_deg_p90"]
    summary = {"sequences": len(rows), "scored": len(done),
               **{k: float(np.median([r[k] for r in done])) if done else None for k in keys}}
    (DATA / "results" / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    lines = ["| sequence | axis dir (deg) | axis pos (mm) | angle median (deg) | angle p90 (deg) | frames |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("failed"):
            lines.append(f"| {r['sequence']} | failed | | | | |")
        else:
            lines.append(f"| {r['sequence']} | {r['axis_direction_error_deg']:.1f} | {r['axis_position_error_mm']:.0f} | "
                         f"{r['angle_error_deg_median']:.1f} | {r['angle_error_deg_p90']:.1f} | "
                         f"{r['frames_measured']}/{r['frames']} |")
    if done:
        lines.append(f"| **median of {len(done)}** | **{summary[keys[0]]:.1f}** | **{summary[keys[1]]:.0f}** | "
                     f"**{summary[keys[2]]:.1f}** | **{summary[keys[3]]:.1f}** | |")
    (DATA / "results" / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
