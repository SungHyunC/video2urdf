#!/usr/bin/env python3
"""Check a run's final downloads again (after changing its model by hand or rerunning a step) and
refresh validation.json and the run's summary: agent_status stays, artifact_status and status
(the worse of the two) are recomputed.

    .venv/bin/python scripts/validate_run.py data/runs/laptop            # check, update summary
    .venv/bin/python scripts/validate_run.py data/runs/laptop --status   # is validation.json current?

RUN is a folder written by scripts/video2urdf.py (model/, viewer/, summary.json). Local only: no
network, no keys. Exit 1 when the artifact status is failed (or, with --status, stale/missing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video2urdf import validation as V  # noqa: E402
from video2urdf.summary import write_summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="a run folder from scripts/video2urdf.py")
    ap.add_argument("--status", action="store_true", help="only say whether validation.json is still current")
    args = ap.parse_args()
    run = args.run
    summary_path = run / "summary.json"
    s = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    if args.status:
        fresh = V.freshness(run)
        print(f"validation.json: {fresh['state']}" + (f" ({', '.join(fresh['changed'])})" if fresh["changed"] else ""))
        sys.exit(0 if fresh["state"] == "current" else 1)

    name = s.get("name") or run.name
    viewer = run / "viewer"  # else --viewer-out's folder, as the summary recorded it
    recorded = (s.get("outputs") or {}).get("viewer")
    if not viewer.exists() and recorded and Path(recorded).parent.exists():
        viewer = Path(recorded).parent
    v = V.validate_run(run, name, run / "model", viewer)
    for line in V.problem_rows(v):
        print(line)
    print(f"artifact status: {v['artifact_status']} on {v['target']} ({v['seconds']} s)")
    if s:
        artifact, reasons, fresh = V.current_artifact_status(run)
        s["artifact_status"] = artifact
        s["validation"] = {"state": fresh["state"], "changed": fresh["changed"], "target": v["target"],
                           "counts": v["counts"], "problems": V.problem_rows(v)}
        s["issues"] = [i for i in s.get("issues", []) if not i.startswith("downloads: ")] + \
            [f"downloads: {r}" for r in reasons]
        if not s.get("agent_status") and s.get("status") in ("checks_passed", "needs_review", "failed"):
            s["agent_status"] = s["status"]  # a run from before validation: its status was the agent's
        if s.get("agent_status"):
            s["status"] = V.combine(s["agent_status"], artifact)
        s.pop("measured_vs_estimated", None)
        write_summary(run, s)
        print(f"status: {s['status']} (agent {s.get('agent_status')}, downloads {artifact}) -> {run / 'summary.md'}")
    sys.exit(1 if v["artifact_status"] == "failed" else 0)


if __name__ == "__main__":
    main()
