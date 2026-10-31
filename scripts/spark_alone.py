"""Spark alone vs Video2URDF: ask Muse Spark plainly for a URDF of each object, then score it.

The baseline for "why not just ask an AI?" (video2urdf/spark_alone.py has the prompt, the cases
and the scoring). Same Meta model, same video; no tools.

  .venv/bin/python scripts/spark_alone.py ask --runs 5          # every case, 5 answers each
  .venv/bin/python scripts/spark_alone.py ask laptop_lidar --runs 5
  .venv/bin/python scripts/spark_alone.py score                 # URDF Doctor etc. on every answer
  .venv/bin/python scripts/spark_alone.py table                 # data/spark_alone/summary.{md,json}

ask resumes: a run folder that already has response.json is kept. The RBO cases are scored against
copies refit with the current code (scripts/refit_lidar.py data/rbo/agent/<seq> data/rbo/prep/<seq>
--out data/spark_alone/measured_rbo/<seq>).
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from video2urdf import spark_alone as S  # noqa: E402


def runs_of(case: str) -> list[Path]:
    return sorted((S.OUT / case).glob("run*"), key=lambda p: int(p.name[3:]))


def cmd_ask(cases: list[str], runs: int, workers: int):
    import os

    from openai import OpenAI
    from video2urdf.agent import API_KEY_ENV, BASE_URL
    client = OpenAI(base_url=BASE_URL, api_key=os.environ[API_KEY_ENV], timeout=900)
    jobs = [(c, S.OUT / c / f"run{i}") for c in cases for i in range(1, runs + 1)
            if not (S.OUT / c / f"run{i}" / "response.json").exists()]
    print(f"{len(jobs)} requests to {S.MODEL}")
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(S.ask, c, d, client): (c, d) for c, d in jobs}
        for f in as_completed(futures):
            c, d = futures[f]
            try:
                r = f.result()
                print(f"{c}/{d.name}: {'URDF' if r['urdf_found'] else 'no URDF'} in {r['seconds']} s")
            except Exception as e:
                print(f"{c}/{d.name}: failed: {type(e).__name__}: {e}")


def _score_one(args):
    case, run = args
    urdf = run / "spark.urdf"
    result = S.score(urdf, case) if urdf.exists() else {"usable": False, "why": "no URDF in the answer"}
    (run / "score.json").write_text(json.dumps(result, indent=2, default=float))
    return case, run.name, result


def cmd_score(cases: list[str], workers: int):
    for c in cases:
        out = S.OUT / c / "measured.json"
        out.write_text(json.dumps(S.measured_scores(c), indent=2, default=float))
    jobs = [(c, r) for c in cases for r in runs_of(c)]
    with ProcessPoolExecutor(workers) as pool:
        for c, name, r in pool.map(_score_one, jobs):
            if r["usable"]:
                brief = (f"travel {r['travel_mm']}" if "travel_mm" in r else
                         f"axis {r['axis_direction_error_deg']:.1f} deg, {r['axis_offset_mm']:.0f} mm"
                         + (f"; mocap {r['mocap']['axis_direction_error_deg']:.1f} deg, "
                            f"{r['mocap']['axis_position_error_mm']:.0f} mm" if "mocap" in r else ""))
            else:
                brief = r["why"]
            print(f"{c}/{name}: {brief}")


def _stat(values: list[float], fmt: str = "{:.0f}") -> str:
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return "-"
    med = fmt.format(float(np.median(v)))
    return med if len(v) == 1 else f"{med} ({fmt.format(min(v))}-{fmt.format(max(v))})"


def cmd_table(cases: list[str]):
    rows, data = [], {}
    for c in cases:
        case = S.CASES[c]
        meas = json.loads((S.OUT / c / "measured.json").read_text())
        scores = [json.loads((r / "score.json").read_text()) for r in runs_of(c) if (r / "score.json").exists()]
        ok = [s for s in scores if s["usable"]]
        data[c] = {"label": case.label, "runs": len(scores), "usable": len(ok), "measured": meas, "spark": scores}
        name = f"{case.label} ({len(ok)} of {len(scores)} answers usable)"
        if case.kind == "slide":
            rows.append((name, "pulls out", _stat([s["travel_mm"] for s in ok]) + " mm",
                         f"{meas['travel_mm']:.0f} mm", "-"))
            continue
        ext = [s["moving_extents_mm"] for s in ok]
        size = (_stat([e[0] for e in ext]) + " x " + _stat([e[1] for e in ext]) + " mm") if ext else "-"
        ours = meas["plate_size_mm"]  # the moving part's outline, as the run reports it
        truth = case.truth.get("moving_mm")
        rows.append((name, "moving part: along the hinge x across", size, f"{ours[0]:.0f} x {ours[1]:.0f} mm",
                     f"{truth[0]} x {truth[1]} mm ({case.truth['source']})" if truth else "-"))
        if case.rbo:
            m = meas["mocap"]
            rows.append(("", "hinge axis vs motion capture",
                         _stat([s["mocap"]["axis_direction_error_deg"] for s in ok], "{:.1f}") + " deg",
                         f"{m['axis_direction_error_deg']:.1f} deg", "0"))
            rows.append(("", "hinge position vs motion capture",
                         _stat([s["mocap"]["axis_position_error_mm"] for s in ok]) + " mm",
                         f"{m['axis_position_error_mm']:.0f} mm", "0"))
        else:
            rows.append(("", "hinge axis vs the measured one",
                         _stat([s["axis_direction_error_deg"] for s in ok], "{:.1f}") + " deg", "(reference)", "-"))
            rows.append(("", "hinge position vs the measured one",
                         _stat([s["axis_offset_mm"] for s in ok]) + " mm", "(reference)", "-"))
        # Joint range: how many degrees the URDF's limits allow against how far the lid opened in
        # the video (motion capture where there is one). Spans, not limits, so neither model's
        # choice of zero counts; a range shorter than the opening can't reproduce the video.
        seen = meas["mocap"]["true_range_deg"] if case.rbo else (
            float(np.subtract(*ok[0]["opening_seen_deg"][::-1])) if ok else None)
        spans = [s["urdf_limits_deg"][1] - s["urdf_limits_deg"][0] if s["urdf_limits_deg"] else 360.0 for s in ok]
        short = sum(sp < seen - S.doctor.LIMIT_TOL_DEG for sp in spans) if seen else 0
        rows.append(("", f"joint range allowed (the video opens it {seen:.0f} deg)" if seen else "joint range allowed",
                     _stat(spans) + f" deg; {short} of {len(ok)} shorter than the opening",
                     f"{meas['limit_span_deg']:.0f} deg",
                     f"{seen:.0f} deg opened (motion capture)" if case.rbo else "-"))
    lines = ["| Object | Check | Muse Spark alone (median, range) | Video2URDF | Truth |", "|---|---|---|---|---|"]
    lines += [f"| {' | '.join(r)} |" for r in rows]
    (S.OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (S.OUT / "summary.json").write_text(json.dumps(data, indent=2, default=float))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("ask", "score", "table"))
    ap.add_argument("cases", nargs="*", help=f"default: all of {', '.join(S.CASES)}")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cases = args.cases or list(S.CASES)
    unknown = [c for c in cases if c not in S.CASES]
    if unknown:
        ap.error(f"unknown case(s): {', '.join(unknown)}")
    if args.command == "ask":
        cmd_ask(cases, args.runs, args.workers)
    elif args.command == "score":
        cmd_score(cases, args.workers)
    else:
        cmd_table(cases)


if __name__ == "__main__":
    main()
