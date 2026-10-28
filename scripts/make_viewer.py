"""Build a local HTML page to review runs: replay video, Spark's report and its step log.

Usage: .venv/bin/python scripts/make_viewer.py [--tracks data/tracks] [--open]
Writes <tracks>/index.html (next to the run folders, so videos and images load locally).
"""

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STYLE = """
:root { --bg: #fafafa; --fg: #1b1b1b; --muted: #666; --card: #fff; --line: #ddd; --accent: #2a6df4; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #141414; --fg: #eee; --muted: #999; --card: #1e1e1e; --line: #333; --accent: #7aa7ff; }
}
body { margin: 0; padding: 16px; background: var(--bg); color: var(--fg); font: 15px/1.5 -apple-system, system-ui, sans-serif; }
h1 { font-size: 22px; } h2 { font-size: 18px; margin: 0 0 8px; }
nav a { margin-right: 12px; color: var(--accent); }
section { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 16px 0; }
.top { display: flex; gap: 16px; flex-wrap: wrap; }
video { max-height: 70vh; max-width: 100%; border-radius: 8px; background: #000; }
.report { flex: 1 1 320px; white-space: pre-wrap; font-size: 14px; }
.joints { font-family: ui-monospace, monospace; font-size: 13px; color: var(--muted); white-space: pre-wrap; }
details { margin-top: 12px; }
.step { border-top: 1px solid var(--line); padding: 8px 0; }
.step img { max-width: 100%; max-height: 420px; display: block; margin-top: 6px; border-radius: 6px; }
.tool { font-family: ui-monospace, monospace; font-size: 13px; color: var(--accent); }
.result { font-family: ui-monospace, monospace; font-size: 12px; color: var(--muted); white-space: pre-wrap; }
.spark { white-space: pre-wrap; }
"""


def log_html(log: str, base: str) -> str:
    out = []
    for block in log.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        img = re.fullmatch(r"!\[\]\((.+)\)", block)
        if img:
            out.append(f'<img loading="lazy" src="{base}/{html.escape(img.group(1))}">')
        elif block.startswith("## "):
            out.append(f'<div class="step"><b>{html.escape(block[3:])}</b></div>')
        elif block.startswith("**"):
            out.append(f'<div class="tool">{html.escape(block.replace("**", "").replace("`", ""))}</div>')
        elif block.startswith(">"):
            text = "\n".join(line[2:] if line.startswith("> ") else line.lstrip(">") for line in block.splitlines())
            out.append(f'<div class="result">{html.escape(text)}</div>')
        elif block.startswith("# "):
            continue
        else:
            out.append(f'<div class="spark">{html.escape(block)}</div>')
    return "\n".join(out)


def joints_text(run: Path) -> str:
    f = run / "joints.json"
    if (run / "model3d.json").exists():
        m = json.loads((run / "model3d.json").read_text())
        lo, hi = m["hinge_range_deg"]
        return (f"3D (iPhone LiDAR): plate {m['plate_size_mm'][0]:.0f} x {m['plate_size_mm'][1]:.0f} mm, "
                f"base {m['base_thickness_mm']:.0f} mm thick, hinge {lo:.0f}..{hi:.0f} deg")
    if not f.exists():
        return ""
    j = json.loads(f.read_text())
    lines = [f"scale {j['px_per_m']:.0f} px/m, reference frame {j['reference_frame']}"]
    for k in j["joints"]:
        q = [v for v in k["q"] if v is not None]
        if k["type"] == "revolute":
            rng = f"{min(q) * 57.2958:.1f}..{max(q) * 57.2958:.1f} deg"
        elif k["type"] == "prismatic":
            rng = f"{1000 * min(q):.0f}..{1000 * max(q):.0f} mm"
        else:
            rng = ""
        lines.append(f"{k['parent']} -> {k['child']}: {k['type']} {rng}, residual {1000 * k['residual_m']:.1f} mm")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=Path, default=ROOT / "data" / "tracks")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    def replay_of(d):
        return next((f for f in ("replay3d.mp4", "replay.mp4") if (d / f).exists()), None)

    runs = sorted((d for d in args.tracks.iterdir() if d.is_dir() and replay_of(d)),
                  key=lambda d: (not (d / "agent" / "report.md").exists(), d.name))
    sections = []
    for run in runs:
        report = run / "agent" / "report.md"
        log = run / "agent" / "log.md"
        body = [f'<section id="{run.name}"><h2>{html.escape(run.name)}</h2><div class="top">',
                f'<video controls loop muted playsinline src="{run.name}/{replay_of(run)}"></video>',
                *([f'<video controls loop muted playsinline src="{run.name}/robot_demo.mp4"></video>']
                  if (run / "robot_demo.mp4").exists() else []),
                '<div class="report">']
        if report.exists():
            body.append(f"<b>Spark report</b>\n{html.escape(report.read_text())}")
        body.append(f'<div class="joints">{html.escape(joints_text(run))}</div></div></div>')
        if log.exists():
            body.append(f"<details><summary>Agent steps (what Spark saw and did)</summary>"
                        f"{log_html(log.read_text(), run.name + '/agent')}</details>")
        body.append("</section>")
        sections.append("\n".join(body))
    nav = " ".join(f'<a href="#{r.name}">{html.escape(r.name)}</a>' for r in runs)
    page = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width, initial-scale=1'><title>Video2URDF runs</title>"
            f"<style>{STYLE}</style></head><body><h1>Video2URDF runs</h1><nav>{nav}</nav>"
            + "\n".join(sections) + "</body></html>")
    out = args.tracks / "index.html"
    out.write_text(page)
    print(out)
    if args.open:
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    sys.exit(main())
