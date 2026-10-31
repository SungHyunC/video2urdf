"""Complete the hidden body of a partly seen object (amodal completion; video2urdf/completion.py).

A door model whose body is only the part SAM tracked (a microwave's control-panel strip) gets
its body completed: the front measured from LiDAR depth, the hidden depth bounded by the video's
silhouettes (and a wall seen behind it), the rest estimated by Muse Spark (category defaults if
it gives nothing usable), built as a hollow carcass that collides. The measured-only model is
kept (model.measured.{xml,urdf}; completion/measured/ holds every file it changed).

Usage:
  .venv/bin/python scripts/complete_body.py RUN VIDEO [--capture DIR] [--check | --remove]
      [--force] [--open-weights] [--prompt NOUN] [--no-spark] [--note TEXT]

Without --force it only acts on runs whose body is known to be partial (a door model whose base
is a fragment, or a complete_body.flag file in the run); other runs are left untouched.
--check prints the decision and writes nothing. --remove restores the measured-only model.
--open-weights sends the estimate to Muse Spark Open (Baseten) and segments with SAM 3.1's open
weights (Modal), as scripts/run_agent.py --open-weights does.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("video", type=Path)
    ap.add_argument("--capture", type=Path, help="the Stray Scanner capture (default: found next to the video)")
    ap.add_argument("--check", action="store_true", help="say whether the body is partial; write nothing")
    ap.add_argument("--remove", action="store_true", help="restore the measured-only model")
    ap.add_argument("--force", action="store_true", help="complete even if the body isn't known to be partial")
    ap.add_argument("--open-weights", action="store_true", help="Muse Spark Open + SAM 3.1 weights only")
    ap.add_argument("--prompt", help="SAM noun for the whole object (default: the word its parts' prompts share)")
    ap.add_argument("--no-spark", action="store_true", help="don't ask Muse Spark: category defaults for hidden sizes")
    ap.add_argument("--note", help="a line appended to the run's 'Not measured' note (e.g. caveats kept from before)")
    args = ap.parse_args()
    if args.open_weights:
        from run_agent import OPEN_WEIGHTS
        os.environ.update(OPEN_WEIGHTS)
    from video2urdf import completion  # reads the backend settings on import
    if args.remove:
        completion.restore(args.run)
        for f in ("completion.json", "model.measured.xml", "model.measured.urdf"):
            (args.run / f).unlink(missing_ok=True)
        print(f"restored the measured-only model in {args.run}")
        return
    partial, why = completion.needs_completion(args.run)
    print(("partial body: " if partial else "left as it is: ") + why)
    if args.check or not (partial or args.force):
        return
    from video2urdf.stray import find_capture
    capture = args.capture or find_capture(args.video.parent)
    result = completion.complete_body(args.run, args.video, capture, spark=not args.no_spark, prompt=args.prompt,
                                      note=args.note)
    size = result["built"]["size_mm"]
    print(f"completed {result['object']}: {size[0]:.0f} wide x {size[1]:.0f} deep x {size[2]:.0f} tall (mm); "
          f"see {args.run / 'completion.json'}")


if __name__ == "__main__":
    main()
