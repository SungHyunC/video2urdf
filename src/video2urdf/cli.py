"""Command line entry point: `v2u <command>`."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def _synth(args: argparse.Namespace) -> None:
    from .synth import PRESETS, SynthConfig, generate

    if args.list:
        print("\n".join(sorted(PRESETS)))
        return
    width, height = (int(v) for v in args.size.lower().split("x"))
    for seed in range(args.seed, args.seed + args.count):
        cfg = SynthConfig(
            preset=args.preset,
            seed=seed,
            width=width,
            height=height,
            fps=args.fps,
            seconds=args.seconds,
            fovy_deg=args.fovy,
            tilt_deg=args.tilt,
            motion=args.motion,
            base_motion=args.base_motion,
            noise_std=args.noise,
        )
        out = Path(args.out) / f"{args.preset}_s{seed:03d}"
        start = time.perf_counter()
        gt = generate(cfg, out)
        joints = ", ".join(f"{j['name']}({j['type'][0]})" for j in gt["joints"])
        print(f"{out}  {gt['video']['num_frames']} frames  joints: {joints}  [{time.perf_counter() - start:.1f}s]")


def _llm_check(args: argparse.Namespace) -> None:
    from .config import load_settings
    from .llm import chat, image_part

    try:
        settings = load_settings(args.profile)
    except (RuntimeError, ValueError) as e:
        raise SystemExit(f"error: {e}") from None
    print(f"profile={settings.profile} base_url={settings.base_url} model={settings.model}")
    content: list[dict] = [{"type": "text", "text": "Reply with one word: ready."}]
    if args.image:
        content = [
            {"type": "text", "text": "In one sentence, list the colored parts you see and how they connect."},
            image_part(args.image),
        ]
    from openai import APIConnectionError

    start = time.perf_counter()
    try:
        response = chat([{"role": "user", "content": content}], settings, max_tokens=args.max_tokens)
    except APIConnectionError:
        hint = (
            " (is scripts/serve_glimmer.sh running and the ssh -L 8000 tunnel open?)"
            if "localhost" in settings.base_url
            else ""
        )
        raise SystemExit(f"error: cannot reach {settings.base_url}{hint}") from None
    message = response.choices[0].message
    print(f"[{time.perf_counter() - start:.1f}s, usage={response.usage}]")
    print(message.content)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="v2u", description="Video2URDF")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="render synthetic videos of mechanisms with known joints")
    p.add_argument("--preset", default="lamp", help="mechanism preset (see --list)")
    p.add_argument("--list", action="store_true", help="list presets and exit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--count", type=int, default=1, help="number of videos, seeds seed..seed+count-1")
    p.add_argument("--out", default="data/synth")
    p.add_argument("--size", default="1280x720", help="WIDTHxHEIGHT")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--fovy", type=float, default=45.0, help="vertical field of view, degrees")
    p.add_argument("--tilt", type=float, default=0.0, help="camera tilt away from the motion plane normal, degrees")
    p.add_argument("--motion", choices=["sequential", "simultaneous"], default=None)
    p.add_argument(
        "--base-motion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="hand-held drift of the root link (default: per preset)",
    )
    p.add_argument("--noise", type=float, default=0.0, help="Gaussian pixel noise std (0-255)")
    p.set_defaults(func=_synth)

    p = sub.add_parser("llm-check", help="send a test prompt to the configured model endpoint")
    p.add_argument("--profile", choices=["meta", "local"], default=None, help="override V2U_PROFILE")
    p.add_argument("--image", default=None, help="optional image to test vision input (e.g. a synth overlay.png)")
    p.add_argument("--max-tokens", type=int, default=2048, help="reasoning models spend tokens before answering")
    p.set_defaults(func=_llm_check)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
