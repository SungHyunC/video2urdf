"""Resume appearance on a separate copy of an interrupted fitted model.

An optional existing appearance.json avoids a new classification call. Reuse is
explicitly recorded; a resumed result is not an uninterrupted end-to-end run.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from infer_appearance import infer_appearance  # noqa: E402
from run_agent import write_measurement_source  # noqa: E402
from video2urdf.assessment import assess  # noqa: E402


def resume(source: Path, video: Path, destination: Path, appearance: Path | None = None,
           texture: bool = False, capture: Path | None = None) -> dict:
    source, video, destination = source.resolve(), video.resolve(), destination.resolve()
    if not video.is_file():
        raise ValueError('source video is missing')
    if destination.exists():
        raise ValueError('destination already exists; choose a new directory')
    if source == destination or source in destination.parents:
        raise ValueError('destination must not be inside the source')
    for name in ('model.xml', 'model3d.json', 'replay3d.npz', 'parts.json', 'tracks.sam.npz'):
        if not (source / name).is_file():
            raise ValueError(f'source fit is missing {name}')
    if capture is not None:
        capture = capture.resolve()
        if not capture.is_dir():
            raise ValueError('capture directory is missing')
    if not (source / 'measurement_source.json').exists() and capture is None:
        raise ValueError('source has no depth provenance; supply the original capture directory')
    if appearance is not None:
        appearance = appearance.resolve()
        data = json.loads(appearance.read_text())
        if not isinstance(data, dict) or not isinstance(data.get('object'), str) or not data['object'].strip():
            raise ValueError('appearance file must contain a named object')
        if data.get('category') not in ('laptop', 'drawer', 'book', 'box', 'other'):
            raise ValueError('appearance file must contain a supported category')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.resume-appearance-', dir=destination.parent) as tmp:
        stage = Path(tmp) / 'result'
        shutil.copytree(source, stage)
        if not (stage / 'measurement_source.json').exists():
            write_measurement_source(stage, capture)
        if appearance is not None:
            shutil.copy2(appearance, stage / 'appearance.json')
        record = {'method': 'resumed_appearance', 'source_fit': str(source), 'video': str(video),
                  'classification': 'reused' if appearance is not None else 'new',
                  'appearance_source': str(appearance) if appearance is not None else None,
                  'note': 'This result resumes an existing fit and is not an uninterrupted autonomous run.'}
        (stage / 'resume.json').write_text(json.dumps(record, indent=2) + '\n')
        infer_appearance(stage, video, reuse=appearance is not None, texture=texture)
        assess(stage)
        stage.rename(destination)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('video', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--appearance-from', type=Path, help='explicitly reuse a prior appearance.json')
    parser.add_argument('--video-texture', action='store_true')
    parser.add_argument('--capture', type=Path, help='original capture directory if source lacks depth provenance')
    args = parser.parse_args()
    print(json.dumps(resume(args.source, args.video, args.destination, args.appearance_from,
                            args.video_texture, args.capture), indent=2))


if __name__ == '__main__':
    main()
