"""Make a separately calibrated copy of an existing MapAnything RGB capture.

Example: python scripts/calibrate_capture.py CAPTURE NEW_CAPTURE --reference FRAME X1 Y1 X2 Y2 LENGTH_MM
Optional --check-reference uses different points to check the global correction. A single
reference cannot correct local depth or focal errors. The source capture is never modified.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from video2urdf.calibration import reference_scale, check_references  # noqa: E402
from video2urdf.pseudo_depth import rescale_capture  # noqa: E402


def calibrate_copy(source: Path, destination: Path, reference, checks=()):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise ValueError(f'Destination already exists: {destination}')
    if destination == source or source in destination.parents:
        raise ValueError('Destination must be outside the source capture.')
    meta = source / 'mapanything.json'
    if not meta.is_file():
        raise ValueError('Source must be a complete MapAnything capture.')
    if len(reference) != 6 or not float(reference[0]).is_integer():
        raise ValueError('Reference requires an integer frame, four pixel coordinates and a length.')
    for check in checks:
        if len(check) != 6 or not float(check[0]).is_integer():
            raise ValueError('Check requires an integer frame, four pixel coordinates and a length.')
        if check[0] == reference[0] and {(check[1], check[2]), (check[3], check[4])} == {(reference[1], reference[2]), (reference[3], reference[4])}:
            raise ValueError('Check reference must use different points from the calibration reference.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.calibrate-', dir=destination.parent) as tmp:
        stage = Path(tmp) / 'capture'
        shutil.copytree(source, stage)
        frame, x1, y1, x2, y2, length = reference
        result = reference_scale(stage, int(frame), (x1, y1), (x2, y2), length)
        checked = check_references(stage, checks, result['scale_factor'])
        rescale_capture(stage, result['scale_factor'])
        info = json.loads((stage / 'mapanything.json').read_text())
        if info.get('reference_calibration'):
            info.setdefault('calibration_history', []).append({
                'reference_calibration': info['reference_calibration'],
                'reference_checks': info.get('reference_checks', []),
                'scale_applied': info.get('scale_applied', 1.0),
            })
        info['scale_applied'] = info.get('scale_applied', 1.0) * result['scale_factor']
        info['reference_calibration'] = result
        info['reference_checks'] = checked
        (stage / 'mapanything.json').write_text(json.dumps(info, indent=2) + '\n')
        stage.rename(destination)
    return result, checked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('destination', type=Path)
    ap.add_argument('--reference', nargs=6, type=float, required=True,
                    metavar=('FRAME', 'X1', 'Y1', 'X2', 'Y2', 'LENGTH_MM'))
    ap.add_argument('--check-reference', nargs=6, type=float, action='append', default=[],
                    metavar=('FRAME', 'X1', 'Y1', 'X2', 'Y2', 'LENGTH_MM'))
    args = ap.parse_args()
    try:
        ref, checks = calibrate_copy(args.source, args.destination, args.reference, args.check_reference)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    print(f"Calibrated copy: {args.destination}; scale ×{ref['scale_factor']:.5f}. Original capture unchanged.")
    for check in checks:
        print(f"Check at frame {check['frame']}: {check['relative_error']:+.1%} against supplied length.")

if __name__ == '__main__':
    main()
