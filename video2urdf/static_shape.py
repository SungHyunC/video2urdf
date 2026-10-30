"""Geometry primitives for a rigid, non-articulated object's observed depth points.

This is a geometric building block, not a complete video-to-model pipeline. A bounding
box over visible surface points cannot measure hidden surfaces or guarantee true size.
"""
import numpy as np


def drawer_handle_up(c, part, floor, frame, detections):
    """Infer a vertical direction from aligned handles and a visible floor.

    Returns reference-camera coordinates, or None if the evidence is ambiguous.
    The direction remains a visual hypothesis and is not a gravity measurement.
    """
    import cv2
    from .lift3d import mask_cloud
    if not c.has(part, frame) or not c.has(floor, frame):
        return None
    obj = c.by[part][frame].full_mask(c.h, c.w)
    ground = c.by[floor][frame].full_mask(c.h, c.w)
    if not obj.any() or not ground.any():
        return None
    distances = cv2.distanceTransform((~ground).astype(np.uint8), cv2.DIST_L2, 5)
    items = []
    for detection in detections:
        mask = detection.full_mask(c.h, c.w)
        if mask.sum() < 10 or (mask & obj).sum() < .8 * mask.sum() or mask.sum() > .12 * obj.sum():
            continue
        points = mask_cloud(mask & obj, c.cap.depth[frame], c.cap.intrinsics)
        if len(points) < 30:
            continue
        pixels = np.argwhere(mask & obj)
        y, x = np.median(pixels, axis=0).astype(int)
        items.append((np.array([x, y], dtype=float), np.median(points, axis=0), float(distances[y, x])))
    if not 3 <= len(items) <= 8:
        return None
    xy = np.array([item[0] for item in items])
    _, singular, axes = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)
    if singular[0] < .08 * min(c.w, c.h) or singular[1] > .25 * singular[0]:
        return None
    order = np.argsort(xy @ axes[0])
    first, last = items[order[0]], items[order[-1]]
    if abs(first[2] - last[2]) < .05 * min(c.w, c.h):
        return None
    top, bottom = (first, last) if first[2] > last[2] else (last, first)
    direction = top[1] - bottom[1]
    spread = np.array([item[1] for item in items])
    _, values, _ = np.linalg.svd(spread - spread.mean(axis=0), full_matrices=False)
    if np.linalg.norm(direction) < .02 or values[1] > .35 * values[0]:
        return None
    direction = np.linalg.inv(c.world[frame])[:3, :3] @ direction
    return direction / np.linalg.norm(direction)


def observed_track_shift(c, part, reference, samples=12):
    """Compare tracked depth centroids in a common frame; viewpoint can affect them."""
    frames = sorted(t for t in c.by.get(part, {}) if 0 <= t < c.n)
    points = c.cloud(part, reference)
    if len(points) < 50:
        raise ValueError('Reference object mask has too few depth points.')
    center = np.median(points, axis=0)
    indices = np.unique(np.linspace(0, len(frames) - 1, min(samples, len(frames))).astype(int))
    shifts = []
    for i in indices:
        cloud = c.cloud(part, frames[i])
        if len(cloud) >= 50:
            shifts.append(float(np.linalg.norm(np.median(cloud, axis=0) - center)))
    if len(shifts) < 3:
        raise ValueError('Too few tracked depth frames to check whether the object is stationary.')
    return float(np.percentile(shifts, 90)), len(shifts)


def upright_box(points, up, *, trim_percent=2.0):
    """Fit a straight-edged upright box to 3D points in the capture frame.

    The vertical axis comes from gravity or a separately measured supporting plane.
    Horizontal PCA supplies an orientation; its direction is unstable for near-square
    objects. Bounds use robust percentiles and describe observed points only.
    """
    points = np.asarray(points, dtype=float)
    up = np.asarray(up, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 50:
        raise ValueError('Need at least 50 three-dimensional depth points.')
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 50:
        raise ValueError('Too few finite depth points.')
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-9:
        raise ValueError('A valid gravity or floor normal is required.')
    if not 0 <= trim_percent < 25:
        raise ValueError('trim_percent must be between 0 and 25.')
    z = up / np.linalg.norm(up)
    candidate = np.eye(3)[np.argmin(np.abs(z))]
    x0 = candidate - z * (candidate @ z)
    x0 /= np.linalg.norm(x0)
    y0 = np.cross(z, x0)
    projected = np.column_stack((points @ x0, points @ y0))
    covariance = np.cov(projected, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    horizontal = vectors[:, np.argmax(values)]
    x = horizontal[0] * x0 + horizontal[1] * y0
    y = np.cross(z, x)
    axes = np.stack([x, y, z])
    coords = points @ axes.T
    bounds = np.percentile(coords, [trim_percent, 100 - trim_percent], axis=0)
    size = bounds[1] - bounds[0]
    if not np.isfinite(size).all() or np.min(size) < .001:
        raise ValueError('Observed depth is too thin or incomplete to form a 3D box.')
    center = ((bounds[0] + bounds[1]) / 2) @ axes
    return {'center': center, 'axes': axes, 'size': size,
            'horizontal_orientation_uncertain': bool(values[-1] <= 1.1 * max(values[0], 1e-12)),
            'note': 'Bounds describe observed depth points; hidden surfaces and full dimensions are not verified.'}


def static_profile(size, shape):
    """Build a centered horizontal outline from observed extents, in metres."""
    size = np.asarray(size, dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('Static shape needs three positive finite extents.')
    hx, hy = size[:2] / 2
    if shape == 'box':
        return np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    if shape == 'cylinder':
        # The visible depth extent can be just the front arc. Averaging it
        # with the silhouette width systematically collapses the cylinder.
        # Use the larger observed extent as the hypothesized diameter.
        radius = max(hx, hy)
        angle = np.arange(32) * (2 * np.pi / 32)
        return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    raise ValueError('Static shape must be box or cylinder.')


def observed_circle(points, axes, extent):
    """Fit a horizontal circle to observed depth, or return None if poorly supported."""
    from scipy.optimize import least_squares
    xy = np.asarray(points, float) @ np.asarray(axes, float)[:2].T
    origin = np.median(xy, axis=0)
    xy = xy - origin
    if len(xy) < 50 or not np.isfinite(xy).all():
        return None
    covariance = np.linalg.eigvalsh(np.cov(xy, rowvar=False))
    if covariance[0] < .005 * covariance[1]:
        return None
    matrix = np.column_stack((2 * xy, np.ones(len(xy))))
    sol, _, rank, _ = np.linalg.lstsq(matrix, (xy * xy).sum(axis=1), rcond=None)
    radius2 = sol[2] + sol[0] ** 2 + sol[1] ** 2
    if rank < 3 or radius2 <= 0:
        return None
    initial = np.r_[sol[:2], np.sqrt(radius2)]
    fit = least_squares(lambda v: np.linalg.norm(xy - v[:2], axis=1) - v[2], initial,
                        loss='soft_l1', f_scale=.003, max_nfev=50)
    if not fit.success or not np.isfinite(fit.x).all():
        return None
    center, radius = fit.x[:2], float(fit.x[2])
    residual = float(np.median(np.abs(np.linalg.norm(xy - center, axis=1) - radius)))
    if not .5 * extent <= 2 * radius <= 1.5 * extent or residual > max(.003, .05 * radius):
        return None
    return origin + center, radius, residual


def transform_reference_up(direction, reference_to_frame):
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
        raise ValueError('up override must be a finite nonzero 3D direction')
    up = np.asarray(reference_to_frame)[:3, :3] @ direction
    return up / np.linalg.norm(up)


def build_static_model(c, part, floor=None, *, shape='box', log=print, up_override=None):
    """Build and replay one simple rigid object from existing RGB-D tracks.

    The object's outline and colour use one observed frame, so this is an approximate
    static envelope. It is not suitable for representing an articulated object.
    """
    import json
    from . import emit3d
    from .lift3d import fit_plane
    from .pipeline import read_frame
    from .replay import replay3d
    from .shape import part_color, prism, write_stl

    frames = [t for t in c.by.get(part, {}) if t < c.n and (not floor or c.has(floor, t))]
    if not frames:
        raise ValueError('No frame has a usable object and supporting surface track.')
    ref = max(frames, key=lambda t: c.by[part][t].area)
    if up_override is not None:
        # Override is in the capture's current reference-camera coordinates.
        up_override = transform_reference_up(up_override, c.world[ref])
    c.set_ref(ref)
    if up_override is not None:
        up = up_override
    elif c.up is not None:
        up = c.up
    elif floor:
        points = c.cloud(floor, ref)
        if len(points) < 50:
            raise ValueError('Supporting surface has too few depth points.')
        up, _, _ = fit_plane(points, np.random.default_rng(0))
        if up @ (-points.mean(axis=0)) < 0:
            up = -up
    else:
        raise ValueError('A floor track or measured gravity is required.')
    object_points = c.cloud(part, ref)
    envelope = upright_box(object_points, up)
    size, center, rot = envelope['size'], envelope['center'], envelope['axes']
    observed_size = size.copy()
    circle = observed_circle(object_points, rot, float(max(size[:2]))) if shape == 'cylinder' else None
    if circle is not None:
        circle_center, radius, _ = circle
        center = center + (circle_center - (center @ rot[:2].T)) @ rot[:2]
        size = size.copy()
        size[:2] = 2 * radius
    elif shape == 'cylinder':
        size = size.copy()
        size[:2] = max(size[:2])
    shift, shift_frames = observed_track_shift(c, part, ref)
    if shift > max(.05, .25 * float(max(size))):
        raise ValueError(f'The tracked object changes position or visible depth substantially '
                         f'({shift * 1000:.0f} mm over {shift_frames} sampled frames). '
                         'Inspect whether it moves, the mask changes, or camera tracking drifts; '
                         'do not fit it as stationary until this is resolved.')
    hz = size[2] / 2
    profile = static_profile(size, shape)
    vertices, faces = prism(profile, np.zeros(3), *np.eye(3), -hz, hz)
    mesh_dir = c.track_dir / 'meshes'
    mesh_dir.mkdir(exist_ok=True)
    write_stl(mesh_dir / 'static_object.stl', vertices, faces)
    rgb = part_color(read_frame(c.video, ref), c.by[part][ref].full_mask(c.h, c.w))
    links = [emit3d.Link3D('base', [emit3d.Visual('meshes/static_object.stl', rgb)])]
    (c.track_dir / 'model.urdf').write_text(emit3d.urdf(c.video.stem, links))
    (c.track_dir / 'model.xml').write_text(emit3d.mjcf(c.video.stem, links, (0, 0, hz), (c.w, c.h)))
    (c.track_dir / 'joints.json').write_text(json.dumps({'joints': []}))
    (c.track_dir / 'static.json').write_text(json.dumps({'part': part, 'floor': floor, 'reference_frame': ref, 'shape': shape}, indent=2))
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = -rot @ center + [0, 0, hz]
    cameras = np.array([transform @ np.linalg.inv(c.world[t]) for t in range(c.n)])
    coverage = replay3d(c.track_dir / 'model.xml', cameras, c.cap.intrinsics, {}, {'base': c.by[part]},
                        c.video, c.track_dir / 'replay3d.mp4')['base']
    good = coverage[np.isfinite(coverage)]
    warnings = [f'WARNING: The static {shape} approximation uses visible depth points; hidden surfaces and complete dimensions are not measured.']
    if up_override is not None:
        warnings.append('WARNING: Upright direction was supplied as a visual hypothesis, not measured gravity.')
    if shape == 'cylinder' and circle is None:
        warnings.append('WARNING: Visible depth did not establish a circular cross-section; the cylinder is a shape assumption. Inspect the video and consider a box or another shape.')
    summary = {'reference_frame': ref, 'observed_box_size_mm': (observed_size * 1000).tolist(),
               'generated_size_mm': (size * 1000).tolist(),
               'video_frames': c.n, 'mask_coverage': {'base': float(good.mean()) if len(good) else None},
               'mask_coverage_frames': {'base': len(good)},
               'static_envelope': True, 'static_shape': shape, 'horizontal_orientation_uncertain': envelope['horizontal_orientation_uncertain'],
               'cylinder_curve_fit': bool(circle is not None),
               'observed_centroid_shift_p90_mm': shift * 1000, 'stationarity_sample_frames': shift_frames,
               'fit_warnings': warnings}
    (c.track_dir / 'model3d.json').write_text(json.dumps(summary, indent=2) + '\n')
    # These describe prior articulated fits, not this single-link model.
    for name in ('roles.json', 'hinge.json', 'slide.json', 'slides.json', 'drawers.json'):
        (c.track_dir / name).unlink(missing_ok=True)
    log(f'Static {shape}: observed envelope {np.round(observed_size * 1000, 1).tolist()} mm; '
        f'generated size {np.round(size * 1000, 1).tolist()} mm; replay coverage {len(good)}/{c.n} frames.')
    return summary
