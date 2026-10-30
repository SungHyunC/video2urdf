"""Experimental transfer of an observed texture onto a nearby generated surface.

Visibility must be established separately from the view where the source was observed.
The distance is a visual interpolation tolerance, not a dimensional measurement.
"""
import numpy as np


def transfer_texture(vertices, colors, visible, source_vertices, source_uv, source_faces,
                     source_uv_faces, atlas_bgr, max_distance_m):
    """Copy colours from projected source triangles to visible target vertices.

    Source and target vertices must share the link coordinate system. Atlas UVs use
    OBJ's bottom-left origin. Points outside triangles and hidden points are unchanged.
    """
    vertices = np.asarray(vertices, dtype=float)
    colors = np.asarray(colors).copy()
    visible = np.asarray(visible, dtype=bool)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or colors.shape != vertices.shape
            or visible.shape != (len(vertices),) or not np.isfinite(vertices).all()
            or not np.isfinite(max_distance_m) or max_distance_m <= 0):
        raise ValueError("invalid target vertices, colours, visibility or distance")
    source_vertices = np.asarray(source_vertices, dtype=float)
    source_uv = np.asarray(source_uv, dtype=float)
    source_faces = np.asarray(source_faces, dtype=int)
    source_uv_faces = np.asarray(source_uv_faces, dtype=int)
    atlas_bgr = np.asarray(atlas_bgr)
    if (source_vertices.ndim != 2 or source_vertices.shape[1] != 3 or source_uv.ndim != 2
            or source_uv.shape[1] != 2 or source_faces.ndim != 2 or source_faces.shape[1] != 3
            or source_uv_faces.shape != source_faces.shape or atlas_bgr.ndim != 3
            or atlas_bgr.shape[2] != 3 or not atlas_bgr.size or not np.isfinite(source_vertices).all()
            or not np.isfinite(source_uv).all() or np.any(source_faces < 0) or np.any(source_uv_faces < 0)
            or np.any(source_faces >= len(source_vertices)) or np.any(source_uv_faces >= len(source_uv))):
        raise ValueError("invalid source triangles, UV coordinates or atlas")
    best = np.full(len(vertices), max_distance_m)
    hit = np.zeros(len(vertices), dtype=bool)
    height, width = atlas_bgr.shape[:2]
    for face, uv_face in zip(source_faces, source_uv_faces):
        tri, uv = source_vertices[face], source_uv[uv_face]
        a, b = tri[1] - tri[0], tri[2] - tri[0]
        aa, ab, bb = a @ a, a @ b, b @ b
        denom = aa * bb - ab * ab
        if denom < 1e-14:
            continue
        delta = vertices - tri[0]
        u = (bb * (delta @ a) - ab * (delta @ b)) / denom
        v = (aa * (delta @ b) - ab * (delta @ a)) / denom
        normal = np.cross(a, b)
        distance = np.abs(delta @ normal) / np.linalg.norm(normal)
        ids = np.flatnonzero(visible & (u >= 0) & (v >= 0) & (u + v <= 1) & (distance < best))
        if not len(ids):
            continue
        coords = uv[0] + u[ids, None] * (uv[1] - uv[0]) + v[ids, None] * (uv[2] - uv[0])
        x = np.clip((coords[:, 0] * width).astype(int), 0, width - 1)
        y = np.clip(((1 - coords[:, 1]) * height).astype(int), 0, height - 1)
        colors[ids] = atlas_bgr[y, x, ::-1]
        best[ids] = distance[ids]
        hit[ids] = True
    return colors, hit


def visible_from_depth(camera_points, depth, focal, cx, cy, scale=1., tolerance_m=.004):
    """Select points in front of the camera that agree with its rendered depth map."""
    points = np.asarray(camera_points, dtype=float)
    depth = np.asarray(depth)
    if (points.ndim != 2 or points.shape[1] != 3 or depth.ndim != 2 or not depth.size
            or not np.isfinite([focal, cx, cy, scale, tolerance_m]).all()
            or focal <= 0 or scale <= 0 or tolerance_m <= 0):
        raise ValueError("invalid camera points, depth or projection parameters")
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    uv = np.zeros((len(points), 2))
    uv[valid, 0] = (focal * points[valid, 0] / points[valid, 2] + cx) * scale
    uv[valid, 1] = (focal * points[valid, 1] / points[valid, 2] + cy) * scale
    h, w = depth.shape
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    result = np.zeros(len(points), bool)
    ids = np.flatnonzero(valid)
    x = np.minimum(np.rint(uv[ids, 0]).astype(int), w - 1)
    y = np.minimum(np.rint(uv[ids, 1]).astype(int), h - 1)
    measured = depth[y, x]
    result[ids] = np.isfinite(measured) & (measured > 0) & (np.abs(points[ids, 2] - measured) < tolerance_m)
    return result
