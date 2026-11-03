"""Joint values on frames without a measurement, by render-and-compare against the video.

A 3D fit leaves some frames without a joint value: a lid seen edge-on while it is being closed
fast, a door whose glossy face returns no depth, a rod frame the angle cleaning dropped. The
replay used to draw them at the value interpolated between the measured neighbours, which is
visibly wrong when the part moves fast through the gap (the LiDAR laptop's lid lay flat while
the real one stood upright).

Here each such frame searches its joint's value: the model is rendered from that frame's own
camera (replay3d's poses) at each candidate value and the moving part's silhouette is compared
with its SAM masks on that frame (IoU). The search stays in a window around the interpolated
value (wider the more the joint moves across the gap), a coarse grid first, then a fine one
around the best, with a small pull toward the interpolated value so that equally good values
(a part seen end-on) resolve toward it. A frame keeps the interpolated value, and says so, when
its silhouette can't tell: too few mask pixels, or the score barely changes across the window.

Pure logic (search, fallback, gap runs) is separate from rendering (SilhouetteRenderer) so it
can be tested without MuJoCo.
"""

from dataclasses import dataclass, field

import numpy as np

MEASURED, SILHOUETTE, INTERPOLATED = "measured", "silhouette", "interpolated"


@dataclass
class Settings:
    coarse_steps: int = 25  # grid points across the window
    fine_steps: int = 11  # grid points across +-1 coarse step around the coarse best
    min_window: float = np.deg2rad(20)  # half-width of the search around the interpolated value
    span_factor: float = 1.0  # + this x how far the joint moves across the gap
    pull: float = 0.02  # score lost at the window's edge (quadratic): a tie-breaker toward the interpolated value
    min_range: float = 0.05  # best minus worst score in the window below this: uninformative
    min_pixels: int = 150  # target mask pixels (at the scoring resolution) below this: uninformative
    min_score: float = 0.3  # best IoU below this: the silhouette doesn't match the model at any value


@dataclass
class FrameFill:
    value: float
    source: str  # SILHOUETTE or INTERPOLATED
    reason: str = ""
    best_score: float | None = None
    interp_score: float | None = None
    score_range: float | None = None


@dataclass
class Fill:
    q: np.ndarray  # every frame: measured, from the silhouette, or interpolated
    source: list = field(default_factory=list)  # per frame
    frames: dict = field(default_factory=dict)  # gap frame -> FrameFill

    def summary(self) -> dict:
        return {s: int(sum(x == s for x in self.source)) for s in (MEASURED, SILHOUETTE, INTERPOLATED)}


def gap_runs(missing: np.ndarray) -> list[tuple[int, int]]:
    """Consecutive runs of True as inclusive (first, last) frame pairs."""
    out, start = [], None
    for i, v in enumerate(list(np.asarray(missing, bool)) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def interpolate(q: np.ndarray) -> np.ndarray:
    """Linear between measured neighbours, held at the ends (replay3d's old behaviour)."""
    q = np.asarray(q, float)
    known = np.flatnonzero(~np.isnan(q))
    if not len(known):
        return np.zeros_like(q)
    return np.interp(np.arange(len(q)), known, q[known])


def window(q: np.ndarray, first: int, last: int, lo: float, hi: float, s: Settings) -> float:
    """Half-width of the search for frames first..last: min_window plus how far the joint moves
    between the measured frames on either side."""
    before = q[first - 1] if first > 0 else np.nan
    after = q[last + 1] if last + 1 < len(q) else np.nan
    span = abs(after - before) if np.isfinite(before) and np.isfinite(after) else 0.0
    return min(hi - lo, s.min_window + s.span_factor * span)


def search(score, prior: float, half: float, lo: float, hi: float, s: Settings) -> FrameFill:
    """Best value of ``score(q)`` (higher is better) in [prior - half, prior + half] within the
    limits, coarse then fine, with a quadratic pull toward ``prior``; or the prior itself when
    the score can't tell (flat or low)."""
    a, b = max(lo, prior - half), min(hi, prior + half)
    prior_c = float(np.clip(prior, lo, hi))
    if b <= a:
        return FrameFill(prior_c, INTERPOLATED, "no room within the joint's limits")
    pull = lambda q: s.pull * ((q - prior_c) / max(half, 1e-9)) ** 2
    grid = np.linspace(a, b, s.coarse_steps)
    vals = np.array([score(float(q)) for q in grid])
    interp_score = float(score(prior_c))
    rng = float(max(vals.max(), interp_score) - min(vals.min(), interp_score))
    if rng < s.min_range:
        return FrameFill(prior_c, INTERPOLATED, f"silhouette flat across the window (score range {rng:.3f})",
                         float(vals.max()), interp_score, rng)
    i = int(np.argmax(vals - pull(grid)))
    step = grid[1] - grid[0] if len(grid) > 1 else 0.0
    fine = np.linspace(max(a, grid[i] - step), min(b, grid[i] + step), s.fine_steps)
    fvals = np.array([score(float(q)) for q in fine])
    k = int(np.argmax(fvals - pull(fine)))
    best, best_score = float(fine[k]), float(fvals[k])
    if best_score < s.min_score:
        return FrameFill(prior_c, INTERPOLATED, f"best silhouette IoU only {best_score:.2f}", best_score, interp_score, rng)
    return FrameFill(best, SILHOUETTE, "", best_score, interp_score, rng)


def fill(q: np.ndarray, scorer, lo: float, hi: float, settings: Settings | None = None, log=None) -> Fill:
    """Fill the NaN frames of ``q`` (one joint). ``scorer(t)`` returns None when frame t has no
    usable silhouette (too few pixels), else a function value -> score."""
    s = settings or Settings()
    q = np.asarray(q, float)
    prior = interpolate(q)
    out = prior.copy()
    source = [MEASURED if np.isfinite(v) else INTERPOLATED for v in q]
    frames = {}
    if not np.isfinite(q).any():
        return Fill(out, source, frames)
    for first, last in gap_runs(np.isnan(q)):
        half = window(q, first, last, lo, hi, s)
        for t in range(first, last + 1):
            score = scorer(t)
            if score is None:
                frames[t] = FrameFill(float(prior[t]), INTERPOLATED, "too few silhouette pixels")
            else:
                frames[t] = search(score, float(prior[t]), half, lo, hi, s)
            out[t], source[t] = frames[t].value, frames[t].source
        if log:
            n = sum(frames[t].source == SILHOUETTE for t in range(first, last + 1))
            log(f"gap {first}..{last}: {n} of {last - first + 1} frames from the silhouette")
    return Fill(out, source, frames)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 0.0


class SilhouetteRenderer:
    """The model drawn from a frame's phone camera (as replay3d draws it) at a reduced size:
    the visible pixels of chosen bodies for given joint values."""

    def __init__(self, model_xml, cam_to_world: np.ndarray, intr, size: tuple[int, int], scale: int = 4):
        import mujoco
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.data = mujoco.MjData(self.model)
        self.cam = self.model.camera("video").id
        w, h = size
        self.size = (w // scale, h // scale)
        self.scale = scale
        self.model.cam_fovy[self.cam] = np.rad2deg(2 * np.arctan(h / 2 / intr.f))
        self.renderer = mujoco.Renderer(self.model, self.size[1], self.size[0])
        self.renderer.enable_segmentation_rendering()
        self.cam_to_world = cam_to_world
        self.base_q = {}

    def subtree(self, body: str) -> np.ndarray:
        """Body ids of ``body`` and everything it carries."""
        root = self.model.body(body).id
        ids = [root]
        for b in range(self.model.nbody):
            p = b
            while p > 0 and p != root:
                p = self.model.body_parentid[p]
            if p == root and b != root:
                ids.append(b)
        return np.array(ids)

    def bodies(self, t: int, qs: dict[str, float]) -> np.ndarray:
        """Body id per pixel (-1: none) on frame t with joint values ``qs``."""
        mujoco = self.mujoco
        quat = np.zeros(4)
        flip = np.diag([1.0, -1.0, -1.0])
        mujoco.mju_mat2Quat(quat, (self.cam_to_world[t][:3, :3] @ flip).flatten())
        self.model.cam_pos[self.cam], self.model.cam_quat[self.cam] = self.cam_to_world[t][:3, 3], quat
        for j, v in {**self.base_q, **qs}.items():
            self.data.joint(j).qpos[0] = v
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="video")
        seg = self.renderer.render()
        geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        return np.where(geom >= 0, self.model.geom_bodyid[np.maximum(geom, 0)], -1)

    def shrink(self, mask: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.resize(mask.astype(np.uint8), self.size, interpolation=cv2.INTER_AREA) > 0

    def close(self):
        self.renderer.close()


def joint_scorer(r: SilhouetteRenderer, joint: str, bodies: np.ndarray, target, others: dict | None = None,
                 min_pixels: int = Settings.min_pixels):
    """scorer(t) for ``fill``: IoU of the rendered ``bodies`` against ``target(t)`` (a full-size
    bool mask or None) with ``joint`` at the candidate value and the other joints at
    ``others[j][t]``."""
    def scorer(t: int):
        m = target(t)
        if m is None:
            return None
        m = r.shrink(m)
        if np.count_nonzero(m) < min_pixels:
            return None
        rest = {j: float(v[t]) for j, v in (others or {}).items()}
        cache = {}

        def score(v: float) -> float:
            key = round(v, 6)
            if key not in cache:
                cache[key] = iou(np.isin(r.bodies(t, {**rest, joint: v}), bodies), m)
            return cache[key]
        return score
    return scorer
