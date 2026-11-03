"""Gap filling by silhouette (video2urdf/gapfill.py): the search and its fallbacks, no MuJoCo."""

import numpy as np

from video2urdf import gapfill
from video2urdf.gapfill import INTERPOLATED, MEASURED, SILHOUETTE, Settings


def peak(at, width=0.1, floor=0.1, top=0.8):
    return lambda q: floor + (top - floor) * np.exp(-((q - at) / width) ** 2)


def test_gap_runs():
    m = np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)
    assert gapfill.gap_runs(m) == [(1, 2), (4, 4), (7, 7)]
    assert gapfill.gap_runs(np.zeros(3, bool)) == []


def test_interpolate_holds_ends():
    q = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
    assert np.allclose(gapfill.interpolate(q), [1, 1, 2, 3, 3])


def test_window_grows_with_the_motion_across_the_gap():
    s = Settings(min_window=0.2, span_factor=1.0)
    q = np.array([0.0, np.nan, np.nan, -1.0])
    assert np.isclose(gapfill.window(q, 1, 2, -3, 3, s), 1.2)
    assert np.isclose(gapfill.window(q, 1, 2, -0.5, 0.5, s), 1.0)  # never wider than the range


def test_search_finds_the_peak_away_from_the_prior():
    f = gapfill.search(peak(-0.7), prior=-0.2, half=1.0, lo=-2, hi=0.2, s=Settings())
    assert f.source == SILHOUETTE
    assert abs(f.value + 0.7) < 0.02


def test_search_stays_in_limits():
    f = gapfill.search(peak(0.5), prior=0.0, half=1.0, lo=-1, hi=0.2, s=Settings())
    assert f.value <= 0.2 + 1e-9


def test_flat_silhouette_keeps_the_interpolated_value():
    f = gapfill.search(lambda q: 0.5 + 0.001 * q, prior=0.3, half=1.0, lo=-2, hi=2, s=Settings())
    assert f.source == INTERPOLATED and f.value == 0.3 and "flat" in f.reason


def test_poor_match_keeps_the_interpolated_value():
    f = gapfill.search(peak(0.0, top=0.25, floor=0.0), prior=0.3, half=1.0, lo=-2, hi=2, s=Settings())
    assert f.source == INTERPOLATED and "IoU" in f.reason


def test_equal_peaks_resolve_toward_the_prior():
    two = lambda q: max(peak(-0.5)(q), peak(0.5)(q))
    f = gapfill.search(two, prior=0.3, half=1.0, lo=-2, hi=2, s=Settings())
    assert abs(f.value - 0.5) < 0.02


def test_fill_marks_sources_and_falls_back_without_pixels():
    q = np.array([0.0, np.nan, np.nan, np.nan, -1.0])
    truth = {1: -0.8, 2: None, 3: -0.9}
    f = gapfill.fill(q, lambda t: None if truth[t] is None else peak(truth[t]), lo=-2, hi=0.5)
    assert f.source == [MEASURED, SILHOUETTE, INTERPOLATED, SILHOUETTE, MEASURED]
    assert abs(f.q[1] + 0.8) < 0.02 and abs(f.q[3] + 0.9) < 0.02
    assert f.q[2] == -0.5  # interpolated
    assert f.q[0] == 0.0 and f.q[4] == -1.0  # measured values untouched
    assert f.summary() == {MEASURED: 2, SILHOUETTE: 2, INTERPOLATED: 1}


def test_fill_without_any_measurement_changes_nothing():
    f = gapfill.fill(np.full(3, np.nan), lambda t: peak(1.0), lo=-2, hi=2)
    assert np.allclose(f.q, 0) and f.frames == {}
