#!/usr/bin/env python3
"""Unit tests for the pure plane-geometry constraint primitives.

These pin the maths that a constraint-based reconstruction relies on — TLS line
fitting (stable for verticals), corner-as-intersection, bilateral symmetry, and
collinearity — using the *actual* numbers from the ``AI`` logo reconstruction, so a
regression fails here in milliseconds and the cases read as documentation of why
each primitive exists.
"""
from __future__ import annotations

import math
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_shadow = sys.modules.get("frameforge")
if _shadow is not None and not hasattr(_shadow, "__path__"):
    del sys.modules["frameforge"]
sys.path[:0] = [ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "docs")]

import pytest  # noqa: E402

from frameforge_vision.domain import geometry as G  # noqa: E402


# ── the real reconstruction's feature pairs (left, right), image px ──
FEET_OUTER = ((134.0, 597.0), (529.0, 596.0))   # mid 331.5
FEET_INNER = ((216.0, 597.0), (443.0, 595.0))   # mid 329.5
COUNTER_BASE = ((283.0, 438.0), (381.0, 438.0))  # mid 332.0
NOTCH = ((248.0, 508.0), (413.0, 506.0))         # mid 330.5
APEX_BAD = ((315.0, 200.0), (366.0, 198.0))      # mid 340.5 — the 9px-shifted apex
GOOD_PAIRS = [FEET_OUTER, FEET_INNER, COUNTER_BASE, NOTCH]


# ─────────────────────────────────────────────────────────────
# Line — direction, distance, project, canonical equality
# ─────────────────────────────────────────────────────────────
def test_line_direction_is_unit_and_canonical():
    line = G.Line(10.0, 20.0, 3.0, 4.0)
    assert math.hypot(line.ux, line.uy) == pytest.approx(1.0)
    # canonical sign: a→b and b→a yield the same unit direction...
    ab = G.Line.from_points((0, 0), (1, 2))
    ba = G.Line.from_points((1, 2), (0, 0))
    assert (ab.ux, ab.uy) == pytest.approx((ba.ux, ba.uy))
    # ...and each line still passes through both anchor points
    assert ab.distance((1, 2)) == pytest.approx(0.0, abs=1e-9)
    assert ba.distance((0, 0)) == pytest.approx(0.0, abs=1e-9)


def test_line_distance_and_project():
    line = G.Line.from_points((0, 0), (0, 10))   # the y-axis (vertical)
    assert line.distance((5, 3)) == pytest.approx(5.0)
    assert line.project((5, 3)) == pytest.approx((0.0, 3.0))


def test_line_angle_from_vertical():
    # left outer edge of the A: (306,201) -> (134,597); atan(172/396) ≈ 23.47°
    left = G.Line.from_points((306, 201), (134, 597))
    assert left.angle_from_vertical_deg == pytest.approx(23.47, abs=0.1)


# ─────────────────────────────────────────────────────────────
# fit_line — TLS is stable where y = m·x + b is not
# ─────────────────────────────────────────────────────────────
def test_fit_line_perfect_vertical():
    # ordinary least squares y=f(x) is undefined here; TLS is not
    pts = [(100.0, y) for y in range(0, 200, 10)]
    line = G.fit_line(pts)
    assert abs(line.ux) == pytest.approx(0.0, abs=1e-9)   # direction is vertical
    assert all(line.distance(p) == pytest.approx(0.0, abs=1e-9) for p in pts)


def test_fit_line_perfect_horizontal():
    pts = [(x, 50.0) for x in range(0, 200, 10)]
    line = G.fit_line(pts)
    assert abs(line.uy) == pytest.approx(0.0, abs=1e-9)


def test_fit_line_recovers_near_vertical_edge_under_noise():
    # sample the A's left outer edge (steep) and perturb ±0.4px alternating
    a, b = (306.0, 201.0), (134.0, 597.0)
    truth = G.Line.from_points(a, b)
    pts = []
    for i in range(21):
        t = i / 20.0
        x = a[0] + t * (b[0] - a[0]) + (0.4 if i % 2 else -0.4)
        y = a[1] + t * (b[1] - a[1])
        pts.append((x, y))
    line = G.fit_line(pts)
    assert line.angle_from_vertical_deg == pytest.approx(truth.angle_from_vertical_deg, abs=0.3)
    assert all(line.distance(p) < 1.0 for p in pts)


def test_fit_line_requires_two_distinct_points():
    with pytest.raises(ValueError):
        G.fit_line([(5.0, 5.0)])
    with pytest.raises(ValueError):
        G.fit_line([(5.0, 5.0), (5.0, 5.0)])


# ─────────────────────────────────────────────────────────────
# intersect — the sub-pixel corner
# ─────────────────────────────────────────────────────────────
def test_intersect_crossing_lines():
    l1 = G.Line.from_points((0, 0), (2, 2))
    l2 = G.Line.from_points((0, 2), (2, 0))
    x, y = G.intersect(l1, l2)
    assert (x, y) == pytest.approx((1.0, 1.0))


def test_intersect_parallel_raises():
    l1 = G.Line.from_points((0, 0), (0, 10))
    l2 = G.Line.from_points((5, 0), (5, 10))
    with pytest.raises(ValueError):
        G.intersect(l1, l2)


def test_intersect_recovers_apex_from_the_two_outer_edges():
    # the A's outer edges, extended, meet at the (virtual, pointed) apex on the axis
    left = G.fit_line([(306, 201), (134, 597)])
    right = G.fit_line([(357, 199), (529, 596)])
    ax, ay = G.intersect(left, right)
    assert ax == pytest.approx(331.5, abs=6.0)   # on the symmetry axis
    assert ay < 201                               # above the truncated flat top


# ─────────────────────────────────────────────────────────────
# symmetry — the constraint the luminance diff is blind to
# ─────────────────────────────────────────────────────────────
def test_reflect_across_vertical():
    assert G.reflect_across_vertical((306.0, 201.0), 331.5) == pytest.approx((357.0, 201.0))


def test_symmetry_axis_is_robust_to_one_bad_pair():
    # median axis is unmoved by the 9px-shifted apex outlier
    axis = G.symmetry_axis_x(GOOD_PAIRS + [APEX_BAD])
    assert axis == pytest.approx(331.5, abs=1.0)


def test_symmetry_report_flags_the_shifted_apex():
    rep = G.symmetry_report(GOOD_PAIRS + [APEX_BAD], tol=2.0)
    assert rep["n_outliers"] == 1
    outliers = [e for e in rep["pairs"] if e["outlier"]]
    assert outliers[0]["mid_x"] == pytest.approx(340.5, abs=0.1)
    assert outliers[0]["axis_dev_px"] == pytest.approx(9.0, abs=1.0)
    # every honest pair sits within tolerance of the axis
    assert all(not e["outlier"] for e in rep["pairs"][:4])


# ─────────────────────────────────────────────────────────────
# collinearity — a leg's inner edge is one straight line
# ─────────────────────────────────────────────────────────────
def test_collinearity_flags_the_notch_kink_and_enforce_fixes_it():
    # c_bl, the eyeballed (11px-off) notch-left, and foot-inner should be collinear
    c_bl, notch_bad, foot_in = (283.0, 438.0), (242.0, 508.0), (216.0, 597.0)
    before = G.collinearity_residual([c_bl, notch_bad, foot_in])
    assert before["max_dist_px"] > 3.0            # the kink is real and detectable

    fixed = G.enforce_collinear([c_bl, notch_bad, foot_in])
    after = G.collinearity_residual(fixed)
    assert after["max_dist_px"] == pytest.approx(0.0, abs=1e-6)
    # the middle point is pulled toward the straight edge (rightward, +x)
    assert fixed[1][0] > notch_bad[0]


# ─────────────────────────────────────────────────────────────
# mirror-slope — the trigonometric symmetry check
# ─────────────────────────────────────────────────────────────
def test_mirror_slope_passes_for_symmetric_edges():
    left = G.fit_line([(306, 201), (134, 597)])
    right = G.fit_line([(357, 199), (529, 596)])
    rep = G.mirror_slope_report(left, right)
    assert rep["symmetric"] is True
    assert rep["delta_deg"] < 0.5


def test_mirror_slope_fails_for_the_shifted_apex():
    # the 9px-off apex makes the two legs unequal in steepness (24.5° vs 22.3°)
    left = G.fit_line([(315, 200), (134, 597)])
    right = G.fit_line([(366, 198), (529, 596)])
    rep = G.mirror_slope_report(left, right, tol_deg=1.0)
    assert rep["symmetric"] is False
    assert rep["delta_deg"] == pytest.approx(2.2, abs=0.4)


# ─────────────────────────────────────────────────────────────
# consistency_report — the bundled metric a luminance diff can't be
# ─────────────────────────────────────────────────────────────
def test_consistency_report_bundles_symmetry_and_collinearity():
    rep = G.consistency_report(
        symmetry_pairs=GOOD_PAIRS + [APEX_BAD],
        collinear_groups=[[(283, 438), (242, 508), (216, 597)]],  # the notch kink
        tol=2.0)
    assert rep["symmetry"]["n_outliers"] == 1
    assert len(rep["collinearity"]) == 1
    # worst_dev is dominated by the 9px apex shift, well over tolerance
    assert rep["worst_dev_px"] >= 8.0
    assert rep["within_tol"] is False


def test_consistency_report_passes_for_clean_geometry():
    # symmetric apex (306/357) + a genuinely collinear edge → within tolerance
    good_apex = ((306.0, 201.0), (357.0, 199.0))
    rep = G.consistency_report(
        symmetry_pairs=[FEET_OUTER, COUNTER_BASE, good_apex],
        collinear_groups=[[(100, 100), (150, 150), (200, 200)]],
        tol=2.0)
    assert rep["within_tol"] is True
    assert rep["worst_dev_px"] <= 2.0


def test_consistency_report_empty_is_trivially_within_tol():
    rep = G.consistency_report(tol=2.0)
    assert rep["within_tol"] is True
    assert rep["worst_dev_px"] == 0.0
