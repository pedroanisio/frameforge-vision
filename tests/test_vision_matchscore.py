#!/usr/bin/env python3
"""Closed-loop reconstruction scoring — the NUMERIC convergence signal.

``compare_images`` shows a human/vision-model *where* a reconstruction is off;
``matchscore`` answers *how far* the constructed vector geometry sits from the
source image's real edges, so an agent can converge over passes on a number, not
just an eyeball. These tests pin the pure shape sampling and the edge-match score
(shapes ON an edge score high, shapes OFF it score low), plus the structured errors.
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

pytest.importorskip("PIL")

from frameforge_vision.infrastructure import matchscore as MS  # noqa: E402


def _png_bytes(draw_fn, size=(240, 180), bg=(255, 255, 255)) -> bytes:
    from io import BytesIO

    from PIL import Image, ImageDraw

    im = Image.new("RGB", size, bg)
    draw_fn(ImageDraw.Draw(im))
    buf = BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# pure shape sampling (no numpy)
# ─────────────────────────────────────────────────────────────
def test_sample_line_walks_the_segment():
    pts = MS.sample_shape({"kind": "line", "points": [[0, 0], [10, 0]]}, spacing=2.0)
    assert all(abs(y) < 1e-9 for _, y in pts)            # stays on y=0
    assert min(x for x, _ in pts) == pytest.approx(0.0)
    assert max(x for x, _ in pts) == pytest.approx(10.0)


def test_sample_rect_covers_all_four_edges():
    pts = MS.sample_shape({"kind": "rect", "points": [[0, 0], [10, 6]]}, spacing=2.0)
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    assert min(xs) == pytest.approx(0.0) and max(xs) == pytest.approx(10.0)
    assert min(ys) == pytest.approx(0.0) and max(ys) == pytest.approx(6.0)


def test_sample_circle_from_centre_and_radius():
    pts = MS.sample_shape({"kind": "circle", "points": [[50, 50]], "r": 10}, spacing=2.0)
    assert pts, "circle produced no samples"
    for x, y in pts:
        assert math.hypot(x - 50, y - 50) == pytest.approx(10.0, abs=0.6)


def test_sample_polygon_is_closed():
    pts = MS.sample_shape({"kind": "triangle", "points": [[0, 0], [10, 0], [5, 8]]}, spacing=3.0)
    # a closed shape returns to its start, so the first vertex appears at both ends of a loop
    assert (0.0, 0.0) in [(round(x, 6), round(y, 6)) for x, y in pts]


# ─────────────────────────────────────────────────────────────
# edge-match score (numpy)
# ─────────────────────────────────────────────────────────────
def test_shape_on_an_edge_scores_high_off_edge_scores_low():
    pytest.importorskip("numpy")
    img = _png_bytes(lambda d: d.line([(0, 90), (240, 90)], fill=(0, 0, 0), width=2))
    on = MS.score(img, [{"kind": "line", "points": [[20, 90], [220, 90]]}], tol=2.0)
    off = MS.score(img, [{"kind": "line", "points": [[20, 20], [220, 20]]}], tol=2.0)
    assert on["on_edge_frac"] > 0.9 and on["mean_dist"] < 2.0
    assert off["on_edge_frac"] < 0.2
    assert off["mean_dist"] > on["mean_dist"]           # off-edge is measurably worse


def test_score_reports_counts_and_roi():
    pytest.importorskip("numpy")
    img = _png_bytes(lambda d: d.rectangle([40, 40, 200, 140], outline=(0, 0, 0), width=2))
    s = MS.score(img, [{"kind": "rect", "points": [[40, 40], [200, 140]]}], tol=3.0)
    assert s["n_samples"] > 0 and s["n_edges"] > 0
    assert s["roi"] == [0, 0, 240, 180]
    assert 0.0 <= s["on_edge_frac"] <= 1.0


def test_score_structured_errors():
    img = _png_bytes(lambda d: d.line([(0, 90), (240, 90)], fill=(0, 0, 0), width=2))
    assert "error" in MS.score(img, [])                 # no shapes
    blank = _png_bytes(lambda d: None)                  # flat white → no edges
    r = MS.score(blank, [{"kind": "line", "points": [[10, 10], [200, 10]]}])
    assert "error" in r


def test_build_score_overlay_returns_image_and_score():
    pytest.importorskip("numpy")
    img = _png_bytes(lambda d: d.line([(0, 90), (240, 90)], fill=(0, 0, 0), width=2))
    overlay, s = MS.build_score_overlay(img, [{"kind": "line", "points": [[20, 90], [220, 90]]}])
    assert overlay.size == (240, 180)                   # coordinate identity (source size)
    assert s["on_edge_frac"] > 0.9


def test_build_score_overlay_on_error_returns_base_image_and_error():
    # scoring error (no shapes) still returns a same-size image so a render can be written
    img = _png_bytes(lambda d: d.line([(0, 90), (240, 90)], fill=(0, 0, 0), width=2))
    overlay, s = MS.build_score_overlay(img, [])
    assert overlay.size == (240, 180)
    assert "error" in s and "on_edge_frac" not in s


# ─────────────────────────────────────────────────────────────
# curve/spline sampling must match what construct DRAWS (open, fixed tension)
# ─────────────────────────────────────────────────────────────
def test_sample_curve_ignores_undocumented_closed_and_tension():
    """construct draws curve/spline as an OPEN polyline with the SDK's fixed Catmull-Rom;
    sampling must be identical regardless of an (undocumented) closed/tension field, so
    the scored geometry equals the drawn geometry."""
    pts = [[0, 0], [10, 5], [20, 0], [30, 5]]
    base = MS.sample_shape({"kind": "curve", "points": pts})
    assert MS.sample_shape({"kind": "curve", "points": pts, "tension": 1.2}) == base
    assert MS.sample_shape({"kind": "curve", "points": pts, "closed": True}) == base


def test_sample_ellipse_star_polygon_unknown():
    # ellipse from a bbox → points on the rim
    ell = MS.sample_shape({"kind": "ellipse", "points": [[0, 0], [40, 20]]}, spacing=2.0)
    assert ell and all(abs(((x - 20) / 20) ** 2 + ((y - 10) / 10) ** 2 - 1.0) < 0.1 for x, y in ell)
    # star has two alternating radii
    star = MS.sample_shape({"kind": "star", "points": [[50, 50]], "r": 20,
                            "points_count": 5, "inner_ratio": 0.4}, spacing=3.0)
    rad = [round(math.hypot(x - 50, y - 50)) for x, y in star]
    assert max(rad) == pytest.approx(20, abs=1) and min(rad) <= 9      # inner ≈ 20*0.4
    # open path stays open; unknown kind samples nothing
    assert MS.sample_shape({"kind": "path", "points": [[0, 0], [10, 0], [10, 10]]})
    assert MS.sample_shape({"kind": "blob", "points": [[0, 0]]}) == []


# ─────────────────────────────────────────────────────────────
# roi windowing + tol gating
# ─────────────────────────────────────────────────────────────
def test_roi_windows_samples_and_reports_errors():
    pytest.importorskip("numpy")
    # a rectangle outline → edges on all four sides
    img = _png_bytes(lambda d: d.rectangle([20, 20, 220, 160], outline=(0, 0, 0), width=2))
    rect = [{"kind": "rect", "points": [[20, 20], [220, 160]]}]
    full = MS.score(img, rect)
    windowed = MS.score(img, rect, roi=[0, 0, 130, 180])           # left half only
    assert windowed["roi"] == [0, 0, 130, 180]
    assert windowed["n_samples"] < full["n_samples"]               # samples cropped to roi
    # a flat corner (no edges in the roi) → structured error
    assert "no edges" in MS.score(img, rect, roi=[60, 60, 180, 120])["error"]
    # edges in the roi but the shape is entirely outside it → structured error
    assert "no shape samples" in MS.score(
        img, [{"kind": "line", "points": [[30, 40], [210, 40]]}], roi=[0, 120, 240, 180])["error"]


def test_tol_gates_the_on_edge_fraction():
    pytest.importorskip("numpy")
    img = _png_bytes(lambda d: d.line([(0, 90), (240, 90)], fill=(0, 0, 0), width=1))
    off = [{"kind": "line", "points": [[20, 84], [220, 84]]}]      # ~6px off the edge
    assert MS.score(img, off, tol=1.0)["on_edge_frac"] == 0.0
    loose = MS.score(img, off, tol=8.0)
    assert loose["on_edge_frac"] == 1.0 and loose["tol"] == 8.0
