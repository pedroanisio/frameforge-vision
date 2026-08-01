#!/usr/bin/env python3
"""Sub-pixel edge-snap tests against synthetic images with known edge positions.

Each image places a smooth (logistic) edge at a deliberately non-integer location,
so a pixel-accurate detector would be wrong and only a sub-pixel one passes. This is
the pixel half of the constraint pipeline (the exact maths lives in
``test_vision_geometry.py``).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_shadow = sys.modules.get("frameforge")
if _shadow is not None and not hasattr(_shadow, "__path__"):
    del sys.modules["frameforge"]
sys.path[:0] = [ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "docs")]

import pytest  # noqa: E402

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

from frameforge_vision.infrastructure import edgesnap as E  # noqa: E402

EDGE_X = 80.4     # sub-pixel vertical-edge position (bright to the right)
EDGE_Y = 120.6    # sub-pixel horizontal-edge position (bright above)


def _corner_image(w=200, h=200):
    """Bright in the quadrant x>EDGE_X ∧ y<EDGE_Y — a vertical + horizontal edge.

    Ground truth is authored in CONTINUOUS coordinates (the pixel-centre
    convention edgesnap reads in): pixel index ``i`` carries the field value at
    its centre ``i + 0.5``, so the logistic midpoint — the edge — sits at
    continuous EDGE_X/EDGE_Y exactly.
    """
    from io import BytesIO

    from PIL import Image

    xs = np.arange(w)[None, :] + 0.5                  # pixel centres, continuous
    ys = np.arange(h)[:, None] + 0.5
    sx = 1.0 / (1.0 + np.exp(-(xs - EDGE_X) / 0.8))   # →1 right of EDGE_X
    sy = 1.0 / (1.0 + np.exp((ys - EDGE_Y) / 0.8))    # →1 above EDGE_Y
    img = (255.0 * sx * sy).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(img, "L").save(buf, "PNG")
    return buf.getvalue()


def test_edge_crossings_recover_subpixel_vertical_edge():
    img = _corner_image()
    # rough (and slightly wrong) vertical segment across the edge, away from the corner
    pts = E.edge_crossings_along(img, (78.0, 30.0), (82.0, 100.0), band=8.0, step=4.0)
    assert len(pts) >= 5
    xs = [p[0] for p in pts]
    assert np.mean(xs) == pytest.approx(EDGE_X, abs=0.4)   # sub-pixel, not 80 or 81


def test_refine_edge_line_fits_the_vertical_edge():
    img = _corner_image()
    res = E.refine_edge_line(img, (78.0, 30.0), (82.0, 100.0), band=8.0, step=4.0)
    assert res["ok"] is True
    assert res["n_crossings"] >= 5
    assert res["rms_residual_px"] < 0.5
    line = res["_line"]
    assert abs(line.ux) == pytest.approx(0.0, abs=0.05)   # near-vertical
    assert line.px == pytest.approx(EDGE_X, abs=0.4)


def test_refine_corner_intersects_two_edges():
    img = _corner_image()
    vert = ((80.0, 40.0), (80.0, 95.0))       # rough vertical edge (y < EDGE_Y)
    horiz = ((100.0, 120.0), (160.0, 120.0))  # rough horizontal edge (x > EDGE_X)
    res = E.refine_corner(img, vert, horiz, band=8.0, step=4.0)
    assert res["ok"] is True
    cx, cy = res["corner_px"]
    assert cx == pytest.approx(EDGE_X, abs=0.6)
    assert cy == pytest.approx(EDGE_Y, abs=0.6)


def test_snap_point_to_edge_default_gradient_direction():
    img = _corner_image()
    # a point 5px left of the vertical edge snaps onto it (perpendicular)
    res = E.snap_point_to_edge(img, (75.0, 70.0), band=10.0)
    assert res["ok"] is True
    assert res["point_px"][0] == pytest.approx(EDGE_X, abs=0.5)
    assert res["moved_px"] == pytest.approx(5.4, abs=0.6)


def test_snap_point_to_edge_explicit_search_dir():
    img = _corner_image()
    res = E.snap_point_to_edge(img, (75.0, 70.0), search_dir=(1.0, 0.0), band=10.0)
    assert res["ok"] is True
    assert res["point_px"][0] == pytest.approx(EDGE_X, abs=0.5)


def test_snap_reports_failure_on_flat_region():
    img = _corner_image()
    # deep in the bright quadrant, far from any edge → no confident crossing
    res = E.snap_point_to_edge(img, (150.0, 40.0), search_dir=(1.0, 0.0), band=6.0)
    assert res["ok"] is False


def test_refine_edge_line_reports_failure_when_no_edge():
    img = _corner_image()
    # a segment wholly inside the flat bright quadrant crosses no edge
    res = E.refine_edge_line(img, (140.0, 30.0), (160.0, 30.0), band=6.0, step=4.0)
    assert res["ok"] is False
    assert res["n_crossings"] < 2
