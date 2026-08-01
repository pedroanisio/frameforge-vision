#!/usr/bin/env python3
"""Coordinate-identity calibration of the measurement instruments.

One convention everywhere (INSTR-2): coordinates are CONTINUOUS, matching the
SDK/SVG doc space — pixel index ``i`` covers ``[i, i+1)`` and its centre is
``i + 0.5``. A crisp edge authored at x = 100.0 must therefore *measure* as
100.0, not 99.5. The same identity discipline applies to the zoom-crop
transform (INSTR-5: the transform must describe the crop actually taken), the
normalized-crop rounding rule (INSTR-8), and the comparison metrics, which must
name the resolution regime they were computed at (INSTR-4).
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

from frameforge_vision.domain import coordinates as C  # noqa: E402
from frameforge_vision.infrastructure import edgesnap as E  # noqa: E402
from frameforge_vision.infrastructure import image_compare as IC  # noqa: E402
from frameforge_vision.infrastructure import matchscore as MS  # noqa: E402


def _step_png(w=200, h=120, edge_x=100, aa=None) -> bytes:
    """Dark columns with continuous index < ``edge_x``, bright at >= ``edge_x``.

    The continuous edge sits exactly at x = ``edge_x``. With ``aa`` set, the column
    at index ``edge_x`` gets the 50%-coverage value instead (continuous edge at
    ``edge_x + 0.5``).
    """
    from io import BytesIO

    from PIL import Image

    arr = np.full((h, w), 20, dtype=np.uint8)
    arr[:, edge_x:] = 235
    if aa is not None:
        arr[:, edge_x] = aa
    buf = BytesIO()
    Image.fromarray(arr, "L").save(buf, "PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# INSTR-2: the pixel-centre convention — an edge at 100.0 reads 100.0
# ─────────────────────────────────────────────────────────────
def test_refine_edge_line_reads_step_edge_at_100_not_99_5():
    img = _step_png(edge_x=100)
    res = E.refine_edge_line(img, (99.0, 10.0), (101.0, 110.0), band=6.0, step=4.0)
    assert res["ok"] is True
    xs = [p[0] for p in res["points"]]
    assert np.mean(xs) == pytest.approx(100.0, abs=0.05)
    assert res["_line"].px == pytest.approx(100.0, abs=0.1)


def test_snap_point_to_edge_reads_step_edge_at_100():
    img = _step_png(edge_x=100)
    res = E.snap_point_to_edge(img, (95.0, 60.0), search_dir=(1.0, 0.0), band=8.0)
    assert res["ok"] is True
    assert res["point_px"][0] == pytest.approx(100.0, abs=0.1)


def test_half_covered_aa_column_reads_100_5():
    # a 50%-covered antialiased column means the continuous edge is at 100.5; the
    # parabolic refiner's profile-shape error stays within ±0.3 px of that
    img = _step_png(edge_x=100, aa=128)
    pts = E.edge_crossings_along(img, (99.0, 10.0), (101.0, 110.0), band=6.0, step=4.0)
    assert len(pts) >= 5
    assert np.mean([p[0] for p in pts]) == pytest.approx(100.5, abs=0.3)


def test_matchscore_edge_points_are_pixel_centres():
    img = _step_png(w=40, h=30, edge_x=10)
    pts = MS._edge_points(MS._gray(img), (0, 0, 40, 30))
    xs = sorted(set(float(x) for x, _ in pts))
    # the Sobel band marks the two flanking pixel columns; their CENTRES straddle
    # the true continuous edge at 10.0 symmetrically
    assert xs == [9.5, 10.5]
    ys = set(float(y) for _, y in pts)
    assert all(y == int(y) + 0.5 for y in ys)


# ─────────────────────────────────────────────────────────────
# INSTR-8: fractional roi is floor/ceil, not silent truncation
# ─────────────────────────────────────────────────────────────
def test_score_fractional_roi_floors_origin_and_ceils_far_corner():
    img = _step_png(w=200, h=120, edge_x=100)
    s = MS.score(img, [{"kind": "line", "points": [[100.0, 20.0], [100.0, 90.0]]}],
                 roi=[10.6, 10.6, 150.2, 110.2])
    assert s["roi"] == [10, 10, 151, 111]


def test_crop_norm_uses_one_rounding_rule_for_all_edges():
    from PIL import Image

    # left edge 10.7 must round to 11 (nearest), matching the far edge's rule —
    # flooring one edge and rounding the other shifted the window by up to 1 px
    box = (0.107, 0.0, 0.5, 1.0)
    small = IC._crop_norm(Image.new("RGB", (100, 80)), box)
    assert small.size == (50, 80)                       # 61 - 11, not 61 - 10
    # the same normalized box on a 10x-larger frame lands within half a source px
    big = IC._crop_norm(Image.new("RGB", (1000, 80)), box)
    assert big.size == (500, 80)


# ─────────────────────────────────────────────────────────────
# INSTR-5: the crop transform describes the crop actually taken
# ─────────────────────────────────────────────────────────────
def test_crop_transform_snaps_origin_size_and_scale_to_the_raster():
    cs = C.CoordinateSystem("top-left", 3840, 2160)
    xf = C.crop_transform("z", (0.333, 0.25, 0.1, 0.1), cs, render_long_edge=1024)
    ox, oy = xf.origin_px
    sw, sh = xf.size_px
    # origin is the floored pixel the crop really starts at (was 1278.72 → bias)
    assert (ox, oy) == (1278.0, 540.0)
    # whole-pixel extent covering the requested box
    assert sw == float(int(sw)) and sh == float(int(sh))
    assert ox + sw >= 0.333 * 3840 + 0.1 * 3840 and oy + sh >= 0.25 * 2160 + 0.1 * 2160
    # integer zoom: render size is an EXACT multiple of the source extent, so the
    # resized raster and to_source_px/to_render_px agree with no residual skew
    assert xf.scale == float(int(xf.scale)) and xf.scale >= 1.0
    assert xf.render_px == (int(sw * xf.scale), int(sh * xf.scale))
    assert xf.to_source_px(*xf.render_px) == (ox + sw, oy + sh)


def test_crop_transform_integer_boxes_are_untouched():
    cs = C.CoordinateSystem("top-left", 200, 100)
    xf = C.crop_transform("q", (0.5, 0.0, 0.5, 0.5), cs, render_long_edge=1000)
    assert xf.origin_px == (100.0, 0.0)
    assert xf.size_px == (100.0, 50.0)
    assert xf.scale == 10.0
    assert xf.render_px == (1000, 500)


# ─────────────────────────────────────────────────────────────
# INSTR-4: comparison metrics name their resolution regime
# ─────────────────────────────────────────────────────────────
def _gradient_image(size=(64, 48)):
    from PIL import Image

    arr = np.tile(np.linspace(0, 255, size[0], dtype=np.uint8), (size[1], 1))
    return Image.fromarray(arr, "L").convert("RGB")


def test_image_metrics_surface_resolution_regime():
    a = _gradient_image()
    down = IC.image_metrics(a, a)
    assert down["metric_regime"] == "downsampled"
    assert down["metric_px"] == [256, 256]
    full = IC.image_metrics(a, a, size=None)
    assert full["metric_regime"] == "full-res"
    assert full["metric_px"] == [64, 48]
    assert full["ncc"] == 1.0 and full["mae"] == 0.0
    aligned = IC.image_metrics(a, a, align=True)
    assert aligned["metric_regime"] == "full-res-aligned"
    assert aligned["metric_px"] == [64, 48]
    assert aligned["shift_px"] == [0, 0]
