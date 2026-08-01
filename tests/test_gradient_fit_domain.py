#!/usr/bin/env python3
"""RED → domain contract for raster→gradient paint fitting (Gap 1 closure).

``frameforge_vision.domain.gradient_fit`` is the pure fitting authority: path
flattening, signed area, the CSS-angle mapping (including the potrace-ingest
y-flip composition), and ``fit_paint`` — flat/linear/radial candidates ranked
by like-for-like colour rms (the ``fit_primitives`` doctrine: a richer family
must beat the simpler one above the noise floor, never by default).

Everything here is synthetic + deterministic: no PIL, no OpenCV, no raster I/O.
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


# ---------------------------------------------------------------- path flatten
def test_flatten_path_d_square_with_close():
    from frameforge_vision.domain.gradient_fit import flatten_path_d

    subs = flatten_path_d("M0 0 L10 0 L10 10 L0 10 Z")
    assert len(subs) == 1
    pts = subs[0]
    assert pts[0] == (0.0, 0.0)
    assert (10.0, 10.0) in pts
    assert pts[-1] == pts[0], "Z must close the subpath back to its start"


def test_flatten_path_d_relative_cubic_reaches_endpoint():
    from frameforge_vision.domain.gradient_fit import flatten_path_d

    # potrace idiom: absolute M then relative c; endpoint must land exactly.
    subs = flatten_path_d("M10 10 c 0 20 30 20 30 0")
    (pts,) = subs
    assert pts[0] == (10.0, 10.0)
    assert pts[-1] == pytest.approx((40.0, 10.0))
    assert len(pts) > 3, "curves are sampled, not endpoint-only"


def test_flatten_path_d_implicit_lineto_and_two_subpaths():
    from frameforge_vision.domain.gradient_fit import flatten_path_d

    subs = flatten_path_d("M0 0 10 0 10 10 Z M20 20 L30 20 L30 30 Z")
    assert len(subs) == 2
    assert (10.0, 0.0) in subs[0], "coordinate pairs after M are implicit linetos"
    assert subs[1][0] == (20.0, 20.0)


def test_shoelace_sign_and_magnitude():
    from frameforge_vision.domain.gradient_fit import shoelace

    ccw = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert shoelace(ccw) == pytest.approx(100.0)
    assert shoelace(list(reversed(ccw))) == pytest.approx(-100.0)


# ---------------------------------------------------------------- angle mapping
def test_css_angle_image_space_no_flip():
    from frameforge_vision.domain.gradient_fit import css_angle

    # Renderer convention (painters/svg.py): 0 = up, 90 = right, y-down space.
    assert css_angle(0.0, 1.0) == pytest.approx(180.0)
    assert css_angle(1.0, 0.0) == pytest.approx(90.0)
    assert css_angle(0.0, -1.0) % 360.0 == pytest.approx(0.0)
    assert css_angle(-1.0, 0.0) == pytest.approx(270.0)


def test_css_angle_composed_through_ingest_y_flip():
    from frameforge_vision.domain.gradient_fit import css_angle

    # potrace ingest wraps paths in scale(sx, -sy): the object-local bbox space
    # is y-flipped, so the same image-space stop direction needs the composed
    # mapping atan2(dx, dy).
    assert css_angle(0.0, 1.0, y_flipped=True) % 360.0 == pytest.approx(0.0)
    assert css_angle(1.0, 0.0, y_flipped=True) == pytest.approx(90.0)
    assert css_angle(0.0, -1.0, y_flipped=True) == pytest.approx(180.0)
    assert css_angle(-1.0, 0.0, y_flipped=True) == pytest.approx(270.0)


# ---------------------------------------------------------------- fit_paint
def _grid(w, h, step=1):
    return [(float(x), float(y)) for y in range(0, h, step) for x in range(0, w, step)]


def test_fit_paint_recovers_linear_ramp():
    from frameforge_vision.domain.gradient_fit import fit_paint

    pts = _grid(80, 30)
    colors = [(int(20 + 200 * (x / 79.0)), 40, 90) for x, _ in pts]
    out = fit_paint(pts, colors)

    assert out["family"] == "linear"
    fill = out["fill"]
    assert fill["kind"] == "linear"
    stops = fill["stops"]
    assert len(stops) >= 3
    angle = float(fill["angle"])
    first_r = int(stops[0]["color"].lstrip("#")[0:2], 16)
    last_r = int(stops[-1]["color"].lstrip("#")[0:2], 16)
    # Axis sign is free: toward +x with brightening stops, or toward -x with
    # darkening stops — both draw the same ramp.
    if math.isclose(angle, 90.0, abs_tol=8.0):
        assert last_r > first_r + 120
    elif math.isclose(angle, 270.0, abs_tol=8.0):
        assert first_r > last_r + 120
    else:
        raise AssertionError(f"expected a horizontal axis, got angle={angle}")
    # Endpoint stops recover the source extremes.
    assert min(first_r, last_r) == pytest.approx(20, abs=30)
    assert max(first_r, last_r) == pytest.approx(220, abs=30)
    assert out["rms"]["linear"] < out["rms"]["flat"]


def test_fit_paint_prefers_radial_on_radial_field():
    from frameforge_vision.domain.gradient_fit import fit_paint

    pts = [(float(x), float(y)) for y in range(60) for x in range(60)
           if (x - 30) ** 2 + (y - 30) ** 2 <= 28 * 28]
    colors = []
    for x, y in pts:
        r = math.hypot(x - 30, y - 30) / 28.0
        v = int(240 - 190 * r)
        colors.append((v, v // 2, 200))
    out = fit_paint(pts, colors)

    assert out["family"] == "radial"
    fill = out["fill"]
    assert fill["kind"] == "radial"
    fx, fy = fill["at"]
    assert fx == pytest.approx(0.5, abs=0.08)
    assert fy == pytest.approx(0.5, abs=0.08)
    # Centre-out ordering: the first stop is the bright core.
    first_r = int(fill["stops"][0]["color"].lstrip("#")[0:2], 16)
    last_r = int(fill["stops"][-1]["color"].lstrip("#")[0:2], 16)
    assert first_r > last_r + 100
    assert out["rms"]["radial"] < out["rms"]["linear"]


def test_fit_paint_flat_on_uniform_and_below_min_pixels():
    from frameforge_vision.domain.gradient_fit import fit_paint

    pts = _grid(40, 40)
    out = fit_paint(pts, [(90, 140, 200)] * len(pts))
    assert out["family"] == "flat"
    assert out["fill"] == "#5a8cc8"

    tiny = fit_paint([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
                     [(0, 0, 0), (120, 120, 120), (250, 250, 250)],
                     min_pixels=400)
    assert tiny["family"] == "flat", "below min_pixels the fit degrades to flat"


def test_fit_paint_y_flip_changes_only_the_angle():
    from frameforge_vision.domain.gradient_fit import fit_paint

    pts = _grid(24, 90)
    colors = [(int(15 + 220 * (y / 89.0)), 30, 30) for _, y in pts]
    plain = fit_paint(pts, colors)
    flipped = fit_paint(pts, colors, y_flipped=True)
    assert plain["family"] == flipped["family"] == "linear"
    a, b = float(plain["fill"]["angle"]) % 360.0, float(flipped["fill"]["angle"]) % 360.0
    assert abs((a - b) % 360.0) == pytest.approx(180.0, abs=10.0)
    assert [s["color"] for s in plain["fill"]["stops"]] == \
           [s["color"] for s in flipped["fill"]["stops"]]


def test_fit_paint_is_deterministic():
    from frameforge_vision.domain.gradient_fit import fit_paint

    pts = _grid(50, 50)
    colors = [(int(10 + 3 * x % 240), int(y * 2 % 250), 77) for x, y in pts]
    assert fit_paint(pts, colors) == fit_paint(pts, colors)


# ------------------------------------------------- A1 user-space geometry lane
# geometry="user" emits the EXACT fitted geometry (model A1 fields) instead of
# the bbox-relative approximation: linear → `line` endpoints at the projection
# span's ends (image space), radial → px `at` + `radius`. The css_angle y-flip
# composition becomes irrelevant — the object transform carries orientation.
def test_fit_paint_user_linear_emits_exact_line():
    from frameforge_vision.domain.gradient_fit import fit_paint

    pts = _grid(80, 30)
    colors = [(int(20 + 200 * (x / 79.0)), 40, 90) for x, _ in pts]
    out = fit_paint(pts, colors, geometry="user")

    fill = out["fill"]
    assert fill["kind"] == "linear" and "angle" not in fill
    (x1, y1), (x2, y2) = fill["line"]
    assert abs(x2 - x1) > 10 * abs(y2 - y1), "the ramp axis is horizontal"
    xs = sorted((x1, x2))
    assert xs[0] == pytest.approx(0.0, abs=2.0)
    assert xs[1] == pytest.approx(79.0, abs=2.0)
    assert y1 == pytest.approx(14.5, abs=2.0) and y2 == pytest.approx(14.5, abs=2.0)


def test_fit_paint_user_radial_emits_px_centre_and_radius():
    from frameforge_vision.domain.gradient_fit import fit_paint

    pts = [(float(x), float(y)) for y in range(60) for x in range(60)
           if (x - 30) ** 2 + (y - 30) ** 2 <= 28 * 28]
    colors = []
    for x, y in pts:
        r = math.hypot(x - 30, y - 30) / 28.0
        v = int(240 - 190 * r)
        colors.append((v, v // 2, 200))
    out = fit_paint(pts, colors, geometry="user")

    fill = out["fill"]
    assert fill["kind"] == "radial"
    assert fill["at"][0] == pytest.approx(30.0, abs=2.0)
    assert fill["at"][1] == pytest.approx(30.0, abs=2.0)
    assert fill["radius"] == pytest.approx(28.0, abs=2.5)


def test_fit_paint_unknown_geometry_is_loud():
    from frameforge_vision.domain.gradient_fit import fit_paint

    with pytest.raises(ValueError, match="geometry"):
        fit_paint(_grid(10, 10), [(1, 2, 3)] * 100, geometry="mesh")


def test_apply_gradient_fills_lowers_user_line_into_local_space():
    """The wiring converts image-space fitted geometry into the OBJECT's local
    space (inverse translate/scale), so the renderer's userSpaceOnUse gradient
    lands back exactly on the source pixels — including through the potrace
    y-flip transform."""
    pytest.importorskip("PIL")
    from PIL import Image

    from frameforge_vision.infrastructure.vectorize import apply_gradient_fills

    img = Image.new("RGB", (120, 60))
    px = img.load()
    for y in range(60):
        for x in range(120):
            t = x / 119.0
            px[x, y] = (int(10 + 235 * t), 50, 120)

    # one path covering x∈[10,110), y∈[10,50) — potrace-style y-flip transform:
    # local = ((img_x)/0.1, (60 - img_y)/0.1)
    obj = {"type": "path",
           "d": "M100 100 L1100 100 L1100 500 L100 500 Z",
           "fill": "#808080",
           "style": {"transform": "translate(0.000000,60.000000) scale(0.100000,-0.100000)"}}
    summary = apply_gradient_fills([obj], img, min_pixels=50)
    assert summary["fitted"] == 1
    fill = obj["fill"]
    assert fill["kind"] == "linear"
    (x1, y1), (x2, y2) = fill["line"]
    # map local → image and check the ramp axis span
    def to_img(x, y):
        return 0.1 * x + 0.0, -0.1 * y + 60.0
    ix1, iy1 = to_img(x1, y1)
    ix2, iy2 = to_img(x2, y2)
    lo, hi = sorted((ix1, ix2))
    assert lo == pytest.approx(12.0, abs=3.0) and hi == pytest.approx(107.0, abs=3.0)
    assert iy1 == pytest.approx(iy2, abs=1.0)
