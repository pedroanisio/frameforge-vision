#!/usr/bin/env python3
"""The pure coordinate + fitting domain — exact, PIL-free, exhaustively testable.

These pin the invariants the raster→vector tools depend on (coordinate identity,
frame round-trips, the crop inverse, the rotation-free similarity fit) at the domain
level, so a change to the maths fails here in milliseconds instead of surfacing as a
mystery in a PIL-heavy end-to-end tool test. They also assert the infrastructure
modules re-export these exact objects, so the extraction stays a single source of
truth (no silent second copy).
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

from frameforge_vision.domain import coordinates as C  # noqa: E402
from frameforge_vision.domain import fitting as F  # noqa: E402


class _Region:
    """Duck-typed stand-in for the infra Region value object (name + normalized box)."""

    def __init__(self, name, box):
        self.name = name
        self.box = box


# ─────────────────────────────────────────────────────────────
# denorm / normalize primitives (the DUP #1 consolidation target)
# ─────────────────────────────────────────────────────────────
def test_denorm_point_and_normalize_point_are_inverses():
    assert C.denorm_point(0.5, 0.25, 200, 100) == (100.0, 25.0)
    assert C.normalize_point(100, 25, 200, 100) == (0.5, 0.25)
    for nx, ny in [(0.0, 0.0), (1.0, 1.0), (0.3, 0.7)]:
        px, py = C.denorm_point(nx, ny, 640, 480)
        assert C.normalize_point(px, py, 640, 480) == pytest.approx((nx, ny))


def test_normalize_point_zero_dimension_guard():
    assert C.normalize_point(10, 10, 0, 0) == (0.0, 0.0)


def test_denorm_point_is_unclamped():
    # a POINT spec must not clamp — an out-of-bounds point stays out of bounds
    assert C.denorm_point(1.5, -0.2, 200, 100) == (300.0, -20.0)


def test_denorm_box_clamps_to_the_unit_square():
    # a BOX overrunning an edge is clamped before denormalizing (approx: 0.9*200 is
    # 180.00000000000003 in float — the same imprecision the original code rounds away)
    assert C.denorm_box(0.9, 0.0, 0.5, 1.0, 200, 100) == pytest.approx((180.0, 0.0, 20.0, 100.0))
    assert C.denorm_box(-0.1, 0.0, 0.4, 0.5, 200, 100) == pytest.approx((0.0, 0.0, 60.0, 50.0))


def test_resolve_norm_stays_unclamped_after_consolidation():
    cs = C.CoordinateSystem("top-left", 200, 100)
    # regression guard: routing resolve_point_spec's norm branch through denorm_point
    # must keep it UNCLAMPED (the box path clamps; the point path must not).
    assert C.resolve_point_spec({"norm": [1.5, -0.2]}, cs, {}, None) == (300.0, -20.0)


# ─────────────────────────────────────────────────────────────
# CoordinateSystem
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("origin", ["top-left", "bottom-left", "center", "nonsense"])
def test_to_cs_from_cs_are_exact_inverses(origin):
    cs = C.CoordinateSystem(origin=origin, width=200, height=100)
    for px, py in [(0, 0), (200, 100), (73.5, 12.25), (200, 0), (0, 100)]:
        cx, cy = cs.to_cs(px, py)
        rx, ry = cs.from_cs(cx, cy)
        assert rx == pytest.approx(px) and ry == pytest.approx(py)


def test_origin_semantics():
    cs = C.CoordinateSystem("bottom-left", 200, 100)
    assert cs.to_cs(0, 100) == (0, 0)          # bottom-left origin
    assert cs.to_cs(0, 0) == (0, 100)          # top of image = y=height
    c = C.CoordinateSystem("center", 200, 100)
    assert c.to_cs(100, 50) == (0, 0)          # image centre = CS origin
    tl = C.CoordinateSystem("top-left", 200, 100)
    assert tl.to_cs(10, 20) == (10, 20)        # identity


def test_unknown_origin_falls_back_to_top_left_and_never_raises():
    cs = C.CoordinateSystem("weird", 200, 100)
    assert cs.normalized() == "top-left"
    assert cs.y_up is False
    assert cs.to_cs(5, 5) == (5, 5)
    assert cs.describe()["origin"] == "top-left"


# ─────────────────────────────────────────────────────────────
# CropTransform
# ─────────────────────────────────────────────────────────────
def test_crop_transform_inverse_roundtrip():
    cs = C.CoordinateSystem("top-left", 1000, 800)
    xf = C.crop_transform("Z", (0.1, 0.2, 0.3, 0.25), cs, render_long_edge=1024)
    # to_render_px and to_source_px are exact inverses
    for px, py in [(120, 180), (350.5, 400.0), (100, 160)]:
        rx, ry = xf.to_render_px(px, py)
        sx, sy = xf.to_source_px(rx, ry)
        assert sx == pytest.approx(px) and sy == pytest.approx(py)
    # origin sits at the (clamped) normalized box corner
    assert xf.origin_px == (pytest.approx(100.0), pytest.approx(160.0))
    assert xf.scale >= 1.0


def test_crop_transform_scale_matches_long_edge():
    cs = C.CoordinateSystem("top-left", 1000, 1000)
    xf = C.crop_transform("Z", (0.0, 0.0, 0.5, 0.25), cs, render_long_edge=1000)
    # long edge is 0.5*1000 = 500 px → scale = 1000/500 = 2
    assert xf.scale == pytest.approx(2.0)
    assert xf.render_px[0] == 1000  # 500 * 2


# ─────────────────────────────────────────────────────────────
# regions + structural landmarks
# ─────────────────────────────────────────────────────────────
def test_measured_regions_denormalize_and_clamp():
    cs = C.CoordinateSystem("top-left", 200, 100)
    regs = C.measured_regions([_Region("mark", (0.1, 0.2, 0.5, 0.5)),
                               _Region("edge", (0.9, 0.0, 0.5, 1.0))], cs)
    assert regs[0].id == "R1" and regs[0].name == "mark"
    assert regs[0].bbox_px == (pytest.approx(20.0), pytest.approx(20.0),
                               pytest.approx(100.0), pytest.approx(50.0))
    assert regs[0].centroid_px == (pytest.approx(70.0), pytest.approx(45.0))
    # box overrunning the edge is clamped to [0,1] before denormalizing (w=0.5 → 0.1*W)
    assert regs[1].bbox_px[2] == pytest.approx(0.1 * 200)


def test_structural_landmarks_are_nine_exact_anchors():
    cs = C.CoordinateSystem("top-left", 400, 300)
    lms = C.structural_landmarks(cs)
    assert [lm.id for lm in lms] == [f"A{i}" for i in range(1, 10)]
    by_id = {lm.id: lm for lm in lms}
    assert (by_id["A1"].x_px, by_id["A1"].y_px) == (0.0, 0.0)          # tl corner
    assert (by_id["A4"].x_px, by_id["A4"].y_px) == (400.0, 300.0)      # br corner
    assert (by_id["A9"].x_px, by_id["A9"].y_px) == (200.0, 150.0)      # centre
    assert all(lm.source == "structural" and lm.confidence == 1.0 for lm in lms)


# ─────────────────────────────────────────────────────────────
# resolve_point_spec — every frame + errors
# ─────────────────────────────────────────────────────────────
def test_resolve_point_spec_all_frames():
    cs = C.CoordinateSystem("bottom-left", 200, 100)
    lms = {lm.id: lm for lm in C.structural_landmarks(cs)}
    vp = C.crop_transform("V", (0.5, 0.5, 0.5, 0.5), cs)  # origin (100,50), scale...
    assert C.resolve_point_spec({"px": [10, 20]}, cs, lms, None) == (10.0, 20.0)
    assert C.resolve_point_spec({"norm": [0.5, 0.5]}, cs, lms, None) == (100.0, 50.0)
    # cs origin bottom-left: cs (0,0) → image (0, height)
    assert C.resolve_point_spec({"cs": [0, 0]}, cs, lms, None) == (0.0, 100.0)
    # landmark A9 (centre) + delta
    assert C.resolve_point_spec({"landmark": "A9", "dx": 5, "dy": -3}, cs, lms, None) == (105.0, 47.0)
    # viewport_px maps back to source via the crop
    px = C.resolve_point_spec({"viewport_px": [0, 0]}, cs, lms, vp)
    assert px == (pytest.approx(vp.origin_px[0]), pytest.approx(vp.origin_px[1]))


def test_resolve_point_spec_errors():
    cs = C.CoordinateSystem("top-left", 100, 100)
    with pytest.raises(ValueError):
        C.resolve_point_spec("notadict", cs, {}, None)
    with pytest.raises(ValueError):
        C.resolve_point_spec({"landmark": "ZZ"}, cs, {}, None)
    with pytest.raises(ValueError):
        C.resolve_point_spec({"viewport_px": [1, 2]}, cs, {}, None)  # no viewport
    with pytest.raises(ValueError):
        C.resolve_point_spec({"mystery": [1, 2]}, cs, {}, None)


# ─────────────────────────────────────────────────────────────
# point_frames — resolve → report round-trip
# ─────────────────────────────────────────────────────────────
def test_point_frames_reports_every_frame_and_roundtrips():
    cs = C.CoordinateSystem("top-left", 1000, 800)
    vp = C.crop_transform("V", (0.1, 0.1, 0.4, 0.4), cs)
    px, py = 300.0, 250.0
    fr = C.point_frames(px, py, cs, vp)
    assert fr["image_px"] == [300.0, 250.0]
    assert fr["normalized"] == [round(0.3, 6), round(0.3125, 6)]
    # viewport_px is exactly the crop's forward transform, and inverts back to source
    vx, vy = fr["viewport"]["viewport_px"]
    sx, sy = vp.to_source_px(vx, vy)
    assert sx == pytest.approx(px) and sy == pytest.approx(py)


# ─────────────────────────────────────────────────────────────
# fitting
# ─────────────────────────────────────────────────────────────
def test_fit_requires_a_pair():
    with pytest.raises(ValueError):
        F.fit_similarity([])


def test_single_pair_is_pure_translation():
    t = F.fit_similarity([((10, 20), (0, 0))])
    assert (t.scale, t.tx, t.ty) == (1.0, 10.0, 20.0)


def test_fit_recovers_known_scale_and_translation():
    # base = 2*overlay + (10, 5)
    ov = [(0, 0), (100, 0), (100, 100), (0, 100)]
    pairs = [((2 * x + 10, 2 * y + 5), (x, y)) for (x, y) in ov]
    t = F.fit_similarity(pairs)
    assert t.scale == pytest.approx(2.0)
    assert (t.tx, t.ty) == (pytest.approx(10.0), pytest.approx(5.0))
    assert F.rms_residual(F.landmark_offsets(pairs, t)) == 0.0


def test_degenerate_overlay_falls_back_to_scale_one():
    pairs = [((0, 0), (5, 5)), ((10, 10), (5, 5))]  # overlay points coincide
    t = F.fit_similarity(pairs)
    assert t.scale == 1.0


def test_rotation_shows_up_as_large_residual_not_silent_fix():
    # a 90° rotation cannot be absorbed by scale+translation
    ov = [(0, 0), (10, 0), (10, 10), (0, 10)]
    rot = [(-y, x) for (x, y) in ov]
    pairs = list(zip(rot, ov))
    t = F.fit_similarity(pairs)
    assert F.rms_residual(F.landmark_offsets(pairs, t)) > 0.5


def test_offset_vs_residual_semantics():
    pairs = [((60, 60), (10, 10))]  # single pair → translation (50,50), residual 0
    t = F.fit_similarity(pairs)
    off = F.landmark_offsets(pairs, t)[0]
    assert off["offset_px"] == [50.0, 50.0]        # raw base-overlay gap
    assert off["residual_px"] == [0.0, 0.0]        # after the fit
    assert F.rms_residual([]) == 0.0


# ─────────────────────────────────────────────────────────────
# extraction integrity — infra re-exports the exact domain objects
# ─────────────────────────────────────────────────────────────
def test_measure_reexports_the_domain_authority():
    from frameforge_vision.infrastructure import measure as M
    for name in ["CoordinateSystem", "CropTransform", "MeasuredRegion", "Landmark",
                 "measured_regions", "structural_landmarks", "crop_transform",
                 "resolve_point_spec", "point_frames", "nice_step"]:
        assert getattr(M, name) is getattr(C, name), f"measure.{name} is not the domain object"


def test_overlay_align_reexports_the_domain_fitting():
    from frameforge_vision.infrastructure import overlay_align as O
    for name in ["Similarity", "fit_similarity", "landmark_offsets", "rms_residual"]:
        assert getattr(O, name) is getattr(F, name), f"overlay_align.{name} is not the domain object"
