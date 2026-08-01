#!/usr/bin/env python3
"""4K/8K trustworthiness of the reconstruction score — the BHAG proof instrument.

``score_reconstruction`` is how pixel-perfection is *proven*, so it must measure
sub-pixel truth, not sampling artifacts: a geometrically perfect 3840×2160
reconstruction has to score as perfect (no resolution-dependent caps degrading
rich images), a localized 3 px defect has to stay visible (per-shape scores name
the offender instead of drowning in the global mean), and the metric's nonzero
floor has to be named in the payload so absolute gates can calibrate it away.
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

from frameforge_vision.infrastructure import matchscore as MS  # noqa: E402

# Eight 1px-outline rectangles spread over the 4K canvas (PIL pixel-index coords).
_RECTS_4K = [
    (100, 100, 700, 500), (1200, 150, 2200, 600), (2600, 200, 3700, 900),
    (150, 800, 900, 1500), (1400, 900, 2000, 1400), (2500, 1200, 3600, 1900),
    (300, 1700, 1100, 2050), (1600, 1600, 2300, 2000),
]


def _probe_png(size, rects) -> bytes:
    """White canvas with 1px black rect outlines at pixel-index coords ``rects``."""
    from io import BytesIO

    from PIL import Image, ImageDraw

    im = Image.new("L", size, 255)
    d = ImageDraw.Draw(im)
    for x0, y0, x1, y1 in rects:
        d.rectangle([x0, y0, x1, y1], outline=0, width=1)
    buf = BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _exact_shapes(rects):
    """The continuous-coordinate truth of ``rects``: a painted pixel index ``i``
    spans continuous [i, i+1), so the drawn 1px stroke centreline sits at +0.5."""
    return [{"kind": "rect", "points": [[x0 + 0.5, y0 + 0.5], [x1 + 0.5, y1 + 0.5]]}
            for x0, y0, x1, y1 in rects]


# ─────────────────────────────────────────────────────────────
# INSTR-1: no resolution-dependent caps — perfect 4K scores perfect
# ─────────────────────────────────────────────────────────────
def test_perfect_4k_reconstruction_scores_perfect():
    img = _probe_png((3840, 2160), _RECTS_4K)
    s = MS.score(img, _exact_shapes(_RECTS_4K), tol=1.8)
    assert "error" not in s
    # every sample participates — the old max_samples=1500 cap is gone by default
    assert s["n_samples"] > 1500
    # exact geometry reads as exact: every sample within tol of a real edge, and
    # the mean sits at the documented stroke-flank floor (~1 px for a 1px stroke),
    # not at the 1.5-1.7 px the subsampled edge set used to inflate it to.
    assert s["on_edge_frac"] == 1.0
    assert 0.8 <= s["mean_dist"] <= 1.2


def test_perfect_8k_reconstruction_scores_perfect():
    rects = [(200, 200, 3000, 2000), (3600, 500, 7300, 3900), (900, 2600, 2800, 4000)]
    img = _probe_png((7680, 4320), rects)
    s = MS.score(img, _exact_shapes(rects), tol=1.8)
    assert "error" not in s
    assert s["on_edge_frac"] == 1.0
    assert s["mean_dist"] <= 1.2


# ─────────────────────────────────────────────────────────────
# INSTR-6: a localized defect stays visible (per-shape scores name it)
# ─────────────────────────────────────────────────────────────
def test_localized_3px_defect_stays_visible_at_4k():
    img = _probe_png((3840, 2160), _RECTS_4K)
    shapes = _exact_shapes(_RECTS_4K)
    k = 4                                               # one mid-canvas rect is off by 3px
    shapes[k] = {"kind": "rect",
                 "points": [[p[0] + 3.0, p[1] + 3.0] for p in shapes[k]["points"]]}
    s = MS.score(img, shapes, tol=1.8)
    assert "error" not in s
    # the aggregate still moves…
    assert s["on_edge_frac"] < 1.0
    # …and the per-shape breakdown names the offender unambiguously
    assert s["worst_shape"]["index"] == k
    off = s["per_shape"][k]
    assert off["on_edge_frac"] <= 0.1 and off["mean_dist"] >= 1.9
    for j, entry in enumerate(s["per_shape"]):
        if j != k:
            assert entry["on_edge_frac"] == 1.0


def test_per_shape_scores_carry_ids_and_counts():
    img = _probe_png((640, 480), [(50, 50, 300, 200), (350, 250, 600, 430)])
    shapes = _exact_shapes([(50, 50, 300, 200), (350, 250, 600, 430)])
    shapes[1]["id"] = "plate"
    s = MS.score(img, shapes, tol=1.8)
    assert [e["index"] for e in s["per_shape"]] == [0, 1]
    assert s["per_shape"][1]["id"] == "plate"
    assert sum(e["n_samples"] for e in s["per_shape"]) == s["n_samples"]


# ─────────────────────────────────────────────────────────────
# the exact nearest-edge engine must agree with a full brute force
# ─────────────────────────────────────────────────────────────
def test_nearest_edge_distances_match_full_brute_force():
    rects = [(20, 20, 120, 90), (140, 40, 220, 160)]
    img = _probe_png((240, 180), rects)
    shapes = _exact_shapes(rects)
    # one shape deliberately off so far-field distances are exercised too
    shapes.append({"kind": "line", "points": [[10.0, 170.0], [230.0, 170.0]]})
    result, S, d = MS._score_core(img, shapes, tol=2.0)
    assert "error" not in result
    edges = MS._edge_points(MS._gray(img), tuple(result["roi"]))
    brute = np.sqrt(((S[:, None, :] - edges[None, :, :]) ** 2).sum(-1)).min(1)
    assert np.allclose(d, brute, atol=1e-9)


# ─────────────────────────────────────────────────────────────
# INSTR-3: the score floor is named in the payload
# ─────────────────────────────────────────────────────────────
def test_score_payload_names_the_floor():
    img = _probe_png((640, 480), [(50, 50, 300, 200)])
    s = MS.score(img, _exact_shapes([(50, 50, 300, 200)]), tol=1.8)
    assert "floor" in s
    assert s["floor"]["mean_dist_floor_px"] == 0.5
    assert "calibrat" in s["floor"]["note"]             # gates must calibrate, not trust 0
    # the measured floor on this exact-geometry probe sits in the documented band
    assert 0.5 <= s["mean_dist"] <= 1.2


def test_explicit_caps_are_still_honoured():
    # callers may still bound runtime explicitly; thinning is opt-in, not default
    img = _probe_png((640, 480), [(50, 50, 300, 200), (350, 250, 600, 430)])
    s = MS.score(img, _exact_shapes([(50, 50, 300, 200), (350, 250, 600, 430)]),
                 tol=1.8, max_samples=100)
    assert s["n_samples"] == 100
