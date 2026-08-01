#!/usr/bin/env python3
"""Region analysis — the promoted R&D region-detection capability.

``frameforge_vision.infrastructure.regions`` consolidates the root R&D scripts
(closed_region_detector / region_fill / region_preprocess / consensus_regions /
unique_regions and out/region_fill/flat_regions) into one module with a single
canonical ``DetectedRegion`` type and a JSON-serializable ``detect_regions``
funnel. These tests pin the three methods (closed / flat / consensus) on small
synthetic images, the shape-equivalence clustering, the overlay artifact, the
preprocess helpers (mollify / smooth_loop / green_area), and the root-script
deprecation shims.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_shadow = sys.modules.get("frameforge")
if _shadow is not None and not hasattr(_shadow, "__path__"):
    del sys.modules["frameforge"]
sys.path[:0] = [ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "docs")]

import pytest  # noqa: E402

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from frameforge_vision.infrastructure import regions as RG  # noqa: E402


# ─────────────────────────────────────────────────────────────
# synthetic fixtures (deterministic, no I/O beyond tmp_path)
# ─────────────────────────────────────────────────────────────
def _lineart(path=None):
    """White canvas with a thin black rectangle OUTLINE (one enclosed white face)."""
    img = np.full((120, 160, 3), 255, np.uint8)
    cv2.rectangle(img, (30, 20), (120, 90), (0, 0, 0), 2)
    if path is not None:
        cv2.imwrite(str(path), img)
    return img


def _flat_art(path=None):
    """White canvas with two solid red squares (same shape) + one solid blue disc."""
    img = np.full((140, 200, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 20), (50, 50), (40, 40, 200), -1)     # red square (BGR)
    cv2.rectangle(img, (80, 20), (110, 50), (40, 40, 200), -1)    # translated copy
    cv2.circle(img, (160, 90), 22, (200, 60, 40), -1)             # blue disc
    if path is not None:
        cv2.imwrite(str(path), img)
    return img


# ─────────────────────────────────────────────────────────────
# closed method (topological enclosed faces)
# ─────────────────────────────────────────────────────────────
def test_closed_method_finds_the_enclosed_face(tmp_path):
    src = tmp_path / "lineart.png"
    _lineart(src)
    out = RG.detect_regions(str(src), method="closed", min_area=50)
    assert out["ok"] is True and out["method"] == "closed"
    closed = [r for r in out["regions"] if r["closed"]]
    assert len(closed) == 1
    r = closed[0]
    x, y, w, h = r["bbox_px"]
    assert 30 < x < 40 and 20 < y < 30           # inside the outline
    assert r["area_px"] > 1000
    assert r["fill_hex"].startswith("#")
    assert r["kind"] == "enclosed"
    # the surrounding background is an open region
    assert any(not r["closed"] for r in out["regions"])


def test_closed_method_polygon_and_norm_come_from_coordinate_authority(tmp_path):
    src = tmp_path / "lineart.png"
    _lineart(src)
    out = RG.detect_regions(str(src), method="closed", min_area=50)
    r = [r for r in out["regions"] if r["closed"]][0]
    assert len(r["polygon"]) >= 3
    W, H = out["image"]["width_px"], out["image"]["height_px"]
    x, y, w, h = r["bbox_px"]
    bn = r["box_norm"]
    assert bn[0] == pytest.approx(x / W, abs=1e-6)
    assert bn[3] == pytest.approx(h / H, abs=1e-6)


# ─────────────────────────────────────────────────────────────
# flat method (fill partition: every uniform-fill area is a region)
# ─────────────────────────────────────────────────────────────
def test_flat_method_partitions_solid_fills(tmp_path):
    src = tmp_path / "flat.png"
    _flat_art(src)
    out = RG.detect_regions(str(src), method="flat", colors=3, min_area=100)
    assert out["ok"] is True
    solids = [r for r in out["regions"] if r["kind"] == "solid"]
    assert len(solids) == 3                       # two squares + disc
    reds = [r for r in solids if r["fill_rgb"][0] > 150]
    assert len(reds) == 2
    for r in reds:
        assert r["closed"] is True
        assert abs(r["bbox_px"][2] - 31) <= 2 and abs(r["bbox_px"][3] - 31) <= 2
    # the white ground touches the border → open + hollow
    ground = max(out["regions"], key=lambda r: r["area_px"])
    assert ground["kind"] == "hollow" and ground["closed"] is False


def test_flat_method_fill_erode_flags_outline_strokes(tmp_path):
    src = tmp_path / "lineart.png"
    _lineart(src)
    out = RG.detect_regions(str(src), method="flat", min_area=100, fill_erode=5)
    kinds = {r["kind"] for r in out["regions"]}
    assert "outline" in kinds                     # the 3px rectangle stroke
    assert all(r["kind"] != "solid" for r in out["regions"]
               if r["kind"] == "outline")


# ─────────────────────────────────────────────────────────────
# consensus method (ensemble level sets → smooth boundaries)
# ─────────────────────────────────────────────────────────────
def test_consensus_method_returns_smooth_regions_with_ensemble_meta(tmp_path):
    src = tmp_path / "flat.png"
    _flat_art(src)
    out = RG.detect_regions(str(src), method="consensus", sigmas=(3.0, 5.0),
                            levels=(0.3, 0.4), agree=0.5, min_area=150)
    assert out["ok"] is True
    assert out["ensemble"]["members"] == 4 and out["ensemble"]["agree"] == 0.5
    assert out["region_count"] >= 2               # squares blur together or stay apart; disc separate
    r = out["regions"][0]
    assert len(r["polygon"]) > 32                 # smooth loop, not a raw bbox
    assert r["kind"] == "smooth"
    assert r["area_px"] > 0


def test_consensus_regions_svg_helper_emits_paths(tmp_path):
    src = tmp_path / "flat.png"
    _flat_art(src)
    analysis = RG.consensus_smooth_regions(_flat_art(), sigmas=(3.0, 5.0),
                                           levels=(0.3, 0.4), agree=0.5, min_area=150)
    svg = RG.smooth_regions_svg(analysis)
    assert svg.startswith("<svg") and "<path" in svg


# ─────────────────────────────────────────────────────────────
# clustering (unique_regions capability)
# ─────────────────────────────────────────────────────────────
def test_translation_clustering_groups_the_two_squares(tmp_path):
    src = tmp_path / "flat.png"
    _flat_art(src)
    out = RG.detect_regions(str(src), method="flat", colors=3, min_area=100,
                            cluster="translation", cluster_tol=0.85)
    assert out["classes"], "clustering requested but no classes returned"
    sizes = sorted((c["count"] for c in out["classes"]), reverse=True)
    assert sizes[0] >= 2                          # the two identical squares
    clustered = [r for r in out["regions"] if r.get("shape_class") is not None]
    assert len(clustered) >= 3


def test_congruent_clustering_runs(tmp_path):
    src = tmp_path / "flat.png"
    _flat_art(src)
    out = RG.detect_regions(str(src), method="flat", colors=3, min_area=100,
                            cluster="congruent", cluster_tol=0.85)
    assert out["classes"] and out["region_count"] >= 3


# ─────────────────────────────────────────────────────────────
# funnel contract: JSON, overlay, errors
# ─────────────────────────────────────────────────────────────
def test_detect_regions_result_is_json_serializable(tmp_path):
    src = tmp_path / "flat.png"
    _flat_art(src)
    out = RG.detect_regions(str(src), method="flat", colors=3, min_area=100)
    json.dumps(out)                               # must not raise


def test_detect_regions_writes_overlay_png(tmp_path):
    src = tmp_path / "flat.png"
    _flat_art(src)
    overlay = tmp_path / "seg.png"
    out = RG.detect_regions(str(src), method="flat", colors=3, min_area=100,
                            overlay_path=str(overlay))
    assert out["overlay_path"] == str(overlay)
    assert overlay.exists() and overlay.stat().st_size > 0


def test_detect_regions_unknown_method_raises():
    with pytest.raises(ValueError, match="method"):
        RG.detect_regions("nope.png", method="quantum")


def test_detect_regions_unknown_tunable_raises(tmp_path):
    src = tmp_path / "flat.png"
    _flat_art(src)
    with pytest.raises(TypeError, match="tunable"):
        RG.detect_regions(str(src), method="flat", not_a_knob=1)


def test_detect_regions_unreadable_image_raises(tmp_path):
    with pytest.raises(ValueError, match="read"):
        RG.detect_regions(str(tmp_path / "missing.png"), method="closed")


# ─────────────────────────────────────────────────────────────
# preprocess helpers (region_preprocess capability)
# ─────────────────────────────────────────────────────────────
def test_green_area_matches_shoelace_on_a_square():
    P = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert RG.green_area(P) == pytest.approx(100.0)


def test_smooth_loop_stays_near_the_source_loop():
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    P = np.column_stack([50 + 20 * np.cos(t), 50 + 20 * np.sin(t)])
    S = RG.smooth_loop(P, harmonics=8)
    assert S is not None and len(S) == 512
    radii = np.hypot(S[:, 0] - 50, S[:, 1] - 50)
    assert abs(float(radii.mean()) - 20.0) < 1.0


def test_mollify_is_normalized_to_unit_max():
    f = RG.mollify(_lineart(), sigma=4.0)
    assert float(f.max()) == pytest.approx(1.0, abs=1e-5)
    assert float(f.min()) >= 0.0


def test_solid_ink_regions_keeps_fills_drops_thin_outlines():
    img = _flat_art()
    cv2.rectangle(img, (10, 100, ), (60, 130), (0, 0, 0), 2)      # thin outline box
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    labels, kept, stats = RG.solid_ink_regions(gray, fill_erode=5, min_area=80)
    # the solid squares + disc survive erosion; the 2px outline does not
    assert len(kept) == 3


# ─────────────────────────────────────────────────────────────
# root-script deprecation shims
# ─────────────────────────────────────────────────────────────



def test_regions_cli_prints_json_summary(tmp_path, capsys):
    src = tmp_path / "flat.png"
    _flat_art(src)
    rc = RG.main([str(src), "--method", "flat", "--colors", "3", "--min-area", "100"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["method"] == "flat" and out["region_count"] >= 3


def test_consensus_include_polygons_false_omits_polygons(tmp_path):
    """include_polygons must be honored by the DEFAULT (consensus) method too:
    the payload ships no polygon vertex lists when the caller asked to omit them."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from frameforge_vision.infrastructure.regions import detect_regions

    img = np.full((90, 120, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (50, 50), (30, 30, 30), thickness=-1)
    cv2.rectangle(img, (70, 40), (100, 70), (30, 30, 30), thickness=-1)

    result = detect_regions(img, "consensus", include_polygons=False)
    assert result["region_count"] >= 1
    for region in result["regions"]:
        assert region["polygon"] is None
        assert "hole_polygons" not in region
    with_poly = detect_regions(img, "consensus", include_polygons=True)
    assert any(r["polygon"] for r in with_poly["regions"])
