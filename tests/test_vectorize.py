#!/usr/bin/env python3
"""Raster → FrameForge vectorizer (the ingestion front-end).

`region` mode quantises colour and traces filled polygons (a flat vector base);
`outline` mode traces edges into polylines (line art). Both emit FrameForge
object dicts that render through the engine. Needs OpenCV (the `vision` group).
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_shadow = sys.modules.get("frameforge")
if _shadow is not None and not hasattr(_shadow, "__path__"):
    del sys.modules["frameforge"]
sys.path[:0] = [ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "docs")]

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from frameforge_vision.infrastructure.vectorize import raster_to_objects  # noqa: E402

from conftest import needs_author  # noqa: E402


def _two_color_image(tmp_path):
    img = np.zeros((80, 120, 3), np.uint8)
    img[:, :60] = (0, 0, 200)      # BGR -> left red
    img[:, 60:] = (200, 0, 0)      # BGR -> right blue
    p = tmp_path / "two.png"
    cv2.imwrite(str(p), img)
    return p


def test_region_mode_traces_filled_polygons(tmp_path):
    objs, w, h = raster_to_objects(_two_color_image(tmp_path), mode="region", colors=2, min_area=20)
    assert (w, h) == (120, 80)
    polys = [o for o in objs if o["type"] == "polygon"]
    assert len(polys) >= 2
    fills = {o["fill"] for o in polys}
    assert len(fills) >= 2                       # both colours recovered
    assert all(o["fill"].startswith("#") for o in polys)


def test_outline_mode_traces_polylines(tmp_path):
    img = np.full((80, 120, 3), 255, np.uint8)
    cv2.rectangle(img, (30, 20), (90, 60), (0, 0, 0), -1)   # high-contrast shape
    p = tmp_path / "box.png"
    cv2.imwrite(str(p), img)
    objs, w, h = raster_to_objects(p, mode="outline")
    assert any(o["type"] == "polyline" for o in objs)
    assert all("stroke" in o for o in objs if o["type"] == "polyline")


@needs_author
def test_region_objects_roundtrip_through_frameforge(tmp_path):
    objs, w, h = raster_to_objects(_two_color_image(tmp_path), mode="region", colors=3, min_area=20)
    from frameforge.sdk import DocumentBuilder, render_page_svgs
    b = DocumentBuilder(title="trace")
    pg = b.page("p", canvas={"size": [w, h], "units": "px"}, coordinate_mode="absolute")
    layer = pg.layer("m")
    for o in objs:
        layer.add(o)
    svg = render_page_svgs(b.build())[0]
    assert svg.startswith("<svg") and "polygon" in svg


def test_detail_knob_reduces_point_count(tmp_path):
    """A coarser `detail` epsilon yields simpler (fewer-point) polygons."""
    img = _two_color_image(tmp_path)
    fine, _, _ = raster_to_objects(img, mode="region", colors=2, min_area=20, detail=0.001)
    coarse, _, _ = raster_to_objects(img, mode="region", colors=2, min_area=20, detail=0.05)
    pts_fine = sum(len(o.get("points", [])) for o in fine)
    pts_coarse = sum(len(o.get("points", [])) for o in coarse)
    assert pts_coarse <= pts_fine
