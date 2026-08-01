#!/usr/bin/env python3
"""SVG -> FrameForge object importer (the ingestion back-end).

Any tool that emits SVG (a vectorizer, Inkscape, Illustrator) lowers into
FrameForge primitives through svg_to_objects, which the renderer then draws 1:1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_shadow = sys.modules.get("frameforge")
if _shadow is not None and not hasattr(_shadow, "__path__"):
    del sys.modules["frameforge"]
sys.path[:0] = [ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "docs")]

from frameforge_vision.infrastructure.svg_import import svg_to_objects  # noqa: E402

from conftest import needs_author  # noqa: E402


def test_circle_lowers_to_ellipse_with_fill():
    objs = svg_to_objects('<svg viewBox="0 0 10 10"><circle cx="5" cy="6" r="3" fill="#abc"/></svg>')
    assert len(objs) == 1
    o = objs[0]
    assert o["type"] == "ellipse" and o["center"] == [5.0, 6.0]
    assert o["rx"] == 3.0 and o["ry"] == 3.0 and o["fill"] == "#abc"


def test_rect_inherits_group_fill():
    objs = svg_to_objects('<svg viewBox="0 0 20 20"><g fill="#00ff00">'
                          '<rect x="1" y="2" width="4" height="6" rx="2"/></g></svg>')
    r = objs[0]
    assert r["type"] == "rect" and r["box"] == [1.0, 2.0, 4.0, 6.0]
    assert r["fill"] == "#00ff00" and r["radius"] == 2.0


def test_polygon_points_parsed():
    objs = svg_to_objects('<svg viewBox="0 0 10 10"><polygon points="0,0 5,0 5,5" fill="#222"/></svg>')
    assert objs[0]["type"] == "polygon" and objs[0]["points"] == [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]


def test_path_passthrough_and_boxfit_transform():
    objs = svg_to_objects('<svg viewBox="0 0 100 100"><path d="M0 0 L10 0 Z" fill="#111"/></svg>',
                          box=[200, 100, 50, 50])
    o = objs[0]
    assert o["type"] == "path" and o["d"] == "M0 0 L10 0 Z"
    tr = o["style"]["transform"]
    assert "scale(0.5" in tr and "translate(200" in tr


def test_stroke_only_line():
    objs = svg_to_objects('<svg viewBox="0 0 10 10"><line x1="0" y1="0" x2="9" y2="1" '
                          'stroke="#f00" stroke-width="2"/></svg>')
    o = objs[0]
    assert o["type"] == "line" and o["from"] == [0.0, 0.0] and o["to"] == [9.0, 1.0]
    assert o["stroke"] == "#f00" and o["stroke_style"]["stroke_width"] == 2.0


def test_gradient_url_fill_falls_back_not_passed_through():
    objs = svg_to_objects('<svg viewBox="0 0 10 10"><rect x="0" y="0" width="10" height="10" '
                          'fill="url(#grad)"/></svg>')
    assert not objs[0]["fill"].startswith("url(")  # url() can't resolve -> neutral fallback


_GRAD_SVG = (
    '<svg viewBox="0 0 10 10"><defs>'
    '<linearGradient id="g" x1="0" y1="0" x2="10" y2="0">'
    '<stop offset="0" stop-color="#ff0000"/><stop offset="1" stop-color="#0000ff"/>'
    "</linearGradient></defs>"
    '<rect width="10" height="10" fill="url(#g)"/></svg>'
)


def test_linear_gradient_url_resolves_to_fg_gradient():
    g = svg_to_objects(_GRAD_SVG)[0]["fill"]
    assert isinstance(g, dict) and g["kind"] == "linear"
    assert [s["color"] for s in g["stops"]] == ["#ff0000", "#0000ff"]
    assert g["stops"][0]["position"] == "0%" and g["stops"][1]["position"] == "100%"
    assert round(g["angle"]) == 90          # left -> right vector == 90deg in CSS


def test_vertical_gradient_angle_is_to_bottom():
    svg = _GRAD_SVG.replace('x1="0" y1="0" x2="10" y2="0"', 'x1="0" y1="0" x2="0" y2="10"')
    assert round(svg_to_objects(svg)[0]["fill"]["angle"]) == 180


def test_undefined_gradient_still_falls_back_to_grey():
    o = svg_to_objects('<svg viewBox="0 0 10 10"><rect width="10" height="10" '
                       'fill="url(#missing)"/></svg>')[0]
    assert o["fill"] == "#C7CCD6"


@needs_author
def test_resolved_gradient_renders_a_gradient():
    objs = svg_to_objects(_GRAD_SVG)
    from frameforge.sdk import DocumentBuilder, render_page_svgs
    b = DocumentBuilder(title="grad")
    pg = b.page("p", canvas={"size": [10, 10], "units": "px"}, coordinate_mode="absolute")
    layer = pg.layer("m")
    for o in objs:
        layer.add(o)
    svg = render_page_svgs(b.build())[0]
    assert "gradient" in svg.lower()        # the renderer actually paints it


def test_data_attrs_opt_in_carries_into_meta():
    svg = ('<svg viewBox="0 0 10 10"><path d="M0 0 L1 1" fill="#111" '
           'data-class="foreground" data-id="7"/></svg>')
    assert "meta" not in svg_to_objects(svg)[0]                      # default: dropped
    o = svg_to_objects(svg, data_attrs=True)[0]
    assert o["meta"]["data"] == {"class": "foreground", "id": "7"}   # opt-in: under meta.data


def test_data_attrs_no_data_leaves_no_meta():
    svg = '<svg viewBox="0 0 10 10"><rect width="4" height="4" fill="#111"/></svg>'
    assert "meta" not in svg_to_objects(svg, data_attrs=True)[0]


@needs_author
def test_real_corpus_svg_roundtrips_through_frameforge():
    """A real multi-path vector imports and renders through FrameForge's engine."""
    objs = svg_to_objects(Path(ROOT) / "tests/fixtures/corpus/vector/wikimedia-nasa-logo.svg",
                          box=[0, 0, 200, 200])
    assert len(objs) >= 40  # circles (stars) + the 11 paths
    from frameforge.sdk import DocumentBuilder, render_page_svgs
    b = DocumentBuilder(title="import")
    pg = b.page("p", canvas={"size": [200, 200], "units": "px"}, coordinate_mode="absolute")
    layer = pg.layer("m")
    for o in objs:
        layer.add(o)
    svg = render_page_svgs(b.build())[0]
    assert svg.startswith("<svg") and len(svg) > 1000
