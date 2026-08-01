"""SVG → FrameForge object importer (the ingestion back-end).

Lowers an SVG document into FrameForge primitive object dicts (``rect`` /
``ellipse`` / ``line`` / ``polyline`` / ``polygon`` / ``path`` / ``text``) that
the renderer draws 1:1. This is the universal entry point for vector ingestion:
anything that emits SVG — the raster vectorizer, Inkscape, Illustrator, VTracer —
flows in here.

Dependency-free (stdlib ``xml`` only); emits plain dicts, imports no frameforge
model, so it stays a leaf the package boundary is happy with.

Resolves: solid ``fill``/``stroke`` (with group inheritance), CSS ``<style>``
class/tag/id rules and inline ``style=`` attributes, ``<use>``/``<defs>``
instancing (with a cycle guard), gradients (including ``xlink:href`` stop
inheritance and ``gradientTransform`` on linear gradients), single-shape
``clipPath`` references (lowered to the model's ``style.clip_path`` spec),
``<text>`` (lowered to a text object with an ESTIMATED box — glyph metrics are
the renderer's, not this parser's), ``transform`` passthrough, and an optional
box-fit scale.

Honest limits: pattern paints fall back to a neutral grey, ``mask`` and
multi-shape/objectBoundingBox clipPaths are dropped (the object still imports,
unclipped), radial ``gradientTransform`` is ignored (the radial lowering is
centred anyway), CSS selectors beyond ``tag`` / ``.class`` / ``#id`` (and the
``tag.class`` / ``tag#id`` compounds) are ignored, and ``<tspan>`` positioning is
collapsed into the parent ``<text>`` run.
"""
from __future__ import annotations

import base64
import math
import re
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional, Sequence

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_FILL_SHAPES = {"rect", "circle", "ellipse", "polygon", "path"}
_FALLBACK_FILL = "#C7CCD6"
_URL_REF = re.compile(r"url\(\s*['\"]?#([^)'\"]+)['\"]?\s*\)")
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
# containers that define resources — never emitted (or walked) directly
_SKIP_TAGS = {"defs", "symbol", "clipPath", "mask", "style", "linearGradient",
              "radialGradient", "pattern", "marker", "filter", "metadata",
              "title", "desc", "script"}
# the CSS/presentation properties this importer consumes
_KEEP_PROPS = ("fill", "stroke", "stroke-width", "font-size", "font-family",
               "text-anchor", "clip-path")
_MAX_USE_DEPTH = 32


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _decode_svg_data_uri(uri: str) -> str:
    """Decode a ``data:image/svg+xml`` URI (plain, URL-encoded, or base64)."""
    header, sep, payload = uri.partition(",")
    if not sep:
        raise ValueError("data:image/svg+xml URI has no ',' payload separator")
    if ";base64" in header:
        try:
            return base64.b64decode(payload, validate=True).decode("utf-8")
        except Exception as exc:
            raise ValueError(
                "data:image/svg+xml URI payload is not valid base64") from exc
    return urllib.parse.unquote(payload)


def _load(svg: "str | Path") -> str:
    if isinstance(svg, Path):
        return svg.read_text(encoding="utf-8")
    s = str(svg)
    if s.startswith("data:"):
        if not s.startswith("data:image/svg+xml"):
            raise ValueError(
                f"unsupported data URI (expected image/svg+xml): {s[:48]!r}")
        return _decode_svg_data_uri(s)
    if s.lstrip().startswith("<"):
        return s
    return Path(s).read_text(encoding="utf-8")


def _f(v: Optional[str], default: float = 0.0) -> float:
    if v is None:
        return default
    m = re.match(_NUM, v.strip())
    return float(m.group()) if m else default


def _href(el) -> Optional[str]:
    ref = el.get(_XLINK_HREF) or el.get("href")
    if ref and ref.startswith("#"):
        return ref[1:]
    return None


def _viewbox(root) -> Optional[tuple[float, float, float, float]]:
    vb = root.get("viewBox")
    if vb:
        p = [float(x) for x in re.split(r"[ ,]+", vb.strip()) if x]
        if len(p) == 4:
            return (p[0], p[1], p[2], p[3])
    w, h = root.get("width"), root.get("height")
    if w and h:
        return (0.0, 0.0, _f(w), _f(h))
    return None


def _points(s: str) -> list[list[float]]:
    nums = [float(x) for x in re.findall(_NUM, s or "")]
    return [[nums[i], nums[i + 1]] for i in range(0, len(nums) - 1, 2)]


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ─────────────────────────────────────────────────────────────
# affine transforms (for gradientTransform)
# ─────────────────────────────────────────────────────────────
_TRANSFORM_RE = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)")
_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _mat_mul(M, N):
    a1, b1, c1, d1, e1, f1 = M
    a2, b2, c2, d2, e2, f2 = N
    return (a1 * a2 + c1 * b2, b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1)


def _affine(transform: Optional[str]):
    """SVG transform list → one ``(a, b, c, d, e, f)`` matrix (SVG column order)."""
    M = _IDENTITY
    for name, args in _TRANSFORM_RE.findall(transform or ""):
        v = [float(x) for x in re.findall(_NUM, args)]
        if name == "matrix" and len(v) >= 6:
            N = tuple(v[:6])
        elif name == "translate" and v:
            N = (1.0, 0.0, 0.0, 1.0, v[0], v[1] if len(v) > 1 else 0.0)
        elif name == "scale" and v:
            sx = v[0]
            sy = v[1] if len(v) > 1 else sx
            N = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == "rotate" and v:
            a = math.radians(v[0])
            ca, sa = math.cos(a), math.sin(a)
            N = (ca, sa, -sa, ca, 0.0, 0.0)
            if len(v) >= 3:
                cx, cy = v[1], v[2]
                N = _mat_mul(_mat_mul((1, 0, 0, 1, cx, cy), N), (1, 0, 0, 1, -cx, -cy))
        elif name == "skewX" and v:
            N = (1.0, 0.0, math.tan(math.radians(v[0])), 1.0, 0.0, 0.0)
        elif name == "skewY" and v:
            N = (1.0, math.tan(math.radians(v[0])), 0.0, 1.0, 0.0, 0.0)
        else:
            continue
        M = _mat_mul(M, N)
    return M


def _apply_affine(M, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = M
    return (a * x + c * y + e, b * x + d * y + f)


# ─────────────────────────────────────────────────────────────
# CSS <style> + inline style resolution
# ─────────────────────────────────────────────────────────────
def _decls(body: Optional[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (body or "").split(";"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k, v = k.strip().lower(), v.strip()
        if k and v:
            out[k] = v
    return out


def _css_rules(root) -> list[tuple[str, dict[str, str], int]]:
    """Flatten every ``<style>`` block into ``(selector, decls, order)`` rules."""
    rules: list[tuple[str, dict[str, str], int]] = []
    order = 0
    for el in root.iter():
        if _localname(el.tag) != "style" or not el.text:
            continue
        for sel_group, body in re.findall(r"([^{}]+)\{([^}]*)\}", el.text):
            decls = {k: v for k, v in _decls(body).items() if k in _KEEP_PROPS}
            if not decls:
                continue
            for sel in sel_group.split(","):
                sel = sel.strip()
                if sel:
                    rules.append((sel, decls, order))
                    order += 1
    return rules


def _sel_match(sel: str, name: str, classes: set[str], eid: Optional[str]) -> tuple[bool, int]:
    """(matches?, specificity) for the supported selector forms."""
    if sel.startswith("."):
        return sel[1:] in classes, 10
    if sel.startswith("#"):
        return sel[1:] == (eid or ""), 100
    if "." in sel:
        tag, cls = sel.split(".", 1)
        return (tag == name and cls in classes), 11
    if "#" in sel:
        tag, i = sel.split("#", 1)
        return (tag == name and i == (eid or "")), 101
    return sel == name, 1


def _effective_props(el, rules) -> dict[str, str]:
    """The element's consumed properties, cascaded: presentation attribute <
    matching CSS rule (by specificity, then order) < inline ``style=``."""
    props: dict[str, str] = {}
    for k in _KEEP_PROPS:
        v = el.get(k)
        if v is not None:
            props[k] = v
    if rules:
        name = _localname(el.tag)
        classes = set((el.get("class") or "").split())
        eid = el.get("id")
        matched = []
        for sel, decls, order in rules:
            ok, spec = _sel_match(sel, name, classes, eid)
            if ok:
                matched.append((spec, order, decls))
        for _, _, decls in sorted(matched, key=lambda t: (t[0], t[1])):
            props.update(decls)
    inline = {k: v for k, v in _decls(el.get("style")).items() if k in _KEEP_PROPS}
    props.update(inline)
    return props


# ─────────────────────────────────────────────────────────────
# gradients (href stop inheritance + gradientTransform)
# ─────────────────────────────────────────────────────────────
def _stop_position(offset: Optional[str]) -> str:
    """SVG ``offset`` (``0``..``1`` or ``%``) → FrameForge ``position`` percentage."""
    o = (offset or "0").strip()
    if o.endswith("%"):
        return o
    v = _f(o, 0.0)
    return f"{(v * 100 if v <= 1 else v):g}%"


def _stop_color(stop) -> str:
    color = (stop.get("stop-color") or "#000000").strip()
    op = stop.get("stop-opacity")
    if op is not None:
        alpha = _f(op, 1.0)
        if alpha < 1.0 and color.startswith("#"):
            r, g, b = _hex_rgb(color)
            return f"rgba({r},{g},{b},{alpha:g})"
    return color


def _gradient_stops(el, by_id, depth: int = 0) -> list[dict[str, Any]]:
    """The gradient's stops, following ``xlink:href`` inheritance when it has none."""
    stops = [{"color": _stop_color(s), "position": _stop_position(s.get("offset"))}
             for s in el if _localname(s.tag) == "stop"]
    if stops:
        return stops
    ref = _href(el)
    if ref and depth < 8:
        parent = by_id.get(ref)
        if parent is not None and _localname(parent.tag) in ("linearGradient", "radialGradient"):
            return _gradient_stops(parent, by_id, depth + 1)
    return []


def _gradient_to_fg(el, by_id) -> Optional[dict[str, Any]]:
    """Convert an SVG ``<linear/radialGradient>`` to a FrameForge ``Gradient`` paint.

    Honest limits: ``userSpaceOnUse`` coordinates are reduced to a CSS gradient line
    (a direction ``angle`` for linear; a centred ``at`` for radial), which matches
    when the gradient vector spans the painted region — the raster-fit-per-region
    case. ``gradientTransform`` is applied to the linear endpoints before the angle
    is derived; the centred radial lowering ignores it.
    """
    stops = _gradient_stops(el, by_id)
    if not stops:
        return None
    if _localname(el.tag) == "radialGradient":
        return {"kind": "radial", "stops": stops, "at": "50% 50%", "shape": "circle"}
    x1, y1 = _f(el.get("x1"), 0.0), _f(el.get("y1"), 0.0)
    x2, y2 = _f(el.get("x2"), 1.0), _f(el.get("y2"), 0.0)
    gt = el.get("gradientTransform")
    if gt:
        M = _affine(gt)
        x1, y1 = _apply_affine(M, x1, y1)
        x2, y2 = _apply_affine(M, x2, y2)
    angle = math.degrees(math.atan2(x2 - x1, -(y2 - y1))) % 360.0   # 0=up, 90=right (CSS)
    return {"kind": "linear", "stops": stops, "angle": round(angle, 2)}


def _collect_gradients(root, by_id) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for el in root.iter():
        if _localname(el.tag) in ("linearGradient", "radialGradient"):
            gid = el.get("id")
            if not gid:
                continue
            grad = _gradient_to_fg(el, by_id)
            if grad is not None:
                out[gid] = grad
    return out


# ─────────────────────────────────────────────────────────────
# clipPath lowering (single-shape → model style.clip_path spec)
# ─────────────────────────────────────────────────────────────
def _clip_spec_of(el) -> Optional[dict[str, Any]]:
    """A ``clipPath`` element → the model's ``{"shape", "args"}`` spec, or None.

    Only single-shape, user-space clipPaths lower cleanly; anything else is
    reported as unresolvable (the caller drops the clip, keeps the object).
    """
    if (el.get("clipPathUnits") or "userSpaceOnUse") == "objectBoundingBox":
        return None
    children = [c for c in el if _localname(c.tag) in
                ("rect", "circle", "ellipse", "polygon", "path")]
    if len(children) != 1:
        return None
    c = children[0]
    name = _localname(c.tag)
    if name == "rect":
        return {"shape": "inset",
                "args": {"box": [_f(c.get("x")), _f(c.get("y")),
                                 _f(c.get("width")), _f(c.get("height"))]}}
    if name == "circle":
        return {"shape": "circle",
                "args": {"center": [_f(c.get("cx")), _f(c.get("cy"))], "r": _f(c.get("r"))}}
    if name == "ellipse":
        return {"shape": "ellipse",
                "args": {"center": [_f(c.get("cx")), _f(c.get("cy"))],
                         "rx": _f(c.get("rx")), "ry": _f(c.get("ry"))}}
    if name == "polygon":
        pts = _points(c.get("points", ""))
        return {"shape": "polygon", "args": {"points": pts}} if pts else None
    if name == "path" and c.get("d"):
        return {"shape": "path", "args": {"d": c.get("d")}}
    return None


def _collect_clip_specs(root) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for el in root.iter():
        if _localname(el.tag) == "clipPath" and el.get("id"):
            spec = _clip_spec_of(el)
            if spec is not None:
                out[el.get("id")] = spec
    return out


# ─────────────────────────────────────────────────────────────
# paint + shape emission
# ─────────────────────────────────────────────────────────────
def _clean_paint(value: Optional[str], gradients: dict[str, dict[str, Any]]) -> Any:
    if value is None:
        return None
    v = value.strip()
    m = _URL_REF.match(v)
    if m:                              # url(#id): resolve a known gradient, else fall back
        return gradients.get(m.group(1), _FALLBACK_FILL)
    if v.startswith("url("):
        return _FALLBACK_FILL          # patterns / unparsable refs can't resolve here
    return v


def _emit(name: str, el) -> Optional[dict[str, Any]]:
    if name == "rect":
        obj: dict[str, Any] = {"type": "rect",
                               "box": [_f(el.get("x")), _f(el.get("y")),
                                       _f(el.get("width")), _f(el.get("height"))]}
        r = el.get("rx") or el.get("ry")
        if r:
            obj["radius"] = _f(r)
        return obj
    if name == "circle":
        r = _f(el.get("r"))
        return {"type": "ellipse", "center": [_f(el.get("cx")), _f(el.get("cy"))], "rx": r, "ry": r}
    if name == "ellipse":
        return {"type": "ellipse", "center": [_f(el.get("cx")), _f(el.get("cy"))],
                "rx": _f(el.get("rx")), "ry": _f(el.get("ry"))}
    if name == "line":
        return {"type": "line", "from": [_f(el.get("x1")), _f(el.get("y1"))],
                "to": [_f(el.get("x2")), _f(el.get("y2"))]}
    if name == "polyline":
        return {"type": "polyline", "points": _points(el.get("points", ""))}
    if name == "polygon":
        return {"type": "polygon", "points": _points(el.get("points", ""))}
    if name == "path" and el.get("d"):
        return {"type": "path", "d": el.get("d")}
    return None


def _emit_text(el, props: dict[str, str], fill: Optional[str]) -> Optional[dict[str, Any]]:
    """``<text>`` → a FrameForge text object with an ESTIMATED box.

    The SVG ``(x, y)`` is a baseline anchor; the box top is approximated at
    ``y − font_size`` and the width at ``0.6 · font_size`` per character
    (``text-anchor`` shifts it). Exact metrics belong to the renderer's font
    engine — treat the box as a placement draft, not measured geometry.
    """
    content = " ".join("".join(el.itertext()).split())
    if not content:
        return None
    fs = _f(props.get("font-size"), 16.0)
    x, y = _f(el.get("x")), _f(el.get("y"))
    est_w = max(fs * 0.6 * len(content), fs * 0.6)
    anchor = (props.get("text-anchor") or "start").strip()
    if anchor == "middle":
        x -= est_w / 2.0
    elif anchor == "end":
        x -= est_w
    style: dict[str, Any] = {"font_size": fs,
                             "color": fill if fill and fill != "none" else "#000000"}
    fam = props.get("font-family")
    if fam:
        style["font_family"] = [p.strip().strip("'\"") for p in fam.split(",") if p.strip()]
    return {"type": "text", "box": [x, y - fs, est_w, fs * 1.3],
            "text": content, "style": style}


def _apply_paint(obj: dict[str, Any], name: str, fill, stroke, sw, gradients) -> None:
    fill = _clean_paint(fill, gradients)
    stroke = _clean_paint(stroke, gradients)
    if name in _FILL_SHAPES:
        obj["fill"] = fill if fill is not None else "#000000"   # SVG default fill is black
    if stroke and stroke != "none":
        obj["stroke"] = stroke
        obj["stroke_style"] = {"stroke_width": _f(sw, 1.0)}


def svg_to_objects(svg: "str | Path", *, box: Optional[Sequence[float]] = None,
                   data_attrs: bool = False) -> list[dict[str, Any]]:
    """Return FrameForge object dicts for every drawable element in ``svg``.

    ``svg`` is SVG text, a path to a ``.svg`` file, or a ``data:image/svg+xml``
    URI (plain, URL-encoded, or base64 payload). ``box`` (``[x, y, w, h]``),
    when given, fits the document's viewBox into that box (centered, aspect
    preserved) via a ``style.transform`` applied to every object; element/group
    ``transform`` attributes are composed on top.

    ``data_attrs`` (opt-in) carries an element's ``data-*`` attributes onto the
    emitted object as ``meta['data']`` (keyed without the ``data-`` prefix), so
    upstream semantics — e.g. ``data-class="foreground"`` from a segmenting
    rebuild — survive ingestion and can drive region selection. Only leaf
    (drawable) elements are carried; ``data-*`` on a ``<g>`` is not propagated.
    Default ``False`` keeps the output byte-identical to a plain ingest.
    """
    root = ET.fromstring(_load(svg))
    by_id = {el.get("id"): el for el in root.iter() if el.get("id")}
    rules = _css_rules(root)
    gradients = _collect_gradients(root, by_id)
    clip_specs = _collect_clip_specs(root)
    vb = _viewbox(root)
    boxfit = ""
    if box is not None and vb is not None:
        vx, vy, vw, vh = vb
        bx, by, bw, bh = box
        s = min(bw / vw, bh / vh) if vw and vh else 1.0
        tx = bx + (bw - vw * s) / 2 - vx * s
        ty = by + (bh - vh * s) / 2 - vy * s
        boxfit = f"translate({tx:.3f} {ty:.3f}) scale({s:.5f})"

    out: list[dict[str, Any]] = []

    def finish(obj: dict[str, Any], el, cur: dict[str, Any],
               clip_spec: Optional[dict[str, Any]]) -> None:
        if data_attrs:
            data = {k[5:]: v for k, v in el.attrib.items() if k.startswith("data-")}
            if data:
                obj["meta"] = {"data": data}
        style: dict[str, Any] = {}
        parts = ([boxfit] if boxfit else []) + cur["tr"]
        if parts:
            style["transform"] = " ".join(parts)
        if clip_spec is not None:
            style["clip_path"] = clip_spec
        if style:
            obj["style"] = style
        out.append(obj)

    def resolve_clip(props: dict[str, str]) -> Optional[dict[str, Any]]:
        ref = props.get("clip-path")
        if not ref:
            return None
        m = _URL_REF.match(ref.strip())
        return clip_specs.get(m.group(1)) if m else None

    def walk(el, inh: dict[str, Any], active_uses: frozenset[str],
             as_instance: bool = False) -> None:
        name = _localname(el.tag)
        if name == "symbol" and not as_instance:
            return  # per SVG spec, symbols render only when instanced via <use>
        if name in _SKIP_TAGS and name != "symbol":
            return
        props = _effective_props(el, rules)
        cur = {
            "fill": props.get("fill", inh["fill"]),
            "stroke": props.get("stroke", inh["stroke"]),
            "sw": props.get("stroke-width", inh["sw"]),
            "tr": inh["tr"] + ([el.get("transform")] if el.get("transform") else []),
        }
        if name in ("svg", "g", "a", "symbol"):
            for child in el:
                walk(child, cur, active_uses)
            return
        if name == "use":
            ref = _href(el)
            target = by_id.get(ref) if ref else None
            if target is None or ref in active_uses or len(active_uses) >= _MAX_USE_DEPTH:
                return                              # unresolvable or cyclic instance
            x, y = _f(el.get("x")), _f(el.get("y"))
            if x or y:
                cur = dict(cur)
                cur["tr"] = cur["tr"] + [f"translate({x:g} {y:g})"]
            walk(target, cur, active_uses | {ref}, as_instance=True)
            return
        if name == "text":
            obj = _emit_text(el, props, cur["fill"])
            if obj is not None:
                finish(obj, el, cur, resolve_clip(props))
            return
        obj = _emit(name, el)
        if obj is None:
            for child in el:
                walk(child, cur, active_uses)
            return
        _apply_paint(obj, name, cur["fill"], cur["stroke"], cur["sw"], gradients)
        finish(obj, el, cur, resolve_clip(props))

    walk(root, {"fill": None, "stroke": None, "sw": None, "tr": []}, frozenset())
    return out
