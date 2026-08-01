"""Vector construction from anchor points — the raster→vector output stage.

The measurement / workspace layers let the AI *find* precise coordinates; this
layer turns those coordinates into actual FrameForge vector geometry. Given ordered
anchor points (workspace pins or explicit image-pixel points) and a shape kind, it
authors a real FrameForge document via the SDK — so the reconstruction is validated
by the model and rendered by the same pipeline as everything else, and can be diffed
against the source with ``compare_images``.

Supported kinds map onto the SDK's canonical primitives:

    line ......... 2 points          -> line
    path/trace ... >=2 points        -> polyline (open)
    curve/spline . >=2 points        -> smooth polyline (Catmull-Rom -> path)
    polygon ...... >=3 points        -> polygon (closed polyline)
    triangle ..... 3 points          -> polygon
    closed ....... >=3 points        -> closed polyline (custom closed region)
    rect ......... 2+ points (bbox)  -> rect
    ellipse ...... 2+ points (bbox)  -> ellipse
    circle ....... 1 point + r, or 2 points (centre, rim) -> circle
    star ......... 1 point + r, or 2 points; ``points_count``/``inner_ratio`` -> polygon
    arc .......... 3 points (start, on-arc, end -> circumcircle), or
                   1 point (centre) + ``r`` + ``start_deg``/``end_deg`` -> path (A)
    text ......... 1 point (anchor, top-left) + ``text`` + ``size`` (font px),
                   or 2+ points (bbox) + ``text`` + ``size``   -> text

Angles are measured in image coordinates: 0° along +x, positive angles sweep
clockwise on screen (+y is down). The SDK validates every object, so a malformed
shape fails loudly here rather than producing a silently-wrong document.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

_DEFAULT_STYLE = {"stroke": "#e01c2c", "stroke_width": 2}
_OPEN_KINDS = {"line", "path", "trace", "polyline", "curve", "spline", "arc"}


def _normalize_style(style: dict[str, Any]) -> dict[str, Any]:
    """Lower the ergonomic ``stroke_width`` into ``stroke_style`` (P3: stroke geometry
    is not a direct object field; only stroke *paint* is)."""
    s = dict(style)
    sw = s.pop("stroke_width", None)
    if sw is not None:
        ss = dict(s.get("stroke_style") or {})
        ss.setdefault("stroke_width", sw)
        s["stroke_style"] = ss
    return s
_CLOSED_KINDS = {"polygon", "triangle", "closed", "rect", "ellipse", "circle", "star"}
_TEXT_KINDS = {"text"}
SHAPE_KINDS = sorted(_OPEN_KINDS | _CLOSED_KINDS | _TEXT_KINDS)


def _xy(p: Sequence[float]) -> list[float]:
    if len(p) < 2:
        raise ValueError(f"point {p!r} needs [x, y]")
    return [float(p[0]), float(p[1])]


def _bbox(points: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def resolve_arc(shape: dict[str, Any], pts: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Resolve an ``arc`` shape to ``{center, r, start_rad, sweep_rad}``.

    Two forms: 3 points (start, on-arc, end — the circumcircle through them, swept
    from start to end THROUGH the middle point) or 1 centre point + ``r`` +
    ``start_deg``/``end_deg``. Angles in image coordinates (0° = +x, positive =
    clockwise on screen); ``sweep_rad`` is signed. Shared with
    ``matchscore.sample_shape`` so the scored arc is exactly the drawn arc.
    """
    if len(pts) == 3:
        (ax, ay), (bx, by), (cx0, cy0) = pts
        d = 2.0 * (ax * (by - cy0) + bx * (cy0 - ay) + cx0 * (ay - by))
        if abs(d) < 1e-9:
            raise ValueError("arc points are collinear; a 3-point arc needs a real circumcircle")
        a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx0 * cx0 + cy0 * cy0
        ux = (a2 * (by - cy0) + b2 * (cy0 - ay) + c2 * (ay - by)) / d
        uy = (a2 * (cx0 - bx) + b2 * (ax - cx0) + c2 * (bx - ax)) / d
        r = math.hypot(ax - ux, ay - uy)
        a0 = math.atan2(ay - uy, ax - ux)
        a1 = math.atan2(by - uy, bx - ux)
        a2r = math.atan2(cy0 - uy, cx0 - ux)
        two_pi = 2.0 * math.pi
        d1 = (a1 - a0) % two_pi
        d2 = (a2r - a0) % two_pi
        # sweep in the direction that passes through the middle point
        sweep = d2 if d1 <= d2 else -(two_pi - d2)
        return {"center": (ux, uy), "r": r, "start_rad": a0, "sweep_rad": sweep}
    if len(pts) == 1:
        r = float(shape.get("r", 0))
        if r <= 0:
            raise ValueError("arc from a centre point needs a positive 'r'")
        if shape.get("start_deg") is None or shape.get("end_deg") is None:
            raise ValueError("arc from a centre point needs 'start_deg' and 'end_deg'")
        start = math.radians(float(shape["start_deg"]))
        sweep = math.radians(float(shape["end_deg"]) - float(shape["start_deg"]))
        if sweep == 0.0 or abs(sweep) >= 2.0 * math.pi:
            raise ValueError("arc sweep must be non-zero and under 360 degrees "
                             "(use 'circle' for a full circle)")
        return {"center": (float(pts[0][0]), float(pts[0][1])), "r": r,
                "start_rad": start, "sweep_rad": sweep}
    raise ValueError("arc needs 3 points (start, on-arc, end) or 1 centre point "
                     "with r + start_deg/end_deg")


def _arc_path_d(spec: dict[str, Any]) -> str:
    """SVG path ``d`` for a resolved arc (one ``A`` command)."""
    cx, cy = spec["center"]
    r, a0, sweep = spec["r"], spec["start_rad"], spec["sweep_rad"]
    x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
    a1 = a0 + sweep
    x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
    large = 1 if abs(sweep) > math.pi else 0
    # in image coordinates (+y down) a positive sweep is SVG sweep-flag 1
    sf = 1 if sweep > 0 else 0
    return (f"M {x0:.3f} {y0:.3f} A {r:.3f} {r:.3f} 0 {large} {sf} {x1:.3f} {y1:.3f}")


def _star_points(center: Sequence[float], r_outer: float, n: int,
                 inner_ratio: float, rotation_deg: float = -90.0) -> list[list[float]]:
    cx, cy = float(center[0]), float(center[1])
    r_in = r_outer * inner_ratio
    pts: list[list[float]] = []
    for i in range(n * 2):
        ang = math.radians(rotation_deg) + i * math.pi / n
        r = r_outer if i % 2 == 0 else r_in
        pts.append([cx + r * math.cos(ang), cy + r * math.sin(ang)])
    return pts


def _add_shape(pb, shape: dict[str, Any]) -> dict[str, Any]:
    """Add one shape to a PageBuilder; return a summary record."""
    kind = str(shape.get("kind", "")).lower()
    if kind not in SHAPE_KINDS:
        raise ValueError(f"unknown shape kind {kind!r}; use one of {SHAPE_KINDS}")
    pts = [_xy(p) for p in (shape.get("points") or [])]
    style = _normalize_style({**_DEFAULT_STYLE, **(shape.get("style") or {})})
    # closed fills default to none unless the caller sets one
    summary: dict[str, Any] = {"kind": kind, "n_points": len(pts)}

    if kind == "line":
        if len(pts) != 2:
            raise ValueError("line needs exactly 2 points")
        pb.line(pts[0], pts[1], **style)
    elif kind in ("path", "trace", "polyline"):
        if len(pts) < 2:
            raise ValueError(f"{kind} needs >= 2 points")
        pb.polyline(pts, closed=False, **style)
    elif kind in ("curve", "spline"):
        if len(pts) < 2:
            raise ValueError(f"{kind} needs >= 2 points")
        pb.polyline(pts, closed=False, smooth=True, **style)
    elif kind in ("polygon", "triangle"):
        need = 3 if kind == "triangle" else 3
        if len(pts) < need:
            raise ValueError(f"{kind} needs >= {need} points")
        if kind == "triangle" and len(pts) != 3:
            raise ValueError("triangle needs exactly 3 points")
        pb.polygon(pts, **style)
    elif kind == "closed":
        if len(pts) < 3:
            raise ValueError("closed region needs >= 3 points")
        pb.polyline(pts, closed=True, **style)
    elif kind == "rect":
        if len(pts) < 2:
            raise ValueError("rect needs >= 2 points (its bbox)")
        x, y, w, h = _bbox(pts)
        pb.rect([x, y, w, h], **style)
        summary["box"] = [x, y, w, h]
    elif kind == "ellipse":
        if len(pts) < 2:
            raise ValueError("ellipse needs >= 2 points (its bbox)")
        x, y, w, h = _bbox(pts)
        pb.ellipse([x + w / 2, y + h / 2], w / 2, h / 2, **style)
    elif kind == "circle":
        if len(pts) == 1:
            r = float(shape.get("r", 0))
            if r <= 0:
                raise ValueError("circle from 1 point needs a positive 'r'")
            center = pts[0]
        elif len(pts) >= 2:
            center = pts[0]
            r = math.hypot(pts[1][0] - center[0], pts[1][1] - center[1])
        else:
            raise ValueError("circle needs a centre point")
        pb.circle(center, r, **style)
        summary["center"], summary["r"] = center, r
    elif kind == "star":
        if len(pts) == 1:
            r = float(shape.get("r", 0))
            if r <= 0:
                raise ValueError("star from 1 point needs a positive 'r'")
            center = pts[0]
        elif len(pts) >= 2:
            center = pts[0]
            r = math.hypot(pts[1][0] - center[0], pts[1][1] - center[1])
        else:
            raise ValueError("star needs a centre point")
        n = int(shape.get("points_count", 5))
        inner = float(shape.get("inner_ratio", 0.5))
        if n < 2:
            raise ValueError("star needs points_count >= 2")
        star = _star_points(center, r, n, inner)
        pb.polygon(star, **style)
        summary["center"], summary["r"], summary["points_count"] = center, r, n
    elif kind == "arc":
        spec = resolve_arc(shape, pts)
        style.setdefault("fill", "none")            # a stroked arc, not a filled chord
        pb.path(_arc_path_d(spec), **style)
        summary["center"] = [spec["center"][0], spec["center"][1]]
        summary["r"] = spec["r"]
        summary["start_deg"] = math.degrees(spec["start_rad"])
        summary["sweep_deg"] = math.degrees(spec["sweep_rad"])
    elif kind == "text":
        content = str(shape.get("text") or "").strip()
        if not content:
            raise ValueError("text needs non-empty 'text' content")
        size = float(shape.get("size", 0))
        if size <= 0:
            raise ValueError("text needs a positive 'size' (font px)")
        if len(pts) >= 2:                           # explicit bbox
            x, y, w, h = _bbox(pts)
        elif len(pts) == 1:                         # anchor = box top-left; width estimated
            x, y = pts[0]
            w, h = max(size * 0.62 * len(content), size), size * 1.3
        else:
            raise ValueError("text needs an anchor point (or 2+ bbox points)")
        user = dict(shape.get("style") or {})
        for stroke_key in ("stroke", "stroke_width", "stroke_style"):
            user.pop(stroke_key, None)              # text carries no stroke geometry
        color = user.pop("color", None) or user.pop("fill", None) or _DEFAULT_STYLE["stroke"]
        pb.text([x, y, w, h], content, style={"font_size": size, "color": color, **user})
        summary["box"], summary["text"] = [x, y, w, h], content
    return summary


def build_document(shapes: Sequence[dict[str, Any]], *, width: int, height: int,
                   background: str | None = None,
                   title: str = "Vector reconstruction") -> tuple[str, list[dict[str, Any]]]:
    """Author a FrameForge YAML document from resolved shapes; return (yaml, summaries).

    ``shapes`` carry points already resolved to image pixels. The page is sized to
    the source image (px), so the drawn geometry overlays the raster 1:1.
    """
    from frameforge.sdk import DocumentBuilder, serialize

    if not shapes:
        raise ValueError("no shapes to construct")
    doc = DocumentBuilder(title=title)
    canvas: dict[str, Any] = {"size": [int(width), int(height)], "units": "px"}
    page = doc.page("reconstruction", canvas=canvas, coordinate_mode="absolute")
    # A page background is a full-canvas fill rect (the CanvasObject has no
    # `background` field), drawn first so the shapes sit on top of it.
    if background:
        page.rect([0, 0, int(width), int(height)], fill=background)
    summaries = [_add_shape(page, dict(s)) for s in shapes]
    yaml_text = serialize(doc.build(), format="yaml")
    return yaml_text, summaries
