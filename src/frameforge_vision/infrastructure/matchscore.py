"""Closed-loop reconstruction scoring — the NUMERIC convergence signal.

The raster→vector loop is *measure → pin → construct_vectors → compare_images →
refine*. ``compare_images`` shows a vision model **where** a reconstruction is off;
this module answers **how far** — it samples the constructed vector geometry and
measures each sample's distance to the source image's real edges, giving an agent a
single number to drive down over passes (``on_edge_frac`` up, ``mean_dist`` down)
instead of only eyeballing a diff.

⚠ ARCHITECTURAL CONTRACT (PALS's LAW). The edges are found by an adaptive Sobel
threshold — a **heuristic**, not ground truth. Use the score as a *relative* guide
across refinement passes (did this nudge move the vectors closer to the edges?), not
as an absolute correctness proof. The rendered overlay is a drawing aid; the numbers
are the signal, and the signal is advisory.

Coordinate convention (shared with ``edgesnap`` and the SDK/SVG doc space): all
coordinates are CONTINUOUS — pixel index ``i`` covers ``[i, i + 1)`` and its centre
is ``i + 0.5``. Detected edge pixels are therefore emitted at their centres
(``index + 0.5``), so authored geometry and measured edges live in one frame.

The score floor: ``mean_dist`` never reaches 0 even for geometrically exact shapes —
edge pixels are quantized to centres (±0.5 px against a continuous edge) and the
Sobel response peaks on a stroke's *flank* pixels (~1.0 px from a 1 px stroke's
centreline). The floor is named in the result payload (``floor``); gates on absolute
distances must calibrate it away with a known-exact probe, never assume 0.

numpy + Pillow are imported lazily so ``import frameforge_vision`` stays cheap; only
scoring/overlay touch them. Shape *sampling* is pure ``math`` and always available.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

# sample_shape mirrors construct.py's kind → geometry mapping so a scored shape is the
# one that gets drawn (see the per-kind dispatch in sample_shape).


# ─────────────────────────────────────────────────────────────
# pure shape sampling (points along a shape's geometry, image px)
# ─────────────────────────────────────────────────────────────
def _pts(shape: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in (shape.get("points") or [])]


def _seg_samples(points: Sequence[tuple[float, float]], *, closed: bool,
                 spacing: float) -> list[tuple[float, float]]:
    P = [(float(a), float(b)) for a, b in points]
    if len(P) < 2:
        return P
    if closed:
        P = P + [P[0]]
    out: list[tuple[float, float]] = []
    for (ax, ay), (bx, by) in zip(P[:-1], P[1:]):
        n = max(2, int(math.hypot(bx - ax, by - ay) / max(spacing, 1e-6)) + 1)
        for i in range(n):
            t = i / (n - 1)
            out.append((ax + t * (bx - ax), ay + t * (by - ay)))
    return out


def _catmull_samples(points, *, closed: bool, tension: float, spacing: float):
    P = [(float(a), float(b)) for a, b in points]
    if len(P) < 3:
        return _seg_samples(P, closed=closed, spacing=spacing)
    P = ([P[-1]] + P + [P[0], P[1]]) if closed else ([P[0]] + P + [P[-1]])
    k = tension * 2.0
    out: list[tuple[float, float]] = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0 * k, p1[1] + (p2[1] - p0[1]) / 6.0 * k)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0 * k, p2[1] - (p3[1] - p1[1]) / 6.0 * k)
        n = max(2, int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / max(spacing, 1e-6)) + 1)
        for j in range(n):
            t = j / (n - 1)
            mt = 1 - t
            out.append((
                mt**3 * p1[0] + 3 * mt * mt * t * c1[0] + 3 * mt * t * t * c2[0] + t**3 * p2[0],
                mt**3 * p1[1] + 3 * mt * mt * t * c1[1] + 3 * mt * t * t * c2[1] + t**3 * p2[1],
            ))
    return out


def _bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _circle_center_r(shape: dict[str, Any], points):
    if shape.get("r") is not None:
        c = points[0] if points else (0.0, 0.0)
        return c, float(shape["r"])
    (x0, y0), (x1, y1) = points[0], points[1]
    return (x0, y0), math.hypot(x1 - x0, y1 - y0)


def _ellipse_samples(cx, cy, rx, ry, *, spacing, rot_deg=0.0):
    a = math.radians(rot_deg)
    ca, sa = math.cos(a), math.sin(a)
    n = max(24, int(2 * math.pi * max(rx, ry, 1e-6) / max(spacing, 1e-6)))
    out = []
    for i in range(n):
        t = 2 * math.pi * i / n
        ex, ey = rx * math.cos(t), ry * math.sin(t)
        out.append((cx + ex * ca - ey * sa, cy + ex * sa + ey * ca))
    return out


def sample_shape(shape: dict[str, Any], *, spacing: float = 2.0) -> list[tuple[float, float]]:
    """Points sampled along a shape's drawn geometry (image px), ~``spacing`` px apart.

    ``shape`` uses the same schema ``construct_vectors`` consumes (``kind`` +
    ``points`` already resolved to image pixels; ``r`` / ``points_count`` /
    ``inner_ratio`` where relevant), so what is scored is what gets drawn. curve/spline
    are sampled exactly as construct draws them — an OPEN Catmull-Rom at the SDK's fixed
    1/6 tangent (== tension 0.5); any ``closed``/``tension`` field is ignored, since
    construct never honours it, so the scored geometry never diverges from the render.
    """
    kind = str(shape.get("kind", "")).lower()
    points = _pts(shape)
    if kind == "line":
        return _seg_samples(points[:2], closed=False, spacing=spacing)
    if kind in ("path", "trace", "polyline"):
        return _seg_samples(points, closed=False, spacing=spacing)
    if kind in ("curve", "spline"):
        return _catmull_samples(points, closed=False, tension=0.5, spacing=spacing)
    if kind in ("polygon", "triangle", "closed"):
        return _seg_samples(points, closed=True, spacing=spacing)
    if kind == "rect":
        x0, y0, x1, y1 = _bbox(points)
        return _seg_samples([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True, spacing=spacing)
    if kind == "ellipse":
        x0, y0, x1, y1 = _bbox(points)
        return _ellipse_samples((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2,
                                spacing=spacing)
    if kind == "circle":
        (cx, cy), r = _circle_center_r(shape, points)
        return _ellipse_samples(cx, cy, r, r, spacing=spacing)
    if kind == "star":
        (cx, cy), r = _circle_center_r(shape, points)
        n = max(2, int(shape.get("points_count", 5)))
        inner = float(shape.get("inner_ratio", 0.5))
        verts = []
        for i in range(n * 2):
            ang = math.radians(-90.0) + i * math.pi / n
            rr = r if i % 2 == 0 else r * inner
            verts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        return _seg_samples(verts, closed=True, spacing=spacing)
    if kind == "arc":
        # construct.resolve_arc is the single arc authority — the scored arc is
        # byte-for-byte the drawn arc (same circumcircle, same sweep direction).
        from .construct import resolve_arc

        spec = resolve_arc(shape, points)
        (cx, cy), r = spec["center"], spec["r"]
        a0, sweep = spec["start_rad"], spec["sweep_rad"]
        n = max(8, int(abs(sweep) * max(r, 1e-6) / max(spacing, 1e-6)) + 1)
        return [(cx + r * math.cos(a0 + sweep * i / n),
                 cy + r * math.sin(a0 + sweep * i / n)) for i in range(n + 1)]
    # 'text' (and any unknown kind) contributes no edge samples: glyph outlines are
    # font-renderer geometry this pure-math layer cannot reproduce.
    return []


# ─────────────────────────────────────────────────────────────
# geometry-arg resolution (workspace pin ids → image px)
# ─────────────────────────────────────────────────────────────
def _geometry_point(p: Any, anchors: dict[str, tuple[float, float]]) -> list[float]:
    if isinstance(p, str):
        if p not in anchors:
            raise ValueError(f"unknown pin/landmark {p!r} in geometry args "
                             f"(known: {sorted(anchors)[:12]})")
        x, y = anchors[p]
        return [float(x), float(y)]
    return [float(p[0]), float(p[1])]


def resolve_geometry_args(symmetry_pairs: "Sequence[Any] | None",
                          collinear_groups: "Sequence[Any] | None",
                          anchors: "dict[str, tuple[float, float]] | None" = None,
                          ) -> tuple["list[list[list[float]]] | None",
                                     "list[list[list[float]]] | None"]:
    """Resolve ``score_reconstruction``'s geometry args to raw pixel points.

    Each point may be ``[x, y]`` (passed through) or a workspace pin/landmark id
    string (``"P3"`` / ``"A9"``), resolved via ``anchors`` — the same
    ``{id: (x_px, y_px)}`` map the shape ``pins`` resolution uses. ``None`` args
    pass through; an unknown id raises (a typo must not silently score nothing).
    """
    anchors = anchors or {}
    pairs = ([[_geometry_point(a, anchors), _geometry_point(b, anchors)]
              for a, b in symmetry_pairs] if symmetry_pairs else None)
    groups = ([[_geometry_point(p, anchors) for p in g]
               for g in collinear_groups] if collinear_groups else None)
    return pairs, groups


# ─────────────────────────────────────────────────────────────
# edge extraction + scoring (numpy)
# ─────────────────────────────────────────────────────────────
def _gray(image_bytes: bytes):
    from io import BytesIO

    import numpy as np
    from PIL import Image

    g = np.asarray(Image.open(BytesIO(image_bytes)).convert("L"), dtype=float)
    return g


def _edge_mask(gray, roi):
    """Adaptive-Sobel edge mask over ``roi`` (bool, roi-shaped); ``None`` if degenerate."""
    import numpy as np

    x0, y0, x1, y1 = roi
    sub = gray[y0:y1, x0:x1]
    if sub.size == 0 or min(sub.shape) < 3:
        return None
    gx = np.zeros_like(sub)
    gy = np.zeros_like(sub)
    gx[:, 1:-1] = sub[:, 2:] - sub[:, :-2]
    gy[1:-1, :] = sub[2:, :] - sub[:-2, :]
    mag = np.hypot(gx, gy)
    thr = max(float(mag.mean() + mag.std()), 1e-6)      # 1e-6 floor: a flat image has no edges
    return mag >= thr


def _edge_points(gray, roi):
    """Adaptive-Sobel edge-pixel CENTRES (continuous image coords) inside ``roi``.

    Pixel index ``(ix, iy)`` is emitted as ``(ix + 0.5, iy + 0.5)`` — the pixel-centre
    convention (module doc) — so detected edges share the continuous frame authored
    shapes use. ``None`` if none.
    """
    import numpy as np

    mask = _edge_mask(gray, roi)
    if mask is None:
        return None
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return np.stack([xs + roi[0] + 0.5, ys + roi[1] + 0.5], 1).astype(float)


# Chebyshev search radii for the exact nearest-edge lookup; samples farther than the
# last ring from any edge fall back to a subsampled brute force (see below).
_RINGS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def _nearest_edge_dist(mask, roi, S, *, max_edges=None):
    """Distance from each continuous sample in ``S`` to the nearest edge-pixel centre.

    EXACT for every sample within ``_RINGS[-1]`` px (Chebyshev) of an edge: an
    integral image answers "any edge within radius r?" in O(1), and the true
    nearest neighbour is then taken over a window guaranteed to contain it — no
    resolution-dependent caps, so a perfect 4K reconstruction measures its true
    floor instead of a subsampling artifact. Samples beyond the last ring (>64 px
    off every edge) use a brute force against at most ``max_edges`` subsampled
    edge pixels, whose relative error is negligible at that range.
    """
    import numpy as np

    x0, y0 = roi[0], roi[1]
    Hm, Wm = mask.shape
    integ = np.zeros((Hm + 1, Wm + 1), dtype=np.int64)
    integ[1:, 1:] = mask.cumsum(0, dtype=np.int64).cumsum(1)

    # containing pixel of each sample, in mask indices (pixel i covers [i, i+1))
    ix = np.clip(np.floor(S[:, 0]).astype(np.int64) - x0, 0, Wm - 1)
    iy = np.clip(np.floor(S[:, 1]).astype(np.int64) - y0, 0, Hm - 1)

    def counts(r):
        cx0 = np.clip(ix - r, 0, Wm)
        cx1 = np.clip(ix + r + 1, 0, Wm)
        cy0 = np.clip(iy - r, 0, Hm)
        cy1 = np.clip(iy + r + 1, 0, Hm)
        return integ[cy1, cx1] - integ[cy0, cx1] - integ[cy1, cx0] + integ[cy0, cx0]

    found = np.full(len(S), -1, dtype=np.int64)
    for r in _RINGS:
        undecided = found < 0
        if not undecided.any():
            break
        found[undecided & (counts(r) > 0)] = r

    d = np.full(len(S), np.inf)
    for r in _RINGS:
        group = np.where(found == r)[0]
        if group.size == 0:
            continue
        # a hit at Chebyshev r bounds the nearest centre's Euclidean distance by
        # sqrt(2)*(r+1); every pixel that could beat it lies within this window
        R = int(math.ceil((r + 1) * math.sqrt(2.0))) + 1
        offs = np.arange(-R, R + 1)
        oy, ox = np.meshgrid(offs, offs, indexing="ij")
        oy, ox = oy.ravel(), ox.ravel()
        chunk = max(1, 4_000_000 // len(oy))            # bound the gather's memory
        for i in range(0, group.size, chunk):
            g = group[i:i + chunk]
            cy = iy[g][:, None] + oy[None, :]
            cx = ix[g][:, None] + ox[None, :]
            valid = (cy >= 0) & (cy < Hm) & (cx >= 0) & (cx < Wm)
            cy = np.clip(cy, 0, Hm - 1)
            cx = np.clip(cx, 0, Wm - 1)
            dx = S[g, 0][:, None] - (cx + x0 + 0.5)
            dy = S[g, 1][:, None] - (cy + y0 + 0.5)
            dist2 = dx * dx + dy * dy
            dist2[~(mask[cy, cx] & valid)] = np.inf
            d[g] = np.sqrt(dist2.min(1))

    far = np.where(found < 0)[0]
    if far.size:
        ys, xs = np.where(mask)
        E = np.stack([xs + x0 + 0.5, ys + y0 + 0.5], 1).astype(float)
        cap = max_edges or 6000
        if len(E) > cap:
            E = E[np.linspace(0, len(E) - 1, cap).astype(int)]
        for i in range(0, far.size, 256):               # chunked to bound memory
            f = far[i:i + 256]
            d[f] = np.sqrt(((S[f][:, None, :] - E[None, :, :]) ** 2).sum(-1)).min(1)
    return d


_HINT = ("Higher on_edge_frac + lower distances = the vectors sit closer to the "
         "source's edges. Heuristic (adaptive Sobel) — a RELATIVE guide across "
         "refinement passes, not ground truth (PALS's Law).")

_FLOOR = {
    "mean_dist_floor_px": 0.5,
    "note": ("mean_dist never reaches 0 for exact geometry: detected edges are "
             "pixel centres (±0.5 px quantization against a continuous edge) and "
             "the Sobel response peaks on a stroke's flank pixels (~1.0 px from a "
             "1 px stroke's centreline). Gates on absolute distances must be "
             "calibrated against a known-exact probe rendered through the same "
             "pipeline, and subtract that measured floor."),
}


def _score_core(image_bytes: bytes, shapes: Sequence[dict[str, Any]], *,
                roi=None, tol: float = 2.0, spacing: float = 2.0,
                max_edges: "int | None" = None, max_samples: "int | None" = None):
    """Shared math for :func:`score` and :func:`build_score_overlay`.

    Returns ``(result, S, d)`` where ``result`` is the score dict (or an ``error``
    dict), ``S`` the kept shape-sample array, and ``d`` the per-sample nearest-edge
    distance (both ``None`` on error). ``max_edges``/``max_samples`` default to
    ``None`` — every edge pixel and every shape sample participates, so the score
    is resolution-independent (fixed caps made rich 4K content score *worse* than
    the same content at 960 px). Pass ints only to bound runtime explicitly;
    thinning trades exactness for speed.
    """
    import numpy as np

    if not shapes:
        return {"error": "no shapes to score"}, None, None
    gray = _gray(image_bytes)
    H, W = gray.shape
    if roi:
        roi = (max(0, int(math.floor(roi[0]))), max(0, int(math.floor(roi[1]))),
               min(W, int(math.ceil(roi[2]))), min(H, int(math.ceil(roi[3]))))
    else:
        roi = (0, 0, W, H)
    x0, y0, x1, y1 = roi
    mask = _edge_mask(gray, roi)
    if mask is None or not mask.any():
        return {"error": "no edges detected in roi", "roi": list(roi)}, None, None

    samples: list[tuple[float, float]] = []
    shape_of: list[int] = []
    for si, sh in enumerate(shapes):
        pts = sample_shape(sh, spacing=spacing)
        samples.extend(pts)
        shape_of.extend([si] * len(pts))
    keep = [i for i, p in enumerate(samples) if x0 <= p[0] < x1 and y0 <= p[1] < y1]
    if not keep:
        return {"error": "no shape samples inside roi", "roi": list(roi)}, None, None
    S = np.array([samples[i] for i in keep], dtype=float)
    SI = np.array([shape_of[i] for i in keep], dtype=np.int64)
    if max_samples and len(S) > max_samples:
        thin = np.linspace(0, len(S) - 1, max_samples).astype(int)
        S, SI = S[thin], SI[thin]

    d = _nearest_edge_dist(mask, roi, S, max_edges=max_edges)

    per_shape: list[dict[str, Any]] = []
    worst = None
    for si, sh in enumerate(shapes):
        member = SI == si
        entry: dict[str, Any] = {"index": si, "kind": str(sh.get("kind", "")).lower(),
                                 "n_samples": int(member.sum())}
        if sh.get("id") is not None:
            entry["id"] = sh["id"]
        if entry["n_samples"]:
            ds = d[member]
            entry["on_edge_frac"] = round(float((ds <= tol).mean()), 4)
            entry["mean_dist"] = round(float(ds.mean()), 3)
            entry["p90_dist"] = round(float(np.percentile(ds, 90)), 3)
            if worst is None or entry["mean_dist"] > worst["mean_dist"]:
                worst = {"index": si, "mean_dist": entry["mean_dist"]}
        per_shape.append(entry)

    result = {
        "roi": list(roi),
        "n_samples": int(len(S)),
        "n_edges": int(mask.sum()),
        "on_edge_frac": round(float((d <= tol).mean()), 4),
        "mean_dist": round(float(d.mean()), 3),
        "median_dist": round(float(np.median(d)), 3),
        "p90_dist": round(float(np.percentile(d, 90)), 3),
        "tol": tol,
        "per_shape": per_shape,
        "floor": dict(_FLOOR),
        "hint": _HINT,
    }
    if worst is not None:
        result["worst_shape"] = worst
    return result, S, d


def score(image_bytes: bytes, shapes: Sequence[dict[str, Any]], *,
          roi=None, tol: float = 2.0, spacing: float = 2.0,
          max_edges: "int | None" = None, max_samples: "int | None" = None) -> dict[str, Any]:
    """Score how well ``shapes`` sit on the source image's edges (see module doc)."""
    result, _, _ = _score_core(image_bytes, shapes, roi=roi, tol=tol, spacing=spacing,
                               max_edges=max_edges, max_samples=max_samples)
    return result


# ─────────────────────────────────────────────────────────────
# visual overlay (source dimmed + edges + on/off-edge samples + score header)
# ─────────────────────────────────────────────────────────────
_EDGE_C = (24, 176, 196, 150)     # cyan — detected edges
_ON_C = (36, 200, 96)             # green — sample within tol of an edge
_OFF_C = (232, 40, 40)            # red — sample off the edges


def build_score_overlay(image_bytes: bytes, shapes: Sequence[dict[str, Any]], *,
                        roi=None, tol: float = 2.0, spacing: float = 2.0,
                        max_edges: "int | None" = None, max_samples: "int | None" = None):
    """Return ``(overlay_rgb, score)`` — the source (dimmed) with detected edges and
    the shape samples coloured green (on-edge) / red (off-edge), plus a score header.
    ``overlay`` keeps the source pixel size (coordinate identity). On a scoring error
    the source image is returned with the error under ``score``."""
    from io import BytesIO

    from PIL import Image, ImageDraw

    from .image_compare import _font

    base = Image.open(BytesIO(image_bytes)).convert("RGB")
    result, S, d = _score_core(image_bytes, shapes, roi=roi, tol=tol, spacing=spacing,
                               max_edges=max_edges, max_samples=max_samples)
    if S is None:
        return base, result

    canvas = Image.blend(base, Image.new("RGB", base.size, (250, 250, 250)), 0.55).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    edges = _edge_points(_gray(image_bytes), tuple(result["roi"]))
    if edges is not None:
        step = max(1, len(edges) // 4000)
        for ex, ey in edges[::step]:
            draw.point((ex, ey), fill=_EDGE_C)
    for (sx, sy), dist in zip(S, d):
        c = _ON_C if dist <= tol else _OFF_C
        draw.ellipse([sx - 1.5, sy - 1.5, sx + 1.5, sy + 1.5], fill=c)
    over = Image.alpha_composite(canvas, layer)

    head = ImageDraw.Draw(over)
    txt = ("match  on_edge=%.0f%%  mean=%.2fpx  p90=%.2fpx  (tol=%gpx)"
           % (result["on_edge_frac"] * 100, result["mean_dist"], result["p90_dist"], tol))
    head.rectangle([0, 0, base.width, 22], fill=(18, 20, 26, 220))
    head.text((6, 4), txt, font=_font(13, bold=True), fill=(244, 244, 240))
    return over.convert("RGB"), result
