"""Sub-pixel edge sampling — the pixel half of constraint-based reconstruction.

:mod:`frameforge_vision.domain.geometry` fits lines and intersects them (exact
maths); this module supplies the sub-pixel edge points those fits consume. Given a
rough segment laid *across* an edge, it walks the segment and, at each step,
samples the image-gradient profile along the segment **normal** and locates the
edge to sub-pixel precision (parabolic peak of the gradient magnitude). Fitting a
line to those crossings — then intersecting two such lines for a corner — is why a
constraint pipeline beats eyeballing corners: an edge is sampled at many points, so
its position averages down to a fraction of a pixel, whereas a corner is a single
blurred point.

⚠ ARCHITECTURAL CONTRACT (PALS's LAW). Gradient peaks are a **heuristic** edge
locator: a strong step gives a confident crossing, but texture, glow, or a soft
shadow can pull it. Each crossing carries a ``strength``; weak steps are dropped,
and the *number* of surviving crossings + the fit residual are the honest signal
that a refinement is trustworthy — not a guarantee. numpy + Pillow are imported
lazily so ``import frameforge_vision`` stays cheap (mirrors ``matchscore.py``).

Coordinate convention — CONTINUOUS pixel-centre, matching the SDK/SVG doc space
and ``matchscore``/``coordinates``: pixel index ``i`` covers ``[i, i + 1)`` and is
sampled at its centre ``i + 0.5``. Every point this module accepts or returns
(segment anchors, crossings, snapped points, corners) lives in that frame, so a
crisp edge authored at x = 100.0 measures as 100.0 — not the 99.5 an index-space
reading would report (a systematic −0.5 px bias that alone consumes a sub-pixel
error budget). The conversion happens exactly once, at the raster boundary
(:func:`_bilinear`).
"""
from __future__ import annotations

import math
from typing import Any

from ..domain.geometry import fit_line, intersect
from .matchscore import _gray  # single source of truth for bytes → grayscale array

Point = tuple[float, float]


# ─────────────────────────────────────────────────────────────
# sampling helpers (numpy)
# ─────────────────────────────────────────────────────────────
def _bilinear(gray, xs, ys):
    """Bilinearly sample ``gray`` (H×W) at CONTINUOUS coords ``xs, ys`` (numpy arrays).

    The raster boundary of the pixel-centre convention (module doc): array index
    ``i`` holds the intensity at continuous coordinate ``i + 0.5``, so continuous
    inputs are shifted by −0.5 into index space before interpolation. Every peak
    position derived from these profiles is thereby continuous already.
    """
    import numpy as np

    H, W = gray.shape
    xs = np.clip(np.asarray(xs, dtype=float) - 0.5, 0, W - 1.001)
    ys = np.clip(np.asarray(ys, dtype=float) - 0.5, 0, H - 1.001)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = xs - x0
    fy = ys - y0
    return (gray[y0, x0] * (1 - fx) * (1 - fy) + gray[y0, x1] * fx * (1 - fy)
            + gray[y1, x0] * (1 - fx) * fy + gray[y1, x1] * fx * fy)


def _parabolic_peak(prof):
    """Index (float) and value of the profile's gradient-magnitude peak, sub-pixel.

    Returns ``(idx, strength)`` where ``idx`` is refined by fitting a parabola to the
    peak and its two neighbours. ``strength`` is the peak's height above the profile
    median (the edge contrast). Returns ``None`` if the profile is too short.
    """
    import numpy as np

    if prof.size < 3:
        return None
    g = np.abs(np.gradient(prof))
    k = int(np.argmax(g))
    strength = float(g[k] - np.median(g))
    if 0 < k < len(g) - 1:
        gm, gc, gp = g[k - 1], g[k], g[k + 1]
        denom = gm - 2.0 * gc + gp
        delta = 0.5 * (gm - gp) / denom if abs(denom) > 1e-9 else 0.0
        delta = max(-1.0, min(1.0, delta))
        return k + delta, strength
    return float(k), strength


# ─────────────────────────────────────────────────────────────
# edge crossings along a rough segment
# ─────────────────────────────────────────────────────────────
def edge_crossings_along(image_bytes: bytes, a: Point, b: Point, *,
                         band: float = 6.0, step: float = 2.0,
                         min_strength: float = 6.0) -> list[Point]:
    """Sub-pixel edge points found by walking segment ``a→b`` across an edge.

    At each step along ``a→b`` (``step`` px apart) the intensity is sampled along the
    segment normal within ``±band`` px; the gradient peak on that profile is the edge
    crossing, refined to sub-pixel. Steps whose peak is weaker than ``min_strength``
    (contrast above the local median) are dropped, so the result is only confident
    crossings — feed it to :func:`~frameforge_vision.domain.geometry.fit_line`.
    """
    import numpy as np

    gray = _gray(image_bytes)
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    length = math.hypot(bx - ax, by - ay)
    if length < 1e-6:
        return []
    ux, uy = (bx - ax) / length, (by - ay) / length      # along the segment
    nx, ny = -uy, ux                                      # unit normal (search axis)
    offs = np.arange(-band, band + 1e-9, 0.5)             # profile offsets along normal
    n_steps = max(2, int(length / max(step, 1e-6)) + 1)

    out: list[Point] = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        cx, cy = ax + t * (bx - ax), ay + t * (by - ay)
        prof = _bilinear(gray, cx + offs * nx, cy + offs * ny)
        peak = _parabolic_peak(prof)
        if peak is None:
            continue
        idx, strength = peak
        if strength < min_strength:
            continue
        s = -band + idx * 0.5                             # offset along the normal, px
        out.append((cx + s * nx, cy + s * ny))
    return out


def refine_edge_line(image_bytes: bytes, a: Point, b: Point, *,
                     band: float = 6.0, step: float = 2.0,
                     min_strength: float = 6.0) -> dict[str, Any]:
    """Fit a sub-pixel line to the edge crossed by segment ``a→b``.

    Returns ``{"ok", "line", "n_crossings", "rms_residual_px", "points"}``. ``ok`` is
    False (with a ``reason``) when fewer than two confident crossings were found — the
    caller should then keep the rough anchors rather than trust a bad fit (PALS).
    """
    pts = edge_crossings_along(image_bytes, a, b, band=band, step=step, min_strength=min_strength)
    if len(pts) < 2:
        return {"ok": False, "reason": "too few confident edge crossings",
                "n_crossings": len(pts), "points": [[round(x, 2), round(y, 2)] for x, y in pts]}
    line = fit_line(pts)
    resid = [line.distance(p) for p in pts]
    rms = math.sqrt(sum(r * r for r in resid) / len(resid))
    return {
        "ok": True,
        "line": line.to_dict(),
        "_line": line,
        "n_crossings": len(pts),
        "rms_residual_px": round(rms, 3),
        "points": [[round(x, 2), round(y, 2)] for x, y in pts],
    }


def refine_corner(image_bytes: bytes, edge1: tuple[Point, Point], edge2: tuple[Point, Point],
                  *, band: float = 6.0, step: float = 2.0,
                  min_strength: float = 6.0) -> dict[str, Any]:
    """Locate a corner as the intersection of two sub-pixel-refined edge lines.

    ``edge1``/``edge2`` are rough ``(a, b)`` segments each laid across one of the two
    edges meeting at the corner. Returns the intersection plus both fits, or
    ``ok=False`` when either edge could not be refined or the edges are parallel.
    """
    r1 = refine_edge_line(image_bytes, *edge1, band=band, step=step, min_strength=min_strength)
    r2 = refine_edge_line(image_bytes, *edge2, band=band, step=step, min_strength=min_strength)
    if not (r1["ok"] and r2["ok"]):
        return {"ok": False, "reason": "an edge could not be refined", "edge1": r1, "edge2": r2}
    try:
        cx, cy = intersect(r1["_line"], r2["_line"])
    except ValueError as exc:
        return {"ok": False, "reason": str(exc), "edge1": r1, "edge2": r2}
    return {
        "ok": True,
        "corner_px": [round(cx, 3), round(cy, 3)],
        "edge1": {k: r1[k] for k in ("line", "n_crossings", "rms_residual_px")},
        "edge2": {k: r2[k] for k in ("line", "n_crossings", "rms_residual_px")},
    }


def snap_point_to_edge(image_bytes: bytes, p: Point, *,
                       search_dir: Point | None = None, band: float = 8.0,
                       min_strength: float = 6.0) -> dict[str, Any]:
    """Slide ``p`` onto the nearest edge along a search axis (sub-pixel).

    ``search_dir`` is the axis to search along (the edge normal); when omitted the
    local image-gradient direction at ``p`` is used, so the point snaps perpendicular
    to whatever edge it is nearest. Returns ``{"ok", "point_px", "moved_px",
    "strength", "search_dir"}`` or ``ok=False`` when no confident edge is within
    ``±band``.
    """
    import numpy as np

    gray = _gray(image_bytes)
    px, py = float(p[0]), float(p[1])
    if search_dir is None:
        gx = float(_bilinear(gray, np.array([px + 1]), np.array([py]))[0]
                   - _bilinear(gray, np.array([px - 1]), np.array([py]))[0])
        gy = float(_bilinear(gray, np.array([px]), np.array([py + 1]))[0]
                   - _bilinear(gray, np.array([px]), np.array([py - 1]))[0])
        mag = math.hypot(gx, gy)
        if mag < 1e-6:
            return {"ok": False, "reason": "no local gradient to infer a search direction",
                    "point_px": [round(px, 3), round(py, 3)]}
        dx, dy = gx / mag, gy / mag
    else:
        dm = math.hypot(float(search_dir[0]), float(search_dir[1]))
        if dm < 1e-9:
            return {"ok": False, "reason": "search_dir is zero-length"}
        dx, dy = float(search_dir[0]) / dm, float(search_dir[1]) / dm

    offs = np.arange(-band, band + 1e-9, 0.5)
    prof = _bilinear(gray, px + offs * dx, py + offs * dy)
    peak = _parabolic_peak(prof)
    if peak is None or peak[1] < min_strength:
        return {"ok": False, "reason": "no confident edge within band",
                "point_px": [round(px, 3), round(py, 3)],
                "strength": round(peak[1], 3) if peak else 0.0}
    idx, strength = peak
    s = -band + idx * 0.5
    nx, ny = px + s * dx, py + s * dy
    return {
        "ok": True,
        "point_px": [round(nx, 3), round(ny, 3)],
        "moved_px": round(math.hypot(nx - px, ny - py), 3),
        "strength": round(strength, 3),
        "search_dir": [round(dx, 4), round(dy, 4)],
    }
