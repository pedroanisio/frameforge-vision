"""Parametric primitive fitting — measured points → {line | arc | ellipse-arc}.

The bridge between region detection and primitives-first authoring: given a
point sample of a *band* (a filled stroke region — polygon boundary vertices
or pixel samples from ``detect_regions``), recover the parameters an author
would type into the SDK: line endpoints/angle/width, circle centre/radius/
angular span/stroke thickness, or axis-aligned ellipse radii. Every fit
reports an ``rms`` residual in pixels so callers (and ``fit_primitive``'s
classifier) can compare candidates honestly.

Distinct from :mod:`frameforge_vision.domain.fitting`, which owns the overlay
landmark *transform* fit — this module fits *geometry families* to one shape.

numpy-only by design — importable wherever the vision maths runs, no OpenCV
requirement. Angles are degrees in page space (0° = +x, y down, increasing
clockwise on screen).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

__all__ = ["fit_circle_arc", "fit_ellipse_arc", "fit_line", "fit_primitive"]


def _pts(points: Sequence[Sequence[float]]) -> "np.ndarray":
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
        raise ValueError("fitting needs an Nx2 point array with N >= 3")
    return arr


def _band(values: "np.ndarray") -> float:
    """Thickness of a jittered band: the central 94% spread of deviations."""
    lo, hi = np.percentile(values, [3.0, 97.0])
    return float(hi - lo)


def _angular_span(theta_deg: "np.ndarray") -> tuple[float, float]:
    """Longest contiguous angular run (5° bins), wrap-aware.

    Returns ``(a0, a1)`` with ``a1`` allowed past 180 so ``(a1 - a0) % 360``
    is always the span length.
    """
    binned = np.zeros(72, dtype=bool)
    binned[((theta_deg.astype(int) % 360) // 5) % 72] = True
    ext = np.r_[binned, binned]
    best_start, best_run, run, start = 0, 0, 0, 0
    for i, on in enumerate(ext):
        if on:
            if run == 0:
                start = i
            run += 1
            if run > best_run:
                best_start, best_run = start, run
        else:
            run = 0
    best_run = min(best_run, 72)
    a0 = (best_start % 72) * 5
    a1 = a0 + best_run * 5
    if a0 > 180:
        a0, a1 = a0 - 360, a1 - 360
    return float(a0), float(a1)


def fit_line(points: Sequence[Sequence[float]]) -> dict[str, Any]:
    """PCA line fit of a band: endpoints, angle, length, and band width."""
    p = _pts(points)
    mean = p.mean(axis=0)
    centred = p - mean
    cov = np.cov(centred.T)
    evals, evecs = np.linalg.eigh(cov)
    d = evecs[:, int(np.argmax(evals))]
    if d[0] < 0 or (d[0] == 0 and d[1] < 0):
        d = -d                       # canonical direction: line angle in (-90, 90]
    t = centred @ d
    perp = centred @ np.array([-d[1], d[0]])
    p0 = mean + t.min() * d
    p1 = mean + t.max() * d
    return {
        "kind": "line",
        "p0": [float(p0[0]), float(p0[1])],
        "p1": [float(p1[0]), float(p1[1])],
        "angle_deg": float(math.degrees(math.atan2(d[1], d[0]))),
        "length": float(t.max() - t.min()),
        "width": _band(perp),
        "rms": float(np.sqrt(np.mean(perp**2))),
    }


def _refine_circle(x: "np.ndarray", y: "np.ndarray", cx: float, cy: float,
                   iters: int = 6) -> tuple[float, float, float]:
    """Geometric Gauss–Newton refinement of a circle centre.

    The algebraic (Kasa) seed is biased toward the chord on short arcs, which
    inflates the apparent angular span; a few geometric iterations remove it.
    """
    for _ in range(iters):
        dx, dy = x - cx, y - cy
        d = np.hypot(dx, dy)
        d[d == 0] = 1e-9
        r = float(d.mean())
        e = d - r
        J = np.c_[-dx / d, -dy / d]
        Jc = J - J.mean(axis=0)
        g, *_ = np.linalg.lstsq(Jc, -(e - e.mean()), rcond=None)
        cx, cy = cx + float(g[0]), cy + float(g[1])
        if math.hypot(float(g[0]), float(g[1])) < 1e-4:
            break
    d = np.hypot(x - cx, y - cy)
    return cx, cy, float(d.mean())


def fit_circle_arc(points: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Circle fit of a band (Kasa seed + geometric refinement): centre,
    centreline radius, angular span, stroke thickness."""
    p = _pts(points)
    x, y = p[:, 0], p[:, 1]
    A = np.c_[x, y, np.ones(len(x))]
    (a, b, c), *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
    cx, cy = a / 2.0, b / 2.0
    cx, cy, r = _refine_circle(x, y, cx, cy)
    d = np.hypot(x - cx, y - cy)
    theta = np.degrees(np.arctan2(y - cy, x - cx))
    a0, a1 = _angular_span(theta)
    return {
        "kind": "arc",
        "center": [float(cx), float(cy)],
        "radius": float(r),
        "span_deg": [a0, a1],
        "thickness": _band(d - r),
        "rms": float(np.sqrt(np.mean((d - r) ** 2))),
    }


def _refine_ellipse(x: "np.ndarray", y: "np.ndarray", cx: float, cy: float,
                    a: float, b: float, iters: int = 8) -> tuple[float, float, float, float]:
    """Geometric Gauss–Newton refinement of an axis-aligned ellipse.

    The algebraic seed minimizes an algebraic distance; refining the radial
    residual ``d − r_e(θ)`` puts the ellipse on the same footing as the
    geometrically refined circle, so ``fit_primitive``'s rms comparison is
    like-for-like.
    """
    for _ in range(iters):
        dx, dy = x - cx, y - cy
        d = np.hypot(dx, dy)
        d[d == 0] = 1e-9
        phi = np.arctan2(dy / b, dx / a)
        cp, sp = np.cos(phi), np.sin(phi)
        r_e = np.hypot(a * cp, b * sp)
        r_e[r_e == 0] = 1e-9
        e = d - r_e
        J = np.c_[-dx / d, -dy / d, -a * cp**2 / r_e, -b * sp**2 / r_e]
        try:
            g, *_ = np.linalg.lstsq(J, -e, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover — degenerate step
            break
        cx += float(g[0])
        cy += float(g[1])
        a = max(a + float(g[2]), 1e-6)
        b = max(b + float(g[3]), 1e-6)
        if float(np.max(np.abs(g))) < 1e-4:
            break
    return cx, cy, a, b


def _degenerate_ellipse(cx: float, cy: float) -> dict[str, Any]:
    return {"kind": "ellipse-arc", "center": [float(cx), float(cy)],
            "radii": [0.0, 0.0], "span_deg": [0.0, 0.0], "thickness": 0.0,
            "rms": float("inf")}


def fit_ellipse_arc(points: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Axis-aligned ellipse fit: centre, radii ``(a, b)``, span, thickness.

    Solves the linear model ``A·x² + C·y² + D·x + E·y = 1`` (no cross term —
    the axis-aligned family the SDK's sampled ellipse arcs draw), then
    measures radial deviation against the fitted ellipse along each point's
    ray from the centre.
    """
    p = _pts(points)
    mx, my = p.mean(axis=0)
    scale = float(p.std()) or 1.0
    xn, yn = (p[:, 0] - mx) / scale, (p[:, 1] - my) / scale
    M = np.c_[xn**2, yn**2, xn, yn]
    try:
        coef, *_ = np.linalg.lstsq(M, np.ones(len(xn)), rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover — degenerate input
        return _degenerate_ellipse(float(mx), float(my))
    A, C, D, E = (float(v) for v in coef)
    if A <= 0 or C <= 0:
        return _degenerate_ellipse(float(mx), float(my))
    cxn, cyn = -D / (2 * A), -E / (2 * C)
    rhs = 1 + A * cxn**2 + C * cyn**2
    if rhs <= 0:
        return _degenerate_ellipse(mx + scale * cxn, my + scale * cyn)
    ax = scale * math.sqrt(rhs / A)
    bx = scale * math.sqrt(rhs / C)
    cx, cy = mx + scale * cxn, my + scale * cyn
    x, y = p[:, 0], p[:, 1]
    cx, cy, ax, bx = _refine_ellipse(x, y, cx, cy, ax, bx)
    theta = np.arctan2((y - cy) / bx, (x - cx) / ax)
    r_e = np.hypot(ax * np.cos(theta), bx * np.sin(theta))
    d = np.hypot(x - cx, y - cy)
    a0, a1 = _angular_span(np.degrees(np.arctan2(y - cy, x - cx)))
    return {
        "kind": "ellipse-arc",
        "center": [float(cx), float(cy)],
        "radii": [float(ax), float(bx)],
        "span_deg": [a0, a1],
        "thickness": _band(d - r_e),
        "rms": float(np.sqrt(np.mean((d - r_e) ** 2))),
    }


def fit_primitive(points: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Fit every family and pick the best by residual, simplest-on-ties.

    A line within 5% of a curved fit's rms is a line. Circle vs ellipse is
    decided on evidence, not rms alone: the ellipse keeps the win only when
    its fitted axis difference ``|a − b|`` exceeds the band's noise floor
    (15% of the fitted thickness) — an ellipse with ``a ≈ b`` *is* the
    circle, so the extra parameter must show consistent structure. The
    winner is returned with a ``candidates`` list ranked by rms.
    """
    line = fit_line(points)
    arc = fit_circle_arc(points)
    ell = fit_ellipse_arc(points)
    candidates = sorted([line, arc, ell], key=lambda f: f["rms"])
    best = candidates[0]
    if best["kind"] == "arc" and line["rms"] <= arc["rms"] * 1.05:
        best = line
    elif best["kind"] == "ellipse-arc":
        aspect_gap = abs(ell["radii"][0] - ell["radii"][1])
        noise_floor = max(0.15 * max(ell["thickness"], 1.0), 2.5)
        earns_keep = aspect_gap > noise_floor and ell["rms"] < arc["rms"] * 0.995
        if line["rms"] <= ell["rms"] * 1.05:
            best = line
        elif not earns_keep:
            best = arc
    out = dict(best)
    out["candidates"] = candidates
    return out
