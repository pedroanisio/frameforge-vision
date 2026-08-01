"""Raster→gradient paint fitting — the pure domain authority (Gap-1 closure).

Given the pixel cloud of one traced shape (coordinates + sampled source
colours), fit three paint candidates — flat mean colour, linear gradient along
the cloud's principal axis, radial gradient from its centroid — and pick the
family by like-for-like colour rms under the ``fit_primitives`` doctrine: a
richer family must *beat* the simpler one above the noise floor, never win by
default. The output ``fill`` value is a model-ready paint (hex string or a
``Gradient`` dict) the renderer draws as-is.

Also owns the two geometry rules a correct emitter cannot guess:

* ``css_angle`` — the renderer maps a linear ``angle`` to a stop direction of
  ``(sin θ, −cos θ)`` in the object's LOCAL bbox space
  (``rendering/infrastructure/painters/svg.py``). For plain image-space
  geometry (polygons, layers paths) local == image (y-down):
  ``θ = atan2(dx, −dy)``. Potrace-ingested paths carry a ``scale(sx, −sy)``
  transform, so their local space is y-flipped and the composed mapping is
  ``θ = atan2(dx, dy)``.
* ``flatten_path_d`` / ``shoelace`` — subpath flattening + signed area, so the
  infrastructure can rasterise winding-correct masks without an SVG engine.

Pure stdlib + lazily-imported NumPy (already required by the vision group);
no PIL, no OpenCV, no I/O — exhaustively unit-testable.
"""
from __future__ import annotations

import math
import re
from typing import Any, Sequence

Point = tuple[float, float]

_TOKEN_RE = re.compile(r"[A-Za-z]|-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

# Fitting defaults. min_pixels guards against fitting noise on slivers;
# gain: the best gradient must explain enough variance to beat flat
# (rms_gradient < GAIN × rms_flat); radial_margin: radial must beat linear
# like-for-like, mirroring fit_primitives' ellipse-vs-circle rule.
DEFAULT_MIN_PIXELS = 400
DEFAULT_GAIN = 0.92
DEFAULT_RADIAL_MARGIN = 0.97
DEFAULT_N_STOPS = 5
_N_BINS = 12
_MIN_BIN_SAMPLES = 12
_STOP_TS = (0.0, 0.25, 0.5, 0.75, 1.0)


def flatten_path_d(d: str) -> list[list[Point]]:
    """Flatten an SVG path ``d`` into subpath point lists (curves sampled).

    Supports the command set the in-tree tracers emit (M/m, L/l, H/h, V/v,
    C/c, Q/q, Z/z, and implicit linetos after a moveto); an unknown command
    raises ``ValueError`` rather than silently mis-shaping the mask.

    Accepts BOTH forms of ``d``: an SVG string, or the structured segment list
    the SDK emitters produce (``[["M", x, y], ["L", x, y], ..., ["Z"]]`` —
    e.g. ``sdk.outline.stroke_outline``), so fitting and refinement see
    authored primitives, not only traced ones (G1).
    """
    if isinstance(d, (list, tuple)):
        d = " ".join(
            str(seg[0]) + " " + " ".join(f"{float(v):.6f}" for v in seg[1:])
            for seg in d if seg)
    tokens = _TOKEN_RE.findall(d or "")
    i = 0
    cmd: str | None = None
    cur: Point = (0.0, 0.0)
    start: Point = (0.0, 0.0)
    subs: list[list[Point]] = []
    pts: list[Point] = []

    def num() -> float:
        nonlocal i
        value = float(tokens[i])
        i += 1
        return value

    def close() -> None:
        nonlocal pts, cur
        if pts:
            pts.append(start)
            subs.append(pts)
        pts = []
        cur = start

    while i < len(tokens):
        tok = tokens[i]
        if tok.isalpha():
            cmd = tok
            i += 1
            if cmd in "Zz":
                close()
                cmd = None
            continue
        if cmd is None:
            raise ValueError("path data before any command")
        if cmd in "Mm":
            x, y = num(), num()
            if cmd == "m":
                x, y = cur[0] + x, cur[1] + y
            if pts:
                subs.append(pts)
            cur = start = (x, y)
            pts = [cur]
            cmd = "L" if cmd == "M" else "l"  # implicit linetos follow a moveto
        elif cmd in "Ll":
            x, y = num(), num()
            if cmd == "l":
                x, y = cur[0] + x, cur[1] + y
            cur = (x, y)
            pts.append(cur)
        elif cmd in "Hh":
            x = num()
            cur = ((x if cmd == "H" else cur[0] + x), cur[1])
            pts.append(cur)
        elif cmd in "Vv":
            y = num()
            cur = (cur[0], (y if cmd == "V" else cur[1] + y))
            pts.append(cur)
        elif cmd in "CcQq":
            if cmd in "Cc":
                x1, y1, x2, y2, x, y = (num() for _ in range(6))
            else:
                x1, y1, x, y = (num() for _ in range(4))
                x2, y2 = x1, y1
            if cmd in "cq":
                x1, y1 = cur[0] + x1, cur[1] + y1
                x2, y2 = cur[0] + x2, cur[1] + y2
                x, y = cur[0] + x, cur[1] + y
            p0 = cur
            for s in (0.25, 0.5, 0.75, 1.0):
                u = 1.0 - s
                px = u ** 3 * p0[0] + 3 * u * u * s * x1 + 3 * u * s * s * x2 + s ** 3 * x
                py = u ** 3 * p0[1] + 3 * u * u * s * y1 + 3 * u * s * s * y2 + s ** 3 * y
                pts.append((px, py))
            cur = (x, y)
        else:
            raise ValueError(f"unsupported path command {cmd!r}")
    if pts:
        subs.append(pts)
    return subs


def shoelace(pts: Sequence[Point]) -> float:
    """Signed polygon area (positive = counter-clockwise in y-up terms)."""
    area = 0.0
    n = len(pts)
    for k in range(n):
        x0, y0 = pts[k]
        x1, y1 = pts[(k + 1) % n]
        area += x0 * y1 - x1 * y0
    return area / 2.0


def css_angle(dx: float, dy: float, *, y_flipped: bool = False) -> float:
    """CSS gradient angle whose stop run follows image direction ``(dx, dy)``.

    ``(dx, dy)`` is the image-space (y-down) direction from stop 0 toward the
    last stop. ``y_flipped`` composes the potrace-ingest ``scale(sx, −sy)``
    local space (see module docstring for the derivation).
    """
    theta = math.atan2(dx, dy) if y_flipped else math.atan2(dx, -dy)
    return math.degrees(theta) % 360.0


def _to_hex(rgb) -> str:
    r, g, b = (int(max(0, min(255, round(float(v))))) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _binned_stops(t, colors, np) -> list[dict[str, Any]] | None:
    """Mean colour per t-bin, interpolated at the stop positions."""
    bins = np.clip((t * _N_BINS).astype(int), 0, _N_BINS - 1)
    means = np.full((_N_BINS, 3), np.nan)
    for b in range(_N_BINS):
        sel = bins == b
        if int(sel.sum()) >= _MIN_BIN_SAMPLES:
            means[b] = colors[sel].mean(axis=0)
    valid = ~np.isnan(means[:, 0])
    if int(valid.sum()) < 2:
        return None
    centers = (np.arange(_N_BINS) + 0.5) / _N_BINS
    stops = []
    for ts in _STOP_TS:
        col = [float(np.interp(ts, centers[valid], means[valid, ch])) for ch in range(3)]
        stops.append({"color": _to_hex(col), "position": f"{ts * 100:g}%"})
    return stops


def _model_rms(t, colors, np) -> float:
    """rms of the per-channel piecewise-linear colour model over parameter t."""
    bins = np.clip((t * _N_BINS).astype(int), 0, _N_BINS - 1)
    means = np.full((_N_BINS, 3), np.nan)
    for b in range(_N_BINS):
        sel = bins == b
        if int(sel.sum()) >= _MIN_BIN_SAMPLES:
            means[b] = colors[sel].mean(axis=0)
    valid = ~np.isnan(means[:, 0])
    if int(valid.sum()) < 2:
        return float("inf")
    centers = (np.arange(_N_BINS) + 0.5) / _N_BINS
    model = np.stack(
        [np.interp(t, centers[valid], means[valid, ch]) for ch in range(3)], axis=1)
    return float(np.sqrt(((colors - model) ** 2).mean()))


def fit_paint(
    points: Sequence[Point],
    colors: Sequence[tuple[float, float, float]],
    *,
    bbox: tuple[float, float, float, float] | None = None,
    y_flipped: bool = False,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    gain: float = DEFAULT_GAIN,
    radial_margin: float = DEFAULT_RADIAL_MARGIN,
    geometry: str = "bbox",
) -> dict[str, Any]:
    """Fit flat/linear/radial paint to one shape's sampled pixels; rank by rms.

    ``points`` are image-space pixel coordinates inside the shape; ``colors``
    the parallel RGB samples. ``bbox`` (``x0, y0, x1, y1``) is the shape's
    geometry bounds — the frame the renderer resolves a radial ``at`` against;
    defaults to the sample bounds. Returns ``{"family", "fill", "rms"}``.

    ``geometry`` selects the emitted gradient form:

    * ``"bbox"`` (default) — the CSS-relative approximation: linear ``angle``
      (via :func:`css_angle`, including the ``y_flipped`` composition) and
      radial fraction ``at``. Kept for angle-only consumers.
    * ``"user"`` — the EXACT A1 form, in the same image space as ``points``:
      linear ``line`` [[x1,y1],[x2,y2]] at the projection span's endpoints,
      radial px ``at`` + ``radius``. The caller owns any conversion into an
      object's local space (inverse transform); ``y_flipped`` is irrelevant
      here — the object transform carries orientation.
    """
    import numpy as np

    if geometry not in ("bbox", "user"):
        raise ValueError(f"unknown geometry {geometry!r}; use 'bbox' or 'user'")

    pts = np.asarray(points, dtype=np.float64)
    cols = np.asarray(colors, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] != cols.shape[0]:
        raise ValueError("points and colors must be parallel sequences")

    mean_color = cols.mean(axis=0) if len(cols) else np.zeros(3)
    rms_flat = float(np.sqrt(((cols - mean_color) ** 2).mean())) if len(cols) else 0.0
    result_rms: dict[str, float | None] = {"flat": rms_flat, "linear": None, "radial": None}
    flat_out = {"family": "flat", "fill": _to_hex(mean_color), "rms": result_rms}
    if len(pts) < max(3, int(min_pixels)):
        return flat_out

    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    proj = centred @ axis
    span = float(proj.max() - proj.min())
    t_lin = (proj - proj.min()) / (span + 1e-9)
    rms_linear = _model_rms(t_lin, cols, np)
    result_rms["linear"] = None if math.isinf(rms_linear) else rms_linear

    radial = np.linalg.norm(centred, axis=1)
    r_max = float(radial.max())
    t_rad = radial / (r_max + 1e-9)
    rms_radial = _model_rms(t_rad, cols, np)
    result_rms["radial"] = None if math.isinf(rms_radial) else rms_radial

    best_family = "linear"
    best_rms, best_t = rms_linear, t_lin
    if rms_radial < radial_margin * rms_linear:
        best_family, best_rms, best_t = "radial", rms_radial, t_rad
    if math.isinf(best_rms) or best_rms >= gain * rms_flat:
        return flat_out

    stops = _binned_stops(best_t, cols, np)
    if stops is None:
        return flat_out

    centre = pts.mean(axis=0)
    if best_family == "linear":
        if geometry == "user":
            p0 = centre + axis * float(proj.min())
            p1 = centre + axis * float(proj.max())
            fill: dict[str, Any] = {
                "kind": "linear",
                "line": [[round(float(p0[0]), 2), round(float(p0[1]), 2)],
                         [round(float(p1[0]), 2), round(float(p1[1]), 2)]],
                "stops": stops,
            }
        else:
            dx, dy = float(axis[0]), float(axis[1])
            fill = {
                "kind": "linear",
                "angle": round(css_angle(dx, dy, y_flipped=y_flipped), 1),
                "stops": stops,
            }
    else:
        cx, cy = (float(v) for v in centre)
        if geometry == "user":
            if r_max <= 0:
                return flat_out
            fill = {
                "kind": "radial",
                "at": [round(cx, 2), round(cy, 2)],
                "radius": round(r_max, 2),
                "stops": stops,
            }
        else:
            x0, y0, x1, y1 = bbox if bbox is not None else (
                float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()))
            fx = (cx - x0) / ((x1 - x0) or 1.0)
            fy = (cy - y0) / ((y1 - y0) or 1.0)
            if y_flipped:
                fy = 1.0 - fy
            fill = {
                "kind": "radial",
                "at": [round(min(1.0, max(0.0, fx)), 4), round(min(1.0, max(0.0, fy)), 4)],
                "stops": stops,
            }
    return {"family": best_family, "fill": fill, "rms": result_rms}


__all__ = ["flatten_path_d", "shoelace", "css_angle", "fit_paint",
           "DEFAULT_MIN_PIXELS", "DEFAULT_GAIN", "DEFAULT_RADIAL_MARGIN"]
