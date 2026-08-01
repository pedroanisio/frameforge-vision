"""Plane-geometry primitives for constraint-based reconstruction — pure ``math``.

Corners in rasterised art are the *worst*-conditioned features to locate: the
signal is smeared over a blur disk, so pinning them directly has variance on the
order of the blur radius. **Edges** are the best-conditioned features: a straight
edge is a 1-D step sampled at many points, so fitting a line to those points
averages the per-sample noise down by ~1/√N. The right way to place a corner is
therefore *fit the two edges, then intersect them* — the corner inherits the
sub-pixel edge accuracy instead of the corner blur.

This module is the exact-math half of that method (the pixel sampling that feeds
it lives in :mod:`frameforge_vision.infrastructure.edgesnap`):

- :func:`fit_line` — **total least squares** (orthogonal / PCA) line through points.
  Unlike ``y = m·x + b`` it is stable for vertical and near-vertical edges (the
  legs of an ``A``, the sides of an ``I``), which is exactly where naive fits blow up.
- :func:`intersect` — the sub-pixel corner where two fitted edges meet.
- :func:`symmetry_axis_x` / :func:`symmetry_report` — the rigid constraint a
  bilaterally-symmetric mark carries. A luminance diff is blind to a single-corner
  offset (a 9 px apex shift can leave the pixel-match % unchanged); the symmetry
  residual is not.
- :func:`enforce_collinear` / :func:`collinearity_residual` — a straight edge
  interrupted by other geometry (a leg's inner edge split by the cross-bar) is one
  line; project the rough pins back onto it.
- :func:`mirror_slope_report` — the trigonometric check: two mirror edges have
  equal angle-from-vertical; a difference flags a broken symmetry.

⚠ ARCHITECTURAL CONTRACT (PALS's LAW). The geometry here is *exact* — a line fit,
an intersection, a reflection are deterministic. What is UNVERIFIED is the **input**:
edge samples come from a heuristic detector, and "these points lie on one edge" or
"these two features are a symmetric pair" are modelling assumptions the caller
asserts. Trust the maths; treat the point sets as untrusted. PIL/OpenCV-free so it
imports cheaply and is exhaustively unit-testable, like
:mod:`~frameforge_vision.domain.coordinates` and
:mod:`~frameforge_vision.domain.fitting`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Sequence

Point = tuple[float, float]


# ─────────────────────────────────────────────────────────────
# Line — a point + unit direction (handles verticals; slope-free)
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Line:
    """An infinite line as a point ``(px, py)`` on it plus a **unit** direction.

    A point+direction form (not ``y = m·x + b``) is used deliberately so vertical
    and near-vertical lines are ordinary, not singular. The direction is normalised
    to unit length and to a canonical sign (``ux > 0``, or ``uy > 0`` when
    ``ux == 0``) so a line built from ``a→b`` has the same direction as one built
    from ``b→a`` (the stored anchor point still reflects how it was constructed).
    """

    px: float
    py: float
    ux: float
    uy: float

    def __post_init__(self) -> None:
        n = math.hypot(self.ux, self.uy)
        if n < 1e-12:
            raise ValueError("Line needs a non-zero direction")
        ux, uy = self.ux / n, self.uy / n
        # canonical sign: point direction into +x (or +y when vertical)
        if ux < 0 or (abs(ux) < 1e-12 and uy < 0):
            ux, uy = -ux, -uy
        object.__setattr__(self, "ux", ux)
        object.__setattr__(self, "uy", uy)

    @classmethod
    def from_points(cls, a: Point, b: Point) -> "Line":
        return cls(float(a[0]), float(a[1]), float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))

    @property
    def angle_deg(self) -> float:
        """Direction angle from +x in degrees, folded to ``(-90, 90]`` (a line is mod 180)."""
        a = math.degrees(math.atan2(self.uy, self.ux))
        if a > 90.0:
            a -= 180.0
        elif a <= -90.0:
            a += 180.0
        return a

    @property
    def angle_from_vertical_deg(self) -> float:
        """Unsigned angle between the line and the vertical axis, in ``[0, 90]``."""
        return math.degrees(math.atan2(abs(self.ux), abs(self.uy)))

    def distance(self, p: Point) -> float:
        """Perpendicular distance from ``p`` to the line (direction is unit)."""
        return abs((p[0] - self.px) * self.uy - (p[1] - self.py) * self.ux)

    def project(self, p: Point) -> Point:
        """Foot of the perpendicular from ``p`` onto the line."""
        t = (p[0] - self.px) * self.ux + (p[1] - self.py) * self.uy
        return (self.px + t * self.ux, self.py + t * self.uy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_px": [round(self.px, 3), round(self.py, 3)],
            "direction": [round(self.ux, 6), round(self.uy, 6)],
            "angle_deg": round(self.angle_deg, 3),
        }


# ─────────────────────────────────────────────────────────────
# line fitting + intersection
# ─────────────────────────────────────────────────────────────
def fit_line(points: Sequence[Point]) -> Line:
    """Total-least-squares (orthogonal / PCA) best-fit line through ``points``.

    Minimises the sum of squared *perpendicular* distances, so it is orientation-free
    — correct for vertical and near-vertical edges where an ordinary ``y = m·x + b``
    regression is ill-conditioned or undefined. Needs ≥ 2 distinct points.
    """
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2:
        raise ValueError("fit_line needs at least 2 points")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    if sxx + syy < 1e-12:
        raise ValueError("fit_line needs at least 2 distinct points")
    # principal-axis angle of the 2x2 scatter matrix (largest-variance direction)
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return Line(mx, my, math.cos(theta), math.sin(theta))


def intersect(l1: Line, l2: Line, *, eps: float = 1e-9) -> Point:
    """The point where two lines cross. Raises ``ValueError`` if (near-)parallel."""
    cross = l1.ux * l2.uy - l1.uy * l2.ux
    if abs(cross) < eps:
        raise ValueError("lines are parallel (or nearly so); no unique intersection")
    dx, dy = l2.px - l1.px, l2.py - l1.py
    t = (dx * l2.uy - dy * l2.ux) / cross
    return (l1.px + t * l1.ux, l1.py + t * l1.uy)


# ─────────────────────────────────────────────────────────────
# symmetry (bilateral, vertical axis — upright letterforms/marks)
# ─────────────────────────────────────────────────────────────
def reflect_across_vertical(p: Point, axis_x: float) -> Point:
    """Mirror ``p`` across the vertical line ``x = axis_x`` (y unchanged)."""
    return (2.0 * axis_x - p[0], p[1])


def symmetry_axis_x(pairs: Sequence[tuple[Point, Point]]) -> float:
    """Consensus vertical axis of a set of ``(left, right)`` symmetric pairs.

    Uses the **median** of the pair mid-x values so one mis-measured pair (an
    outlier corner) does not drag the axis — the axis stays pinned to the honest
    majority, and the outlier then shows up loudly in :func:`symmetry_report`.
    """
    if not pairs:
        raise ValueError("need at least one symmetric pair")
    return float(median((l[0] + r[0]) / 2.0 for l, r in pairs))


def symmetry_report(pairs: Sequence[tuple[Point, Point]], *,
                    axis: float | None = None, tol: float = 2.0) -> dict[str, Any]:
    """Per-pair deviation from bilateral symmetry about a vertical axis.

    ``mirror_residual_px`` is the distance between the reflected left point and the
    right point (the full 2-D mismatch); ``axis_dev_px`` is how far this pair's own
    mid-x sits from the consensus axis (the single-corner-shift detector). Pairs with
    ``axis_dev_px > tol`` are flagged as outliers — this is what surfaces a corner the
    luminance diff cannot see.
    """
    if not pairs:
        raise ValueError("need at least one symmetric pair")
    ax = symmetry_axis_x(pairs) if axis is None else float(axis)
    entries: list[dict[str, Any]] = []
    for i, (l, r) in enumerate(pairs, start=1):
        mid = (l[0] + r[0]) / 2.0
        refl = reflect_across_vertical(l, ax)
        entries.append({
            "pair": i,
            "left_px": [round(l[0], 2), round(l[1], 2)],
            "right_px": [round(r[0], 2), round(r[1], 2)],
            "mid_x": round(mid, 3),
            "axis_dev_px": round(abs(mid - ax), 3),
            "mirror_residual_px": round(math.hypot(refl[0] - r[0], refl[1] - r[1]), 3),
            "y_delta_px": round(abs(l[1] - r[1]), 3),
            "outlier": abs(mid - ax) > tol,
        })
    devs = [e["axis_dev_px"] for e in entries]
    return {
        "axis_x": round(ax, 3),
        "tol": tol,
        "pairs": entries,
        "max_axis_dev_px": round(max(devs), 3),
        "rms_axis_dev_px": round(math.sqrt(sum(d * d for d in devs) / len(devs)), 3),
        "n_outliers": sum(1 for e in entries if e["outlier"]),
    }


# ─────────────────────────────────────────────────────────────
# collinearity (a straight edge split by other geometry is one line)
# ─────────────────────────────────────────────────────────────
def enforce_collinear(points: Sequence[Point]) -> list[Point]:
    """Project each point onto the best-fit line through all of them (≥ 2 points)."""
    line = fit_line(points)
    return [line.project(p) for p in points]


def collinearity_residual(points: Sequence[Point]) -> dict[str, Any]:
    """Max/RMS perpendicular distance of ``points`` from their best-fit line."""
    line = fit_line(points)
    dists = [line.distance(p) for p in points]
    return {
        "max_dist_px": round(max(dists), 3),
        "rms_dist_px": round(math.sqrt(sum(d * d for d in dists) / len(dists)), 3),
        "n_points": len(dists),
        "line": line.to_dict(),
    }


# ─────────────────────────────────────────────────────────────
# mirror-slope check (the trig verification)
# ─────────────────────────────────────────────────────────────
def mirror_slope_report(left: Line, right: Line, *, tol_deg: float = 1.0) -> dict[str, Any]:
    """Compare two edges that *should* be mirror images: equal angle-from-vertical.

    Returns each edge's unsigned angle from vertical and their difference; a
    ``delta_deg`` above ``tol_deg`` means the two edges are not mirror-symmetric
    (e.g. an apex shifted off the axis makes one leg steeper than the other).
    """
    la, ra = left.angle_from_vertical_deg, right.angle_from_vertical_deg
    return {
        "left_angle_from_vertical_deg": round(la, 3),
        "right_angle_from_vertical_deg": round(ra, 3),
        "delta_deg": round(abs(la - ra), 3),
        "tol_deg": tol_deg,
        "symmetric": abs(la - ra) <= tol_deg,
    }


# ─────────────────────────────────────────────────────────────
# bundled consistency report (the metric a luminance diff can't be)
# ─────────────────────────────────────────────────────────────
def consistency_report(*, symmetry_pairs: Sequence[tuple[Point, Point]] | None = None,
                       collinear_groups: Sequence[Sequence[Point]] | None = None,
                       tol: float = 2.0) -> dict[str, Any]:
    """Bundle the internal-consistency checks a rigid geometric mark should satisfy.

    ``symmetry_pairs`` are ``(left, right)`` points that should be bilaterally symmetric;
    ``collinear_groups`` are point-lists that should each lie on one straight edge.
    Returns the symmetry report and per-group collinearity residual plus a single
    ``worst_dev_px`` — the number to drive to zero. This is the signal that catches a
    single-corner offset (a 9 px apex shift) or an edge kink that a whole-image luminance
    match is blind to, so it complements the edge-distance score, not replaces it.
    """
    out: dict[str, Any] = {"tol": tol}
    worst = 0.0
    if symmetry_pairs:
        pairs = [(tuple(pr[0]), tuple(pr[1])) for pr in symmetry_pairs]
        rep = symmetry_report(pairs, tol=tol)
        out["symmetry"] = rep
        worst = max(worst, rep["max_axis_dev_px"])
    if collinear_groups:
        cols = [collinearity_residual([tuple(p) for p in g]) for g in collinear_groups]
        out["collinearity"] = cols
        worst = max([worst] + [c["max_dist_px"] for c in cols])
    out["worst_dev_px"] = round(worst, 3)
    out["within_tol"] = worst <= tol
    return out
