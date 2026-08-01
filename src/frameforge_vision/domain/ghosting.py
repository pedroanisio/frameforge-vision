"""Per-object ghost vectors — where did the reference's content go? (recon F2)

Given a render and a reference of the same size (grayscale arrays), and the
rendered document's object boxes, each object's rendered patch is searched for
in the reference within a window around its own location. The best match's
displacement is the object's *ghost vector* — the correction an author would
otherwise eyeball off an overlay's double image.

Matching is zero-mean SSD converted to a 0..1 score; flat patches (nothing to
match on) and degenerate boxes are skipped rather than reported as zero — a
zero displacement must mean "measured and in place", never "unmeasurable".
numpy-only by design.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["ghost_vectors"]


def _clamp_box(box: Sequence[float], w: int, h: int,
               pad: float = 0.0) -> tuple[int, int, int, int] | None:
    x0 = max(0, int(round(box[0] - pad)))
    y0 = max(0, int(round(box[1] - pad)))
    x1 = min(w, int(round(box[0] + box[2] + pad)))
    y1 = min(h, int(round(box[1] + box[3] + pad)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1


def ghost_vectors(
    render: "np.ndarray",
    reference: "np.ndarray",
    boxes: Sequence[dict[str, Any]],
    *,
    search: int = 16,
    pad: float = 6.0,
    min_std: float = 2.0,
    max_objects: int = 40,
) -> list[dict[str, Any]]:
    """Displacement of each object's reference match, sorted by magnitude.

    ``boxes`` entries are ``{"id": str, "box": [x, y, w, h]}`` in render
    coordinates; ``render`` and ``reference`` must be same-size 2D arrays.
    Boxes are padded by ``pad`` before matching — an exact object bbox crops
    a *uniform* patch of a solid shape (its edges sit on the box), and edges
    are precisely what a match needs.
    Returns ``{"id", "box", "offset_px": [dx, dy], "score"}`` per measurable
    object — ``offset_px`` is where the reference's matching patch sits
    relative to the rendered object (add it to the object's position to land
    on the reference).
    """
    ren = np.asarray(render, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if ren.shape != ref.shape or ren.ndim != 2:
        raise ValueError("render and reference must be same-size 2D grayscale arrays")
    h, w = ren.shape
    out: list[dict[str, Any]] = []
    for entry in list(boxes)[:max_objects]:
        clamped = _clamp_box(entry.get("box") or (), w, h, pad=pad)
        if clamped is None:
            continue
        x0, y0, x1, y1 = clamped
        patch = ren[y0:y1, x0:x1]
        if float(patch.std()) < min_std:
            continue                     # flat: nothing to anchor a match on
        p = patch - patch.mean()
        p_norm = float(np.sqrt((p * p).sum()))
        if p_norm < 1e-9:
            continue
        best: tuple[float, int, int] | None = None
        for dy in range(-search, search + 1):
            ry0, ry1 = y0 + dy, y1 + dy
            if ry0 < 0 or ry1 > h:
                continue
            for dx in range(-search, search + 1):
                rx0, rx1 = x0 + dx, x1 + dx
                if rx0 < 0 or rx1 > w:
                    continue
                cand = ref[ry0:ry1, rx0:rx1]
                c = cand - cand.mean()
                c_norm = float(np.sqrt((c * c).sum()))
                if c_norm < 1e-9:
                    continue
                ncc = float((p * c).sum()) / (p_norm * c_norm)
                if best is None or ncc > best[0]:
                    best = (ncc, dx, dy)
        if best is None:
            continue
        score, dx, dy = best
        out.append({
            "id": str(entry.get("id") or "?"),
            "box": [x0, y0, x1 - x0, y1 - y0],
            "offset_px": [dx, dy],
            "score": round(score, 4),
        })
    out.sort(key=lambda v: (v["offset_px"][0] ** 2 + v["offset_px"][1] ** 2), reverse=True)
    return out
