"""Inverse primitive fitting (Gap G1): region mask → spine + width profile.

The forward primitive is ``sdk.outline.stroke_outline`` — a spine polyline, a
maximum width, and a width profile make a flame/petal/brush shape. This module
is its inverse: given an observed region mask, recover those parameters, so a
measured shape becomes an AUTHORED parametric object (one named spec-table row)
instead of a traced outline. The authored-lane fidelity ceiling was exactly
this gap: hand-guessed spines plateau; fitted spines land.

Pipeline (deterministic, pure NumPy, lazily imported):

1. **Thinning** — Zhang–Suen topological skeletonisation of the mask.
2. **Main spine** — the longest path across the skeleton graph (double BFS),
   which naturally ignores side spurs; each end is then extended along its
   tangent to the region tip. Degenerate skeletons (disks, blobs) fall back
   to the principal-axis chord — and the reported ``elongation``
   (length / width_max) tells the caller how spine-like the region really is.
3. **Widths** — the chamfer distance field sampled along the spine: full
   width ``2·d``, its max, a normalized profile, and the peak position — the
   exact vocabulary ``stroke_outline`` speaks.
4. **Compact form** — one least-squares cubic (endpoints anchored) over the
   arc-length-uniform spine, with its rms reported so the caller knows when a
   single cubic is honest.

Everything is rounded to 2 dp: results are JSON-ready and run-deterministic.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

_MIN_PIXELS = 64
_MIN_SKELETON_PATH = 4
_TIP_EXTEND_CAP = 200


def chamfer_distance(mask, *, max_rounds: int = 64):
    """Distance-to-boundary in EROSION rounds (octagonal metric), 0 outside.

    Owned by the domain (G1 moved it here from the vectorize infrastructure,
    which keeps an alias): a vectorised shell-peel transform whose round r
    labels the r-th 1-px shell. Rounds ALTERNATE 4- and 8-connected erosion,
    so the metric ball is an octagon — within a few percent of Euclidean in
    every direction (pure 8-connected peeling is chessboard distance, which
    under-measures diagonal widths by up to 29% and made rebuilt diagonal
    strokes visibly thin). Axis-aligned distances are unchanged (a rect
    centre still sits at half its short side). Interiors deeper than
    ``max_rounds`` clip to it — thresholds below the cap stay exact.
    """
    import numpy as np

    cur = np.asarray(mask) > 0
    dist = np.zeros(cur.shape, dtype=np.float32)
    n4 = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
    n8 = tuple((dy, dx) for dy in (0, 1, 2) for dx in (0, 1, 2))
    r = 0
    while cur.any():
        r += 1
        dist[cur] = float(r)
        if r >= max_rounds:
            break
        padded = np.pad(cur, 1, mode="constant")
        nxt = cur.copy()
        for dy, dx in (n4 if r % 2 == 1 else n8):
            nxt &= padded[dy:dy + cur.shape[0], dx:dx + cur.shape[1]]
        cur = nxt
    return dist


def _neighbors(padded, np):
    """The 8 neighbour planes P2..P9 (N, NE, E, SE, S, SW, W, NW) of a mask."""
    h, w = padded.shape[0] - 2, padded.shape[1] - 2
    sl = lambda dy, dx: padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]  # noqa: E731
    return [sl(-1, 0), sl(-1, 1), sl(0, 1), sl(1, 1),
            sl(1, 0), sl(1, -1), sl(0, -1), sl(-1, -1)]


def _thin(mask, np, max_iter: int = 400):
    """Zhang–Suen thinning to a 1-px 8-connected skeleton."""
    img = (np.asarray(mask) > 0).astype(np.uint8)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            padded = np.pad(img, 1, mode="constant")
            p = _neighbors(padded, np)                # P2..P9
            b = sum(x.astype(np.int32) for x in p)
            ring = p + [p[0]]
            a = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int32)
                    for i in range(8))
            if step == 0:
                cond = ((p[0] * p[2] * p[4]) == 0) & ((p[2] * p[4] * p[6]) == 0)
            else:
                cond = ((p[0] * p[2] * p[6]) == 0) & ((p[0] * p[4] * p[6]) == 0)
            kill = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if kill.any():
                img[kill] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


_STEPS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _longest_path(skel, np):
    """The longest path across the skeleton graph via double BFS.

    Thinning leaves stray isolated pixels and occasional islets, so the walk
    runs on the LARGEST connected component; side spurs are ignored by the
    double-BFS construction. Returns a list of (y, x) pixels."""
    from collections import deque

    ys, xs = np.nonzero(skel)
    nodes = sorted(zip((int(v) for v in ys), (int(v) for v in xs)))
    if not nodes:
        return []
    node_set = set(nodes)

    def bfs(start, allowed):
        seen = {start: None}
        queue = deque([start])
        last = start
        while queue:
            cur = queue.popleft()
            last = cur
            for dy, dx in _STEPS:
                nxt = (cur[0] + dy, cur[1] + dx)
                if nxt in allowed and nxt not in seen:
                    seen[nxt] = cur
                    queue.append(nxt)
        return last, seen

    # largest connected component (deterministic: seeds in sorted order)
    unvisited = set(node_set)
    component: set = set()
    for seed in nodes:
        if seed not in unvisited:
            continue
        _, seen = bfs(seed, unvisited)
        unvisited -= seen.keys()
        if len(seen) > len(component):
            component = set(seen.keys())
    start = min(component)
    far, _ = bfs(start, component)
    end, parents = bfs(far, component)
    path = [end]
    while parents[path[-1]] is not None:
        path.append(parents[path[-1]])
    return path[::-1]


def _extend_to_tip(path, mask, np):
    """Extend each spine end along its tangent until it exits the region."""
    h, w = mask.shape

    def walk(end, prev):
        dy, dx = end[0] - prev[0], end[1] - prev[1]
        n = math.hypot(dy, dx) or 1.0
        dy, dx = dy / n, dx / n
        out = []
        y, x = float(end[0]), float(end[1])
        for _ in range(_TIP_EXTEND_CAP):
            y, x = y + dy, x + dx
            iy, ix = int(round(y)), int(round(x))
            if not (0 <= iy < h and 0 <= ix < w) or not mask[iy, ix]:
                break
            out.append((iy, ix))
        return out

    if len(path) < 2:
        return path
    k = min(6, len(path) - 1)
    head = walk(path[0], path[k])
    tail = walk(path[-1], path[-1 - k])
    return list(reversed(head)) + path + tail


def _resample(points, n, np):
    """Arc-length-uniform resampling of an (x, y) polyline to n+1 points."""
    pts = np.asarray(points, dtype=np.float64)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 0:
        return [tuple(pts[0])] * (n + 1), 0.0
    targets = np.linspace(0.0, total, n + 1)
    out_x = np.interp(targets, cum, pts[:, 0])
    out_y = np.interp(targets, cum, pts[:, 1])
    return list(zip(out_x, out_y)), total


def _fit_cubic(points, np):
    """Least-squares cubic with anchored endpoints; returns (controls, rms)."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    t = np.linspace(0.0, 1.0, n)
    u = 1.0 - t
    b0, b1, b2, b3 = u**3, 3 * u * u * t, 3 * u * t * t, t**3
    p0, p3 = pts[0], pts[-1]
    r = pts - np.outer(b0, p0) - np.outer(b3, p3)
    m = np.array([[float((b1 * b1).sum()), float((b1 * b2).sum())],
                  [float((b1 * b2).sum()), float((b2 * b2).sum())]])
    ctrl = []
    for axis in (0, 1):
        rhs = np.array([float((b1 * r[:, axis]).sum()),
                        float((b2 * r[:, axis]).sum())])
        ctrl.append(np.linalg.solve(m, rhs))
    c1 = (ctrl[0][0], ctrl[1][0])
    c2 = (ctrl[0][1], ctrl[1][1])
    curve = (np.outer(b0, p0) + np.outer(b1, c1) + np.outer(b2, c2)
             + np.outer(b3, p3))
    rms = float(np.sqrt(((curve - pts) ** 2).sum(axis=1).mean()))
    return [list(p0), list(c1), list(c2), list(p3)], rms


def _principal_chord(mask, np):
    """PCA-axis fallback spine for degenerate skeletons (disks, blobs)."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    centre = pts.mean(axis=0)
    centred = pts - centre
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    proj = centred @ axis
    lo = centre + axis * float(proj.min())
    hi = centre + axis * float(proj.max())
    return [(int(round(lo[1])), int(round(lo[0]))),
            (int(round(centre[1])), int(round(centre[0]))),
            (int(round(hi[1])), int(round(hi[0])))]


def fit_spine(
    mask,
    *,
    samples: int = 32,
    profile_samples: int = 16,
    min_pixels: int = _MIN_PIXELS,
) -> dict[str, Any]:
    """Fit spine + width profile to a region mask (the stroke_outline inverse).

    Returns ``{"spine", "cubic", "cubic_rms", "width_max", "profile", "peak",
    "length", "elongation"}`` — ``spine`` is the arc-length-uniform polyline
    (``samples``+1 image-space points, tip-to-tip, direction free), ``cubic``
    its anchored 4-control-point least-squares form, ``profile`` the
    normalized width at ``profile_samples`` uniform positions. Raises
    ``ValueError`` on masks below ``min_pixels``.
    """
    import numpy as np

    m = np.asarray(mask) > 0
    area = int(m.sum())
    if area < max(1, int(min_pixels)):
        raise ValueError(
            f"region has {area} pixels — fit_spine needs at least {min_pixels} "
            "to say anything honest about a spine")

    skel = _thin(m, np)
    path = _longest_path(skel, np)
    if len(path) >= _MIN_SKELETON_PATH:
        path = _extend_to_tip(path, m, np)
    else:
        path = _principal_chord(m, np)
        path = _extend_to_tip(path, m, np)
    spine_yx = path
    spine_xy = [(float(x), float(y)) for y, x in spine_yx]
    spine, length = _resample(spine_xy, samples, np)

    h, w = m.shape

    def _inside(x, y):
        iy, ix = int(round(y)), int(round(x))
        return 0 <= iy < h and 0 <= ix < w and bool(m[iy, ix])

    def _perp_width(idx):
        """Exact width: the perpendicular chord through the mask at spine[idx]."""
        x, y = spine[idx]
        j, k = min(idx + 1, samples), max(idx - 1, 0)
        dx, dy = spine[j][0] - spine[k][0], spine[j][1] - spine[k][1]
        n = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / n, dx / n
        total = 0.0
        for sgn in (1.0, -1.0):
            step = 0.0
            while step < 512.0 and _inside(x + nx * sgn * (step + 0.5),
                                           y + ny * sgn * (step + 0.5)):
                step += 0.5
            total += step
        return total

    widths = []
    for i in range(profile_samples):
        t = i / (profile_samples - 1)
        widths.append(_perp_width(min(int(round(t * samples)), samples)))
    width_max = max(widths) or 1.0
    profile = [round(v / width_max, 4) for v in widths]
    peak = round(profile.index(max(profile)) / (profile_samples - 1), 4)

    cubic, cubic_rms = _fit_cubic(spine, np)
    return {
        "spine": [[round(x, 2), round(y, 2)] for x, y in spine],
        "cubic": [[round(v, 2) for v in p] for p in cubic],
        "cubic_rms": round(cubic_rms, 2),
        "width_max": round(width_max, 2),
        "profile": profile,
        "peak": peak,
        "length": round(length, 2),
        "elongation": round(length / width_max, 3),
    }


def spine_profile(samples: Sequence[float]):
    """A ``stroke_outline`` ``profile`` callable from fitted profile samples."""
    vals = [float(v) for v in samples]
    n = len(vals)
    if n == 1:
        return lambda t: vals[0]

    def profile(t):
        x = min(max(float(t), 0.0), 1.0) * (n - 1)
        i = min(int(x), n - 2)
        frac = x - i
        return vals[i] + (vals[i + 1] - vals[i]) * frac
    return profile


__all__ = ["chamfer_distance", "fit_spine", "spine_profile"]
