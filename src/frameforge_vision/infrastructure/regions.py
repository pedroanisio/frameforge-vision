"""Region analysis — what closed/filled/stable regions does an image contain?

Consolidates the R&D region scripts (formerly repo-root ``closed_region_detector`` /
``region_fill`` / ``region_preprocess`` / ``consensus_regions`` / ``unique_regions``
and ``out/region_fill/flat_regions``) into one importable module with a single
canonical region type. Three detection methods, one funnel:

- ``closed``    — purely topological enclosed faces: label the *background* with
  4-connectivity (the dual of 8-connected strokes) and keep components that do not
  touch the image border. Works on any line art (floor plans, mazes, sketches).
- ``flat``      — fill partition: quantise colour (Otsu 2-level, or k-means with
  ``colors``), label connected components per colour. Every maximal uniform-fill
  area is one region — solid shapes and hollow interiors are found the same way.
  ``fill_erode`` reclassifies thin dark strokes as ``outline`` (the
  ``flat_regions`` solid-ink recovery: true fills survive erosion, outlines don't).
- ``consensus`` — ensemble mollified level sets: segment across a (sigma, level)
  grid and keep what the majority agrees on, then low-pass each boundary's Fourier
  descriptors into a C-infinity loop. Robust to parameter choice on tangled/open
  linework that has no intrinsic regions.

Everything funnels into :func:`detect_regions`, a pure function returning one
JSON-serializable dict (regions carry ``bbox_px`` + ``box_norm`` + centroid +
sampled fill + polygon/holes), with an optional overlay PNG. Normalised
coordinates route through ``frameforge_vision.domain.coordinates`` (the
coordinate authority) — this module never re-derives norm ⇄ px maths.

OpenCV/NumPy are imported lazily, so importing this module costs nothing until a
function runs (the package convention for the optional ``vision`` group).

The supported public surface is ``DetectedRegion`` plus ``detect_regions``, in
lockstep with :mod:`frameforge_vision.infrastructure`. Lower-level analysis
helpers remain available for explicit internal imports, but are intentionally
excluded from star imports and compatibility guarantees.

⚠ ARCHITECTURAL CONTRACT (PALS's LAW): thresholds, k-means palettes, and level-set
ensembles are heuristics, not ground truth. Region lists are *measurements to
verify* (render the overlay, check the numbers), never proof of image content.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..domain.coordinates import normalize_point

__all__ = [
    "DetectedRegion",
    "detect_regions",
]

METHODS = ("closed", "flat", "consensus")


# ─────────────────────────────────────────────────────────────
# canonical region type (replaces the four colliding notions:
# closed_region_detector.Region, region_preprocess.Region,
# region_fill.FilledRegion — and stays distinct from the
# caller-supplied image_compare.Region named box)
# ─────────────────────────────────────────────────────────────
@dataclass
class DetectedRegion:
    """One detected region, canonically in image pixels (top-left, +y down)."""

    id: int
    bbox_px: tuple[float, float, float, float]      # x, y, w, h
    area_px: float                                  # pixel count (closed/flat) or Green's area (consensus)
    centroid_px: tuple[float, float]
    closed: bool                                    # False == touches the image border
    kind: str                                       # 'enclosed' | 'open' | 'solid' | 'hollow' | 'outline' | 'smooth'
    fill_rgb: tuple[int, int, int] | None = None    # mean colour sampled from the source
    fill_hex: str | None = None
    holes: int = 0
    polygon: list[list[float]] | None = None        # external boundary (image px)
    hole_polygons: list[list[list[float]]] | None = None
    shape_class: int | None = None                  # set by clustering (1-based, largest first)

    def to_dict(self, width: float, height: float, *,
                include_polygons: bool = True) -> dict[str, Any]:
        x, y, w, h = self.bbox_px
        nx, ny = normalize_point(x, y, width, height)
        nw, nh = normalize_point(w, h, width, height)
        cx, cy = self.centroid_px
        ncx, ncy = normalize_point(cx, cy, width, height)
        out: dict[str, Any] = {
            "id": int(self.id),
            "kind": self.kind,
            "closed": bool(self.closed),
            "bbox_px": [round(float(v), 2) for v in self.bbox_px],
            "box_norm": [round(v, 6) for v in (nx, ny, nw, nh)],
            "area_px": round(float(self.area_px), 1),
            "centroid_px": [round(float(cx), 2), round(float(cy), 2)],
            "centroid_norm": [round(ncx, 6), round(ncy, 6)],
            "fill_rgb": list(self.fill_rgb) if self.fill_rgb is not None else None,
            "fill_hex": self.fill_hex,
            "holes": int(self.holes),
            "polygon": self.polygon if include_polygons else None,
        }
        if self.hole_polygons and include_polygons:
            out["hole_polygons"] = self.hole_polygons
        if self.shape_class is not None:
            out["shape_class"] = int(self.shape_class)
        return out


@dataclass
class RegionAnalysis:
    """A detection pass over one image: regions + the label map they live in.

    ``labels`` is an ``H×W int32`` map (pixel → region id, 0 = unassigned/strokes);
    ``mask`` is the method's binary working mask (strokes / consensus votes).
    """

    method: str
    width: int
    height: int
    regions: list[DetectedRegion]
    labels: Any
    mask: Any | None = None
    params: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def closed(self) -> list[DetectedRegion]:
        return [r for r in self.regions if r.closed]

    @property
    def open(self) -> list[DetectedRegion]:
        return [r for r in self.regions if not r.closed]


# ─────────────────────────────────────────────────────────────
# shared helpers
# ─────────────────────────────────────────────────────────────
def load_image(path: "str | Path", scale: float = 1.0):
    """Read a raster (BGR), or rasterise an ``.svg`` first so SVGs work directly."""
    import cv2

    p = str(path)
    if p.lower().endswith(".svg"):
        import os
        import tempfile

        from frameforge.rendering.infrastructure.cairo import rasterize_svg_cairo

        fd, tmp = tempfile.mkstemp(suffix=".png", prefix="fg-regions-")
        os.close(fd)
        try:
            rasterize_svg_cairo(Path(p).read_text(encoding="utf-8"), tmp, scale=scale)
            return cv2.imread(tmp, cv2.IMREAD_COLOR)
        finally:
            os.remove(tmp)
    return cv2.imread(p, cv2.IMREAD_COLOR)


def _as_image(image: Any):
    """Accept a path or an already-loaded BGR array; fail loudly otherwise."""
    if isinstance(image, (str, Path)):
        img = load_image(image)
        if img is None:
            raise ValueError(f"could not read image: {image}")
        return img
    if getattr(image, "ndim", 0) in (2, 3):
        return image
    raise ValueError("image must be a path or an H×W(x3) uint8 array")


def _fill_of(img, mask) -> tuple[tuple[int, int, int], str]:
    """Mean colour of ``mask`` sampled from the ORIGINAL image → ((R,G,B), '#RRGGBB')."""
    px = img[mask]
    if px.size == 0:
        return (0, 0, 0), "#000000"
    b, g, r = px.reshape(-1, 3).mean(axis=0)
    R, G, B = int(r), int(g), int(b)
    return (R, G, B), f"#{R:02X}{G:02X}{B:02X}"


def distinct_colors(n: int, seed: int = 7):
    """``n`` visually distinct, bright BGR colours (deterministic per seed)."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    hsv = np.stack(
        [
            (np.linspace(0, 179, num=n, endpoint=False) + rng.integers(0, 180)) % 180,
            rng.integers(150, 256, size=n),
            rng.integers(180, 256, size=n),
        ],
        axis=1,
    ).astype(np.uint8)
    return cv2.cvtColor(hsv[None, :, :], cv2.COLOR_HSV2BGR)[0]


def _region_polygons(labels, rid: int, bbox: tuple[int, int, int, int], *,
                     epsilon: float = 1.5):
    """(external polygon, hole polygons, hole count) of one labelled region.

    Contours are traced on the region's bbox crop (offset back to image px) and
    lightly simplified so the JSON stays bounded.
    """
    import cv2

    x, y, w, h = bbox
    crop = (labels[y:y + h, x:x + w] == rid).astype("uint8")
    cnts, hier = cv2.findContours(crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or not cnts:
        return None, None, 0
    hier = hier[0]
    ext_idx = max((i for i in range(len(cnts)) if hier[i][3] == -1),
                  key=lambda i: cv2.contourArea(cnts[i]), default=None)
    if ext_idx is None:
        return None, None, 0

    def _poly(c):
        ap = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
        return [[round(float(px) + x, 2), round(float(py) + y, 2)] for px, py in ap]

    ext = _poly(cnts[ext_idx])
    holes = [_poly(cnts[j]) for j in range(len(cnts))
             if hier[j][3] == ext_idx and len(cnts[j]) >= 4]
    return (ext if len(ext) >= 3 else None), (holes or None), len(holes)


# ─────────────────────────────────────────────────────────────
# method: closed (topological enclosed faces)
# ─────────────────────────────────────────────────────────────
def _binarize(gray, invert: bool, method: str, block: int, c: int):
    """uint8 mask where strokes (foreground) == 255."""
    import cv2

    base = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
    if method == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, base + cv2.THRESH_OTSU)
    elif method == "adaptive":
        block = block if block % 2 == 1 else block + 1      # OpenCV requires odd
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, base, block, c)
    else:
        raise ValueError(f"unknown threshold_method {method!r} (use 'otsu' or 'adaptive')")
    return mask


def detect_closed_regions(
    image: Any,
    *,
    invert: bool = False,
    auto_polarity: bool = True,
    threshold_method: str = "otsu",
    block: int = 35,
    c: int = 5,
    close: int = 2,
    open_gap: int = 0,
    min_area: float = 25.0,
    include_polygons: bool = True,
) -> RegionAnalysis:
    """Every maximal background area completely enclosed by strokes.

    4-connected background labelling is the topological dual of 8-connected
    strokes, so a 1px diagonal line acts as a real unbroken wall (Jordan-curve
    consistency). ``close`` seals hairline JPEG/anti-aliasing gaps; ``open_gap``
    erases strokes thinner than N px first (drop interior hatching).
    """
    import cv2
    import numpy as np

    img = _as_image(image)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bgr = img if img.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    strokes = _binarize(gray, invert, threshold_method, block, c)
    if auto_polarity and strokes.mean() > 127.0:            # strokes should be the minority
        strokes = _binarize(gray, not invert, threshold_method, block, c)
    if open_gap > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_gap, open_gap))
        strokes = cv2.morphologyEx(strokes, cv2.MORPH_OPEN, k)
    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        strokes = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, k)

    background = cv2.bitwise_not(strokes)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(background, connectivity=4)
    edge = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    border_ids = set(np.unique(edge).tolist())

    regions: list[DetectedRegion] = []
    for i in range(1, n):                                   # 0 == strokes
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        bbox = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        is_closed = i not in border_ids
        fill_rgb, fill_hex = _fill_of(bgr, labels == i)
        poly = holes = None
        n_holes = 0
        if include_polygons:
            poly, holes, n_holes = _region_polygons(labels, i, bbox)
        regions.append(DetectedRegion(
            id=i, bbox_px=tuple(float(v) for v in bbox), area_px=float(area),
            centroid_px=(float(centroids[i, 0]), float(centroids[i, 1])),
            closed=is_closed, kind="enclosed" if is_closed else "open",
            fill_rgb=fill_rgb, fill_hex=fill_hex, holes=n_holes,
            polygon=poly, hole_polygons=holes,
        ))
    regions.sort(key=lambda r: r.area_px, reverse=True)
    h, w = labels.shape
    return RegionAnalysis("closed", int(w), int(h), regions, labels, strokes,
                          params={"invert": invert, "auto_polarity": auto_polarity,
                                  "threshold_method": threshold_method, "block": block, "c": c,
                                  "close": close, "open_gap": open_gap, "min_area": min_area})


# ─────────────────────────────────────────────────────────────
# method: flat (fill partition + solid-ink/outline recovery)
# ─────────────────────────────────────────────────────────────
def solid_ink_regions(gray, *, invert: bool = False, fill_erode: int = 3, min_area: float = 80.0):
    """``(label_map, kept_ids, stats)`` for filled-ink shapes only.

    Thin outline strokes vanish under an erosion of ``fill_erode`` px; surviving
    components are kept at their FULL extent (the ``flat_regions`` recovery).
    """
    import cv2
    import numpy as np

    base = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
    _, ink = cv2.threshold(gray, 0, 255, base + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_erode, fill_erode))
    eroded = cv2.erode(ink, k)
    survivors = set(np.unique(lab[eroded > 0]).tolist()) - {0}
    kept = [i for i in survivors if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return lab, kept, stats


def _quantize(img, colors: "int | None"):
    """Per-pixel palette-index map. Otsu 2-level by default; k-means if ``colors>=2``."""
    import cv2

    if colors and colors >= 2:
        cv2.setRNGSeed(7)                                   # deterministic k-means++
        z = img.reshape(-1, 3).astype("float32")
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, lbl, _ = cv2.kmeans(z, colors, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        return lbl.reshape(img.shape[:2]).astype("int32")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (m > 0).astype("int32")                          # 0 = dark, 1 = light


def segment_fill_regions(
    image: Any,
    *,
    colors: "int | None" = None,
    min_area: float = 60.0,
    dark_thresh: float = 110.0,
    fill_erode: int = 0,
    include_polygons: bool = True,
) -> RegionAnalysis:
    """Every maximal area of uniform fill is one region (solid AND hollow alike).

    ``kind`` classifies by sampled luminance: ``solid`` (dark fill) / ``hollow``
    (light interior). With ``fill_erode > 0``, dark regions that vanish under an
    erosion of that many px are reclassified ``outline`` (a stroke, not a fill).
    """
    import cv2
    import numpy as np

    img = _as_image(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    palette = _quantize(img, colors)
    h, w = palette.shape
    glob = np.zeros((h, w), np.int32)
    erode_kernel = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_erode, fill_erode))
                    if fill_erode > 0 else None)

    regions: list[DetectedRegion] = []
    nid = 1
    for v in np.unique(palette):
        mask = (palette == v).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            rmask = lab == i
            glob[rmask] = nid
            fill_rgb, fill_hex = _fill_of(img, rmask)
            lum = 0.299 * fill_rgb[0] + 0.587 * fill_rgb[1] + 0.114 * fill_rgb[2]
            kind = "solid" if lum < dark_thresh else "hollow"
            bbox = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                    int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            if kind == "solid" and erode_kernel is not None:
                x, y, bw, bh = bbox
                crop = rmask[y:y + bh, x:x + bw].astype(np.uint8)
                # borderValue=0: outside the bbox is not-region, so strokes touching
                # the crop edge still erode away (cv2's default border keeps them).
                if not cv2.erode(crop, erode_kernel, borderValue=0).any():
                    kind = "outline"                        # thin stroke, not a true fill
            regions.append(DetectedRegion(
                id=nid, bbox_px=tuple(float(b) for b in bbox), area_px=float(area),
                centroid_px=(float(cent[i, 0]), float(cent[i, 1])),
                closed=True, kind=kind, fill_rgb=fill_rgb, fill_hex=fill_hex,
            ))
            nid += 1

    edge = np.concatenate([glob[0, :], glob[-1, :], glob[:, 0], glob[:, -1]])
    border = set(np.unique(edge).tolist())
    for r in regions:
        if r.id in border:
            r.closed = False
    if include_polygons:
        for r in regions:
            bbox = tuple(int(v) for v in r.bbox_px)
            r.polygon, r.hole_polygons, r.holes = _region_polygons(glob, r.id, bbox)
    regions.sort(key=lambda r: r.area_px, reverse=True)
    return RegionAnalysis("flat", int(w), int(h), regions, glob, None,
                          params={"colors": colors, "min_area": min_area,
                                  "dark_thresh": dark_thresh, "fill_erode": fill_erode})


# ─────────────────────────────────────────────────────────────
# method: consensus (ensemble mollified level sets → smooth loops)
# ─────────────────────────────────────────────────────────────
def mollify(image: Any, sigma: float):
    """Otsu ink indicator convolved with a Gaussian → smooth field ``f`` in [0, 1]."""
    import cv2

    img = _as_image(image)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    f = cv2.GaussianBlur(ink.astype("float32"), (0, 0), sigma)
    return f / max(float(f.max()), 1e-6)


def _resample(P, m: int):
    import numpy as np

    P = np.vstack([P, P[0]])
    seg = np.sqrt((np.diff(P, axis=0) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return None
    u = np.linspace(0.0, s[-1], m, endpoint=False)
    return np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])


def smooth_loop(P, harmonics: int, m: int = 512):
    """C-infinity closed loop: arc-length resample → low-pass Fourier descriptors."""
    import numpy as np

    r = _resample(np.asarray(P, dtype=float), m)
    if r is None:
        return None
    x, y = r
    z = x + 1j * y
    Z = np.fft.fft(z)
    k = np.fft.fftfreq(m, 1.0 / m)
    Z[np.abs(k) > harmonics] = 0
    zr = np.fft.ifft(Z)
    return np.column_stack([zr.real, zr.imag])


def green_area(P) -> float:
    """Region area from the boundary alone (Green's theorem / shoelace)."""
    import numpy as np

    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def ensemble_vote(indicator, sigmas: Sequence[float], levels: Sequence[float]):
    """Sum of binary level-set masks across the (sigma, level) grid → ``(vote, N)``."""
    import cv2
    import numpy as np

    vote = np.zeros(indicator.shape, np.float32)
    n = 0
    for s in sigmas:
        f = cv2.GaussianBlur(indicator, (0, 0), float(s))
        f /= max(float(f.max()), 1e-6)
        for level in levels:
            vote += (f >= float(level)).astype(np.float32)
            n += 1
    return vote, n


def _smooth_regions_from_mask(mask, img, harmonics: int, min_area: float):
    """Smooth-boundary regions (+ label map) from a binary mask."""
    import cv2
    import numpy as np

    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    regions: list[DetectedRegion] = []
    if hier is None:
        return regions, labels
    hier = hier[0]
    edge_touch = np.zeros((h, w), bool)
    edge_touch[0, :] = edge_touch[-1, :] = edge_touch[:, 0] = edge_touch[:, -1] = True
    nid = 1
    for i, c in enumerate(cnts):
        if hier[i][3] != -1 or len(c) < 8:                  # holes handled with their parent
            continue
        ext = smooth_loop(c.reshape(-1, 2), harmonics)
        if ext is None:
            continue
        area = green_area(ext)
        holes = []
        for j, cj in enumerate(cnts):
            if hier[j][3] == i and len(cj) >= 8:
                hl = smooth_loop(cj.reshape(-1, 2), harmonics)
                if hl is not None:
                    holes.append(hl)
                    area -= green_area(hl)
        if area < min_area:
            continue
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [ext.astype(np.int32)], 255)
        for hl in holes:
            cv2.fillPoly(m, [hl.astype(np.int32)], 0)
        if int(m.sum()) == 0:
            continue
        fill_rgb, fill_hex = _fill_of(img, m > 0)
        xs, ys = ext[:, 0], ext[:, 1]
        labels[m > 0] = nid
        regions.append(DetectedRegion(
            id=nid,
            bbox_px=(float(xs.min()), float(ys.min()),
                     float(xs.max() - xs.min()), float(ys.max() - ys.min())),
            area_px=float(area),
            centroid_px=(float(xs.mean()), float(ys.mean())),
            closed=not bool((m > 0)[edge_touch].any()),
            kind="smooth", fill_rgb=fill_rgb, fill_hex=fill_hex, holes=len(holes),
            polygon=[[round(float(x), 2), round(float(y), 2)] for x, y in ext],
            hole_polygons=[[[round(float(x), 2), round(float(y), 2)] for x, y in hl]
                           for hl in holes] or None,
        ))
        nid += 1
    regions.sort(key=lambda r: r.area_px, reverse=True)
    return regions, labels


def consensus_smooth_regions(
    image: Any,
    *,
    sigmas: Sequence[float] = (4.0, 6.0, 8.0, 10.0),
    levels: Sequence[float] = (0.25, 0.30, 0.35, 0.40),
    agree: float = 0.5,
    harmonics: int = 24,
    min_area: float = 220.0,
) -> RegionAnalysis:
    """Ensemble consensus regions: keep what >= ``agree`` of the (sigma, level) grid
    calls region, then smooth each boundary (Fourier low-pass) and sample its fill."""
    import cv2

    img = _as_image(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    indicator = (ink > 0).astype("float32")
    vote, n = ensemble_vote(indicator, sigmas, levels)
    consensus = (vote >= math.ceil(n * agree)).astype("uint8")
    regions, labels = _smooth_regions_from_mask(consensus, img, harmonics, min_area)
    h, w = consensus.shape
    extras: dict[str, Any] = {"members": n, "agree": float(agree),
                              "sigmas": [float(s) for s in sigmas],
                              "levels": [float(v) for v in levels]}
    if consensus.sum():
        extras["mean_agreement"] = round(float((vote / n)[consensus > 0].mean()), 4)
    return RegionAnalysis("consensus", int(w), int(h), regions, labels, consensus,
                          params={"sigmas": list(sigmas), "levels": list(levels),
                                  "agree": agree, "harmonics": harmonics,
                                  "min_area": min_area},
                          extras=extras)


def smooth_regions_svg(analysis: RegionAnalysis, *, stroke: "str | None" = None,
                       bg: str = "#ffffff") -> str:
    """Clean SVG of an analysis' smooth/polygonal regions (evenodd holes)."""
    def path_d(P):
        return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in P) + " Z"

    w, h = analysis.width, analysis.height
    sa = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
             f'viewBox="0 0 {w} {h}" shape-rendering="geometricPrecision">',
             f'<rect width="{w}" height="{h}" fill="{bg}"/>']
    for r in analysis.regions:
        if not r.polygon:
            continue
        d = path_d(r.polygon) + "".join(" " + path_d(hl) for hl in (r.hole_polygons or []))
        cx, cy = r.centroid_px
        lines.append(f'<path d="{d}" fill="{r.fill_hex or "#888888"}" fill-rule="evenodd"{sa} '
                     f'data-area="{r.area_px:.0f}" data-centroid="{cx:.1f},{cy:.1f}" '
                     f'data-holes="{r.holes}"/>')
    lines.append("</svg>")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# shape-equivalence clustering (unique_regions capability)
# ─────────────────────────────────────────────────────────────
def _region_mask(labels, r: DetectedRegion):
    x, y, w, h = (int(v) for v in r.bbox_px)
    return (labels[y:y + h, x:x + w] == r.id).astype("uint8")


def _iou_top_left(a, b) -> float:
    """IoU of two masks aligned at their top-left corner (translation test)."""
    import numpy as np

    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    pa = np.zeros((h, w), np.uint8)
    pa[: a.shape[0], : a.shape[1]] = a
    pb = np.zeros((h, w), np.uint8)
    pb[: b.shape[0], : b.shape[1]] = b
    inter = np.count_nonzero(pa & pb)
    union = np.count_nonzero(pa | pb)
    return inter / union if union else 0.0


def _congruent_feat(mask):
    """Orientation/reflection-invariant descriptor:
    [n_polygon_vertices, side_min, side_max, fill_ratio]. Deliberately NOT Hu
    moments, which are swamped by rasterization noise on thin tiles."""
    import cv2
    import numpy as np

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea) if cnts else None
    if cnt is None or len(cnt) < 3:
        return None
    (_, _), (w, h), _ = cv2.minAreaRect(cnt)
    if w <= 0 or h <= 0:
        return None
    n = len(cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True))
    sides = sorted((float(w), float(h)))
    fill = cv2.contourArea(cnt) / (w * h)
    return np.array([n, sides[0], sides[1], fill], dtype=float)


def _feat_match(f1, f2, rtol: float) -> bool:
    if f1[0] != f2[0]:                                      # vertex count must agree
        return False
    for i in (1, 2):                                        # side lengths, relative
        if abs(f1[i] - f2[i]) > rtol * max(f1[i], f2[i]):
            return False
    return abs(f1[3] - f2[3]) <= rtol                       # fill ratio, absolute


def cluster_regions(analysis: RegionAnalysis, *, mode: str = "translation",
                    tol: float = 0.90) -> list[dict[str, Any]]:
    """Greedy shape-equivalence clustering (prototypes seeded largest-first).

    ``translation`` merges on top-left-aligned mask IoU >= ``tol`` (same shape AND
    orientation — the repeated-tile count); ``congruent`` merges on the invariant
    descriptor within ``1 - tol`` (pure shape, any pose). Sets ``shape_class`` on
    each clustered region (1-based, largest class first) and returns class
    summaries ``[{"class", "count", "median_area_px", "region_ids"}, ...]``.
    """
    import numpy as np

    if mode not in ("translation", "congruent"):
        raise ValueError("cluster mode must be 'translation' or 'congruent'")
    rtol = 1.0 - tol

    prototypes: list[dict[str, Any]] = []
    for r in sorted(analysis.regions, key=lambda x: x.area_px, reverse=True):
        mask = _region_mask(analysis.labels, r)
        if mode == "translation":
            key = mask
            matched = next((p for p in prototypes if _iou_top_left(mask, p["key"]) >= tol), None)
        else:
            key = _congruent_feat(mask)
            if key is None:
                continue
            matched = next((p for p in prototypes if _feat_match(key, p["key"], rtol)), None)
        if matched:
            matched["members"].append(r)
        else:
            prototypes.append({"key": key, "members": [r]})

    classes = sorted((p["members"] for p in prototypes), key=len, reverse=True)
    out = []
    for ci, members in enumerate(classes, 1):
        for r in members:
            r.shape_class = ci
        out.append({
            "class": ci,
            "count": len(members),
            "median_area_px": float(np.median([m.area_px for m in members])),
            "region_ids": [int(m.id) for m in members],
        })
    return out


# ─────────────────────────────────────────────────────────────
# overlay rendering
# ─────────────────────────────────────────────────────────────
def render_overlay(analysis: RegionAnalysis, *, mode: str = "fill", alpha: float = 0.65):
    """Annotated BGR canvas: each region painted (sampled fill or a distinct colour
    per region), region borders drawn, and a one-line banner with the counts."""
    import cv2
    import numpy as np

    h, w = analysis.labels.shape
    canvas = np.full((h, w, 3), 255, np.uint8)
    fill = canvas.copy()
    if mode == "fill" and all(r.fill_rgb is not None for r in analysis.regions):
        for r in analysis.regions:
            R, G, B = r.fill_rgb
            fill[analysis.labels == r.id] = (B, G, R)
    else:
        colors = distinct_colors(max(len(analysis.regions), 1))
        for color, r in zip(colors, analysis.regions):
            fill[analysis.labels == r.id] = color.tolist()
    body = cv2.addWeighted(fill, alpha, canvas, 1 - alpha, 0)
    borders = np.zeros((h, w), bool)
    borders[:, :-1] |= analysis.labels[:, :-1] != analysis.labels[:, 1:]
    borders[:-1, :] |= analysis.labels[:-1, :] != analysis.labels[1:, :]
    body[borders] = (60, 60, 60)
    if analysis.mask is not None and analysis.method == "closed":
        body[analysis.mask > 0] = (0, 0, 0)                 # strokes on top

    band = 28
    out = np.full((h + band, w, 3), 255, np.uint8)
    out[band:] = body
    banner = (f"{analysis.method}: {len(analysis.regions)} regions   "
              f"(closed: {len(analysis.closed)} | open: {len(analysis.open)})")
    cv2.putText(out, banner, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────
# the funnel
# ─────────────────────────────────────────────────────────────
_METHOD_TUNABLES = {
    "closed": {"invert", "auto_polarity", "threshold_method", "block", "c",
               "close", "open_gap", "min_area"},
    "flat": {"colors", "min_area", "dark_thresh", "fill_erode"},
    "consensus": {"sigmas", "levels", "agree", "harmonics", "min_area"},
}


def detect_regions(
    image: Any,
    method: str = "consensus",
    *,
    overlay_path: "str | Path | None" = None,
    include_polygons: bool = True,
    cluster: "str | None" = None,
    cluster_tol: float = 0.90,
    max_regions: int = 400,
    **tunables: Any,
) -> dict[str, Any]:
    """Detect an image's regions with one of the three methods (module doc) and
    return ONE JSON-serializable dict.

    ``image`` is a raster path (or ``.svg``, rasterised first, or a loaded BGR
    array). ``tunables`` are method-specific (unknown names raise ``TypeError`` so
    a typo can't silently run with defaults). ``cluster`` optionally groups the
    regions into shape-equivalence classes (``'translation'`` / ``'congruent'``);
    ``overlay_path`` writes an annotated PNG. ``max_regions`` bounds the reported
    list (largest-area first) so the payload stays tractable.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    allowed = _METHOD_TUNABLES[method]
    unknown = set(tunables) - allowed
    if unknown:
        raise TypeError(f"unknown tunable(s) for method {method!r}: {sorted(unknown)} "
                        f"(allowed: {sorted(allowed)})")

    img = _as_image(image)
    if method == "closed":
        analysis = detect_closed_regions(img, include_polygons=include_polygons, **tunables)
    elif method == "flat":
        analysis = segment_fill_regions(img, include_polygons=include_polygons, **tunables)
    else:
        analysis = consensus_smooth_regions(img, **tunables)

    classes = None
    if cluster:
        classes = cluster_regions(analysis, mode=cluster, tol=cluster_tol)

    written: "str | None" = None
    if overlay_path is not None:
        import cv2

        written = str(overlay_path)
        if not cv2.imwrite(written, render_overlay(analysis)):
            raise ValueError(f"could not write overlay to {written!r}")

    kept = analysis.regions[: max(0, int(max_regions))]
    result: dict[str, Any] = {
        "ok": True,
        "method": method,
        "image": {"path": str(image) if isinstance(image, (str, Path)) else None,
                  "width_px": analysis.width, "height_px": analysis.height},
        "params": _jsonify(analysis.params),
        "region_count": len(analysis.regions),
        "closed_count": len(analysis.closed),
        "open_count": len(analysis.open),
        "regions": [r.to_dict(analysis.width, analysis.height,
                              include_polygons=include_polygons) for r in kept],
        "overlay_path": written,
        "classes": classes,
    }
    if method == "consensus":
        result["ensemble"] = _jsonify(analysis.extras)
    return result


def _jsonify(value: Any) -> Any:
    """Native-type deep copy (numpy scalars → Python scalars) for JSON safety."""
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


# ─────────────────────────────────────────────────────────────
# CLI (shared by the root-script deprecation shims)
# ─────────────────────────────────────────────────────────────
def main(argv: "list[str] | None" = None, *, default_method: str = "consensus") -> int:
    """``python -m frameforge_vision.infrastructure.regions img.png --method flat``"""
    import argparse
    import json as _json

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="raster or .svg")
    p.add_argument("--method", choices=METHODS, default=default_method)
    p.add_argument("--min-area", type=float, default=None)
    p.add_argument("--colors", type=int, default=None, help="flat: k-means palette size")
    p.add_argument("--dark-thresh", type=float, default=None, help="flat: solid-vs-hollow luminance")
    p.add_argument("--fill-erode", type=int, default=None, help="flat: outline-vs-fill erosion px")
    p.add_argument("--invert", action="store_true", help="closed: light strokes on dark")
    p.add_argument("--close", type=int, default=None, help="closed: gap-sealing kernel px")
    p.add_argument("--open-gap", type=int, default=None, help="closed: erase strokes thinner than N px")
    p.add_argument("--sigmas", default=None, help="consensus: comma list, e.g. 4,6,8,10")
    p.add_argument("--levels", default=None, help="consensus: comma list, e.g. 0.25,0.3,0.35,0.4")
    p.add_argument("--agree", type=float, default=None, help="consensus: agreement fraction")
    p.add_argument("--harmonics", type=int, default=None, help="consensus: Fourier harmonics per loop")
    p.add_argument("--cluster", choices=["translation", "congruent"], default=None)
    p.add_argument("--cluster-tol", type=float, default=0.90)
    p.add_argument("--overlay", default=None, help="write an annotated overlay PNG here")
    p.add_argument("--no-polygons", action="store_true")
    a = p.parse_args(argv)

    tunables: dict[str, Any] = {}
    if a.min_area is not None:
        tunables["min_area"] = a.min_area
    if a.method == "flat":
        for k, v in (("colors", a.colors), ("dark_thresh", a.dark_thresh),
                     ("fill_erode", a.fill_erode)):
            if v is not None:
                tunables[k] = v
    if a.method == "closed":
        if a.invert:
            tunables["invert"] = True
        for k, v in (("close", a.close), ("open_gap", a.open_gap)):
            if v is not None:
                tunables[k] = v
    if a.method == "consensus":
        if a.sigmas:
            tunables["sigmas"] = [float(x) for x in a.sigmas.split(",")]
        if a.levels:
            tunables["levels"] = [float(x) for x in a.levels.split(",")]
        for k, v in (("agree", a.agree), ("harmonics", a.harmonics)):
            if v is not None:
                tunables[k] = v

    try:
        result = detect_regions(a.image, a.method, overlay_path=a.overlay,
                                include_polygons=not a.no_polygons,
                                cluster=a.cluster, cluster_tol=a.cluster_tol, **tunables)
    except (ValueError, TypeError) as exc:
        import sys as _sys

        print(f"error: {exc}", file=_sys.stderr)
        return 2
    print(_json.dumps(result, indent=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
