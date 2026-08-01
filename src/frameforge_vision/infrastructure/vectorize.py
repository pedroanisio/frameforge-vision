"""Raster → FrameForge vectorizer (the ingestion front-end).

Turns a bitmap into editable FrameForge vector objects with OpenCV (the optional
``vision`` dependency group). Two modes:

- ``region``  — k-means colour quantisation → per-colour contours → filled
  polygons. A flat, editable *vector base* of the image.
- ``outline`` — Canny edges → contours → polylines. The image's *full outline*
  as line art.

Emits plain FrameForge object dicts (no model import), so the renderer draws them
directly and the package boundary stays clean. OpenCV/NumPy are imported lazily,
so importing this module costs nothing until a function runs.

Honest limits: output is *polygonal* (straight segments), not Bézier-smooth — for
smooth curves route a Potrace/VTracer SVG through ``svg_import``. With the default
flat paint, soft gradients posterise into colour bands (control with ``colors``);
:func:`apply_gradient_fills` closes that gap by re-painting traced shapes with
linear/radial gradients fitted from the source (the pure fit lives in
``vision.domain.gradient_fit``). This is a faithful, editable base, not a semantic
one; per-object *named* layers need the segmentation/OCR tiers layered on top.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..domain.coordinates import denorm_box


def image_size(path: "str | Path") -> tuple[int, int]:
    """Return the (width, height) of an image without vectorising it."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"could not read image: {path}")
    h, w = img.shape[:2]
    return w, h


def _load_scaled(path, max_dim):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"could not read image: {path}")
    h, w = img.shape[:2]
    longest = max(w, h)
    if max_dim and longest > max_dim:
        s = max_dim / longest
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return img


def _hex_bgr(bgr) -> str:
    b, g, r = (int(v) for v in bgr)
    return "#%02X%02X%02X" % (r, g, b)


def _contour_polys(mask, *, detail, min_area, closed):
    import cv2
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL if closed else cv2.RETR_LIST,
                               cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if closed:
            if cv2.contourArea(c) < min_area:
                continue
            eps = max(0.5, detail * cv2.arcLength(c, True))
            ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(ap) >= 3:
                polys.append((cv2.contourArea(c), [[float(x), float(y)] for x, y in ap]))
        else:
            if cv2.arcLength(c, False) < min_area:
                continue
            ap = cv2.approxPolyDP(c, max(0.8, detail * 1000), False).reshape(-1, 2)
            if len(ap) >= 2:
                polys.append((cv2.arcLength(c, False), [[float(x), float(y)] for x, y in ap]))
    return polys


def raster_to_objects(
    path: "str | Path",
    *,
    mode: str = "region",
    colors: int = 10,
    detail: float = 0.004,
    min_area: float = 90.0,
    max_dim: int = 900,
    ink: str = "#1E2440",
    stroke_width: float = 1.0,
) -> tuple[list[dict[str, Any]], int, int]:
    """Vectorise ``path``; return ``(objects, width, height)`` in image pixels.

    ``mode='region'`` traces filled polygons over ``colors`` quantised colours;
    ``mode='outline'`` traces edges into polylines; ``mode='auto'`` classifies the
    raster (see :func:`resolve_auto_mode`) and picks between the two — routes this
    function cannot draw (``trace``/``layers``) clamp to their nearest equivalent
    (outline/region). ``detail`` is the Douglas–Peucker epsilon as a fraction of
    contour length (higher = simpler); ``min_area`` drops noise.
    """
    import cv2
    import numpy as np

    if mode == "auto":
        resolved, _ = resolve_auto_mode(path)
        mode = {"trace": "outline", "layers": "region"}.get(resolved, resolved)

    img = _load_scaled(path, max_dim)
    h, w = img.shape[:2]
    objs: list[dict[str, Any]] = []

    if mode == "region":
        z = img.reshape(-1, 3).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        k = max(2, int(colors))
        _, labels, centers = cv2.kmeans(z, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        centers = centers.astype(np.uint8)
        lab = labels.reshape(h, w)
        shapes = []
        kernel = np.ones((3, 3), np.uint8)
        for ci in range(k):
            mask = (lab == ci).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            hexc = _hex_bgr(centers[ci])
            for area, pts in _contour_polys(mask, detail=detail, min_area=min_area, closed=True):
                shapes.append((area, pts, hexc))
        shapes.sort(key=lambda t: -t[0])              # big shapes first, detail on top
        objs = [{"type": "polygon", "points": pts, "fill": hexc} for _, pts, hexc in shapes]
        return objs, w, h

    if mode == "outline":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Auto-Canny: thresholds from the image median, so contrast is adapted to
        # rather than guessed (a fixed 60/160 misses low-contrast edges).
        med = float(np.median(gray))
        lo = int(max(0, 0.55 * med))
        hi = int(min(255, max(lo + 30, 1.33 * med)))
        edges = cv2.Canny(gray, lo, hi)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        for _, pts in _contour_polys(edges, detail=detail, min_area=max(8.0, min_area / 3), closed=False):
            objs.append({"type": "polyline", "points": pts, "stroke": ink,
                         "stroke_style": {"stroke_width": stroke_width}})
        return objs, w, h

    raise ValueError(f"unknown mode {mode!r} (expected 'region' or 'outline')")


_POTRACE_HINT = (
    "smooth (Bézier) tracing needs the `potrace` binary on PATH (Debian/Ubuntu: "
    "`apt install potrace`; it ships in the FrameForge Docker image). The `region`/"
    "`outline` modes need no extra binary."
)


def potrace_path() -> str | None:
    """Return the potrace executable path, or None when it is not installed."""
    return shutil.which("potrace")


def _potrace_hex(color: str) -> str:
    """potrace's --color needs a 6-digit #rrggbb; expand a 3-digit #rgb shorthand."""
    c = (color or "").strip()
    if len(c) == 4 and c.startswith("#"):
        return "#" + "".join(ch * 2 for ch in c[1:])
    return c


def trace_to_svg(path: "str | Path", *, region_box: "list[float] | None" = None,
                 threshold: int | None = None, invert: Any = "auto",
                 turdsize: int = 2, alphamax: float = 1.0, opttolerance: float = 0.2,
                 fill: str = "#000000", supersample: int = 1) -> tuple[str, dict[str, Any]]:
    """Threshold + potrace-trace an image (or a normalized region) into SVG text.

    The smooth-curve complement to :func:`raster_to_objects`: potrace fits Bézier
    outlines to the thresholded ink, which the caller lowers to FrameForge objects
    via :func:`frameforge_vision.infrastructure.svg_import.svg_to_objects`
    (``box=region_px`` fits the traced crop back to its place in the full image).

    ``supersample`` (B5, 1..4) is the AA-aware subpixel stage: the grayscale is
    LANCZOS-upscaled BEFORE thresholding, so the binarisation locates the
    threshold crossing on a 1/s px grid instead of quantising the anti-aliased
    boundary to whole pixels; the caller's box-fit divides the s×-larger potrace
    viewport back down, landing the geometry subpixel-accurately in the same
    output coordinates. ``turdsize`` keeps SOURCE-pixel semantics (potrace sees
    s²-scaled areas, so ``turdsize * s²`` is passed through — reported as
    ``turdsize_effective``). Cost grows ~s².

    Returns ``(svg_text, meta)``; ``meta`` carries the pixel region, threshold,
    invert decision, supersample factors, and traced path count. ``image`` /
    ``region_px`` stay in SOURCE coordinates; ``traced_px`` is the (upscaled)
    bitmap potrace actually saw.
    """
    exe = potrace_path()
    if not exe:
        raise RuntimeError(_POTRACE_HINT)
    ss = int(supersample)
    if not 1 <= ss <= 4:
        raise ValueError(
            f"supersample must be 1..4 (got {supersample!r}); cost grows with s² — "
            "2..3 recovers the anti-aliased edge, 4 is diminishing returns")
    from PIL import Image, ImageStat

    img = Image.open(str(path)).convert("RGB")
    W, H = img.size
    if region_box:
        # the clamped norm→px lowering is the domain coordinate authority's —
        # the same denorm_box the MCP usecase applies to region/layers crops.
        x, y, w, h = region_box
        ox, oy, cw, ch = denorm_box(x, y, w, h, W, H)
        cw, ch = max(1.0, cw), max(1.0, ch)
        crop = img.crop((int(ox), int(oy), int(round(ox + cw)), int(round(oy + ch))))
    else:
        ox, oy, cw, ch, crop = 0.0, 0.0, float(W), float(H), img

    gray = crop.convert("L")
    mean = ImageStat.Stat(gray).mean[0]
    if ss > 1:
        gray = gray.resize((gray.width * ss, gray.height * ss), Image.LANCZOS)
    # potrace traces BLACK (0) as foreground; `invert=auto` makes the bright pixels
    # the foreground when the ground is dark (a light mark on a dark panel).
    do_invert = (mean < 127.0) if invert == "auto" else bool(invert)
    thr = int(threshold) if threshold is not None else 128
    bw = gray.point(lambda v: 255 if v >= thr else 0, mode="1")
    if do_invert:
        bw = bw.point(lambda v: 0 if v else 255, mode="1")

    turd_eff = int(turdsize) * ss * ss
    tmp = Path(tempfile.mkdtemp(prefix="fg-trace-"))
    try:
        src, out = tmp / "in.bmp", tmp / "out.svg"
        bw.save(src)  # 1-bit BMP; potrace reads BMP + PNM
        cmd = [exe, str(src), "--svg", "-o", str(out),
               "--turdsize", str(turd_eff), "--alphamax", str(float(alphamax)),
               "--opttolerance", str(float(opttolerance))]
        if fill:
            cmd += ["--color", _potrace_hex(fill)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"potrace failed: {proc.stderr.strip() or proc.returncode}")
        svg_text = out.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    meta = {
        "backend": "potrace",
        "image": {"width_px": W, "height_px": H},
        "region_px": [round(ox, 2), round(oy, 2), round(cw, 2), round(ch, 2)],
        "threshold": thr,
        "inverted": do_invert,
        "supersample": ss,
        "turdsize_effective": turd_eff,
        "traced_px": [bw.width, bw.height],
        "path_count": svg_text.count("<path"),
    }
    return svg_text, meta


def ocr_text_objects_status(path: "str | Path", *, max_dim: int = 1400,
                            min_conf: float = 60.0, color: str = "#1E2440",
                            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Detect text via Tesseract → ``(text objects, status)``.

    The status dict makes the degradation observable (PALS's Law: a silent ``[]``
    is indistinguishable from a text-free image):

    - ``available`` — could the OCR backend run here at all?
    - ``status``    — ``ok`` (words found) | ``no_text`` (backend ran, nothing kept)
      | ``unavailable`` (pytesseract/Tesseract missing) | ``error`` (backend crashed)
    - ``reason``    — human-readable cause when not ``ok``/``no_text``
    - ``n_words``   — number of emitted text objects
    """
    status: dict[str, Any] = {"available": False, "status": "unavailable",
                              "reason": None, "n_words": 0}
    try:
        import cv2
        import pytesseract
        from pytesseract import Output
    except Exception as exc:
        status["reason"] = (f"OCR dependency missing: {exc} — install the `vision` "
                            "group's pytesseract (plus OpenCV)")
        return [], status
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        status["reason"] = f"the Tesseract binary is missing or broken: {exc}"
        return [], status
    status["available"] = True

    img = _load_scaled(path, max_dim)
    try:
        data = pytesseract.image_to_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), output_type=Output.DICT)
    except Exception as exc:
        status["status"], status["reason"] = "error", f"OCR run failed: {exc}"
        return [], status
    out: list[dict[str, Any]] = []
    for i, word in enumerate(data.get("text", [])):
        word = (word or "").strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, KeyError):
            conf = -1.0
        if not word or conf < min_conf:
            continue
        x, y, ww, hh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append({"type": "text", "box": [float(x), float(y), float(ww), float(hh)], "text": word,
                    "style": {"font_family": ["Inter", "Arial", "sans-serif"],
                              "font_size": float(hh), "font_weight": 700, "color": color,
                              "vertical_align": "middle"}})
    status["status"] = "ok" if out else "no_text"
    status["n_words"] = len(out)
    return out, status


def ocr_text_objects(path: "str | Path", *, max_dim: int = 1400,
                     min_conf: float = 60.0, color: str = "#1E2440") -> list[dict[str, Any]]:
    """Detect text via Tesseract → editable FrameForge ``text`` objects.

    Back-compat wrapper over :func:`ocr_text_objects_status` — an empty list here
    cannot distinguish 'no text' from 'backend missing'; callers that surface OCR
    results MUST use the status variant instead.
    """
    objects, _ = ocr_text_objects_status(path, max_dim=max_dim, min_conf=min_conf, color=color)
    return objects


# --------------------------------------------------------------------------- #
#  auto-mode router  --  cheap classification → region / outline / trace / layers
# --------------------------------------------------------------------------- #
# Per-route ingest presets, ported from the proven examples/demo_rebuild.py router.
_AUTO_PRESETS: dict[str, dict[str, Any]] = {
    "outline": {"detail": 0.0016, "min_area": 22.0, "max_dim": 1500},
    "region": {"colors": 20, "detail": 0.0032, "min_area": 44.0, "max_dim": 1300},
    "layers": {"colors": 4, "detail": 0.0012},
    "trace": {},
}


def classify_raster(path: "str | Path", *, max_dim: int = 700) -> dict[str, Any]:
    """Cheap raster classification for mode routing.

    ``kind`` is the proven line-art test (high white fraction + few colours + thin
    ink → ``lineart``, else ``illustration``); the extra metrics (``mid_frac`` for
    bilevel-ness, ``solid_bg`` from the corner pixels, ``n_colors`` from a 4-bit
    quantisation) feed :func:`resolve_auto_mode`'s trace/layers routing.
    """
    import cv2
    import numpy as np

    img = _load_scaled(path, max_dim)
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white = float((g >= 235).mean())
    dark = float((g <= 90).mean())
    n_colors = int(len(np.unique((img >> 4).reshape(-1, 3), axis=0)))
    m = min(3, h - 1, w - 1)
    corners = img[[m, m, h - 1 - m, h - 1 - m], [m, w - 1 - m, m, w - 1 - m]].astype(float)
    solid_bg = bool(np.abs(corners - corners.mean(axis=0)).max() <= 12.0)
    kind = "lineart" if (white >= 0.45 and n_colors < 2200 and dark < 0.22) else "illustration"
    return {
        "kind": kind,
        "white_frac": round(white, 4),
        "dark_frac": round(dark, 4),
        "mid_frac": round(max(0.0, 1.0 - white - dark), 4),
        "n_colors": n_colors,
        "solid_bg": solid_bg,
    }


def resolve_auto_mode(path: "str | Path") -> tuple[str, dict[str, Any]]:
    """``mode='auto'`` router: classify ``path`` and pick a vectorize mode.

    Routing, in order: line art → ``outline`` (editable strokes); heavy bilevel ink
    → ``trace`` when the potrace binary is present (smooth Béziers; never chosen
    without the binary); flat colour art on a solid ground → ``layers``; everything
    else → ``region``. Returns ``(mode, meta)`` where ``meta`` carries the
    classification, the chosen route's parameter presets, and the decision — the
    caller reports it so an agent can override (PALS: the router is a heuristic).
    """
    info = classify_raster(path)
    if info["kind"] == "lineart":
        mode = "outline"
    elif info["mid_frac"] <= 0.05 and info["dark_frac"] >= 0.25 and potrace_path():
        mode = "trace"
    elif info["solid_bg"] and info["n_colors"] <= 1500:
        mode = "layers"
    else:
        mode = "region"
    meta = {
        "resolved_mode": mode,
        "classification": info,
        "presets": dict(_AUTO_PRESETS[mode]),
        "hint": ("auto routing is a heuristic — pass an explicit mode to override; "
                 "presets are the route's proven defaults, explicit args win"),
    }
    return mode, meta


# --------------------------------------------------------------------------- #
#  layered logo tracer  --  solid-bg detect + AA-aware palette + even-odd holes
# --------------------------------------------------------------------------- #
def _hex_rgb(rgb) -> str:
    return "#%02X%02X%02X" % tuple(int(max(0, min(255, round(float(v))))) for v in rgb[:3])


def _solid_fill(img, mask) -> str:
    """The layer's true fill, read from its ERODED interior (ignores AA edge pixels)."""
    import cv2
    import numpy as np

    er = cv2.erode(mask.astype("uint8"), np.ones((5, 5), "uint8"))
    px = np.asarray(img, float)[er > 0] if er.any() else np.asarray(img, float)[mask]
    return _hex_rgb([np.median(px[:, i]) for i in range(3)])


def _trace_mask_d(mask, box, *, up: int, eps: float, min_area: float) -> str:
    """cv2 contour trace of a binary mask inside ``box`` → one SVG ``d`` (even-odd holes)."""
    import cv2

    sub = mask[box[1]:box[3], box[0]:box[2]].astype("uint8") * 255
    if sub.size == 0:
        return ""
    big = cv2.resize(sub, (sub.shape[1] * up, sub.shape[0] * up), interpolation=cv2.INTER_LINEAR)
    _, bw = cv2.threshold(big, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ox, oy = box[0], box[1]
    parts = []
    for c in cnts:
        if cv2.contourArea(c) < (up * up) * min_area:
            continue
        p = cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True).reshape(-1, 2).astype(float)
        p[:, 0] = p[:, 0] / up + ox
        p[:, 1] = p[:, 1] / up + oy
        parts.append("M " + " L ".join("%.2f %.2f" % (x, y) for x, y in p) + " Z")
    return " ".join(parts)


def raster_to_layers(
    path: "str | Path",
    *,
    max_colors: int = 4,
    detail: float = 0.0012,
    min_area: float = 2.0,
    upscale: int = 6,
    bg_diff: float = 40.0,
    exclude_corner: bool = False,
) -> tuple[list[dict[str, Any]], int, int]:
    """Layered tracer for flat / logo art on a SOLID background.

    Detects the background from the corners, clusters the *eroded* foreground into
    its distinct true colours (so anti-aliased edge pixels don't spawn phantom
    colours), assigns every pixel to its nearest colour, and traces each colour layer
    with cv2 (all nesting levels → even-odd holes). Emits one ``path`` object per
    layer (largest first, so fine detail paints on top). Returns ``(objects, w, h)``.

    Honest limit: assumes a solid background — a gradient/photographic ground
    posterises. Straight-segment paths (no Béziers); use ``trace`` (potrace) for
    smooth curves, ``region`` for a quick colour base.
    """
    import cv2
    import numpy as np
    from PIL import Image

    img = Image.open(str(path)).convert("RGB")
    W, H = img.size
    A = np.asarray(img, float)
    g = A.mean(2)
    bg = np.median(np.array([A[5, 5], A[5, W - 6], A[H - 6, 5], A[H - 6, W - 6]]), 0)
    fg = np.abs(g - float(bg.mean())) > bg_diff
    if exclude_corner:
        fg[int(H * 0.72):, :int(W * 0.12)] = False
    if not fg.any():
        return [], W, H
    ys, xs = np.where(fg)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    er = cv2.erode(fg.astype("uint8"), np.ones((5, 5), "uint8"), iterations=2).astype(bool)
    src = er if er.sum() > 500 else fg
    px = np.float32(np.asarray(img)[src])
    if len(px) == 0:
        return [], W, H
    uniq = len(np.unique(px.astype(np.uint8).reshape(-1, 3), axis=0))
    K = int(min(8, max(1, uniq)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, cen = cv2.kmeans(px, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(lab.flatten(), minlength=K)
    kept = []
    for i in np.argsort(-counts):
        if counts[i] < 30:
            continue
        if any(np.linalg.norm(cen[i] - k) < 40.0 for k in kept):
            continue
        kept.append(cen[i])
        if len(kept) >= max_colors:
            break
    if not kept:
        return [], W, H

    C = np.array([bg] + list(kept), float)
    full = ((A.reshape(-1, 1, 3) - C.reshape(1, -1, 3)) ** 2).sum(-1).argmin(1).reshape(H, W)
    masks = [full == (i + 1) for i in range(len(kept))]
    order = sorted(range(len(masks)), key=lambda i: -int(masks[i].sum()))

    objs: list[dict[str, Any]] = []
    for k in order:
        d = _trace_mask_d(masks[k], box, up=upscale, eps=detail, min_area=min_area)
        if not d:
            continue
        objs.append({"type": "path", "d": d, "fill": _solid_fill(img, masks[k]),
                     "style": {"fill_rule": "evenodd"}})
    return objs, W, H


# --------------------------------------------------------------------------- #
#  Gradient paint extraction (Gap-1 closure): fit per-shape fills from source  #
# --------------------------------------------------------------------------- #

_TRANSFORM_RE = None  # compiled lazily in _object_transform


def _object_transform(obj) -> "tuple[float, float, float, float] | None":
    """Compose a chain of ``translate(..)``/``scale(..)`` transform ops.

    Returns the composed ``(tx, ty, sx, sy)`` such that a local point maps to
    ``(sx·x + tx, sy·y + ty)``. Handles any left-to-right chain of translate +
    scale (a box-fitted supersampled trace carries FOUR ops:
    ``translate scale translate scale`` — svg_import's fit composed onto
    potrace's y-flip). ``None`` marks an op this sampler does not model
    (rotate/matrix/skew) — the caller must skip fitting that object rather
    than sample the wrong pixels.
    """
    import re
    global _TRANSFORM_RE
    if _TRANSFORM_RE is None:
        _TRANSFORM_RE = re.compile(
            r"([a-zA-Z]+)\(([^)]*)\)")
    transform = (obj.get("style") or {}).get("transform")
    if not transform:
        return (0.0, 0.0, 1.0, 1.0)
    s = str(transform).strip()
    ops = _TRANSFORM_RE.findall(s)
    if not ops or "".join(f"{n}({a})" for n, a in ops).replace(" ", "") != s.replace(" ", ""):
        return None                                   # junk between/around ops
    tx, ty, sx, sy = 0.0, 0.0, 1.0, 1.0              # outer accumulated affine
    for name, args in ops:                            # document order: outermost first
        try:
            vals = [float(v) for v in re.split(r"[\s,]+", args.strip()) if v]
        except ValueError:
            return None
        if name == "translate" and 1 <= len(vals) <= 2:
            ax, ay = vals[0], (vals[1] if len(vals) == 2 else 0.0)
            tx, ty = sx * ax + tx, sy * ay + ty
        elif name == "scale" and 1 <= len(vals) <= 2:
            ax, ay = vals[0], (vals[1] if len(vals) == 2 else vals[0])
            sx, sy = sx * ax, sy * ay
        else:
            return None                               # rotate/matrix/skew: unmodelled
    return (tx, ty, sx, sy)


def _shape_mask(obj, size) -> "tuple[Any, bool] | None":
    """Rasterise one polygon/path object into a winding-aware PIL mask.

    Returns ``(mask_image, y_flipped)`` or ``None`` when the object carries no
    fittable geometry. Subpaths are painted largest-first; a subpath whose
    signed area opposes the dominant one erases (a hole) — exact for the
    opposite-orientation holes both in-tree tracers emit.
    """
    from PIL import Image, ImageDraw

    from ..domain.gradient_fit import flatten_path_d, shoelace

    if obj.get("type") == "polygon" and obj.get("points"):
        subs = [[(float(x), float(y)) for x, y in obj["points"]]]
        flipped = False
    elif obj.get("type") == "path" and obj.get("d"):
        tf = _object_transform(obj)
        if tf is None:
            return None
        tx, ty, sx, sy = tf
        subs = [[(sx * x + tx, sy * y + ty) for x, y in sub]
                for sub in flatten_path_d(obj["d"])]
        flipped = sy < 0
    else:
        return None

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    order = sorted(range(len(subs)), key=lambda k: -abs(shoelace(subs[k])))
    if not order:
        return None
    base_sign = 1.0 if shoelace(subs[order[0]]) >= 0 else -1.0
    for k in order:
        sub = subs[k]
        if len(sub) < 3:
            continue
        sign = 1.0 if shoelace(sub) >= 0 else -1.0
        draw.polygon(sub, fill=255 if sign == base_sign else 0)
    return mask, flipped


def _paint_to_local(fill: "dict[str, Any]", obj: "dict[str, Any]") -> "dict[str, Any]":
    """Convert a user-geometry fill fitted in IMAGE space into the object's
    LOCAL coordinate space (the space its `d`/`points` numbers live in — where
    the renderer resolves userSpaceOnUse gradients).

    Inverse of the ``translate(tx,ty) scale(sx,sy)`` chain: local = (img − t)/s.
    In-tree tracers only produce uniform-magnitude scales (|sx| == |sy|: potrace
    0.1/−0.1, aspect-preserving box fits, identity polygons), so a circular px
    radius stays circular; the mean |s| is used defensively should that ever
    drift.
    """
    tf = _object_transform(obj)
    if tf is None:
        return fill                       # unreachable: _shape_mask skips these
    tx, ty, sx, sy = tf
    if not sx or not sy:
        return fill

    def loc(p):
        return [round((float(p[0]) - tx) / sx, 2), round((float(p[1]) - ty) / sy, 2)]

    if fill.get("kind") == "linear" and fill.get("line") is not None:
        fill["line"] = [loc(fill["line"][0]), loc(fill["line"][1])]
    elif fill.get("kind") == "radial" and fill.get("radius") is not None:
        fill["at"] = loc(fill["at"])
        mag = (abs(sx) + abs(sy)) / 2.0
        fill["radius"] = round(float(fill["radius"]) / mag, 2)
    return fill


# The distance transform moved to the domain with G1 (spine fitting needs it
# too); this alias keeps every existing infrastructure caller and test working.
from ..domain.spine_fit import chamfer_distance as _chamfer_distance  # noqa: E402


_MIN_BAND_DEPTH = 4.0        # interiors shallower than this have no room for rims


def _band_overlay(obj: "dict[str, Any]", paint, width_img: float) -> "dict[str, Any] | None":
    """One rim-band overlay: the SAME geometry, stroke-only, self-clipped.

    An inner stroke of width ``2·t`` covers the pixels within ``t`` of the
    boundary — the contour-following band — with zero new geometry. The clip
    (in the object's LOCAL coordinates, riding inside its transform) keeps the
    stroke's outer half from painting outside the shape.
    """
    tf = _object_transform(obj)
    if tf is None:
        return None
    mag = (abs(tf[2]) + abs(tf[3])) / 2.0 or 1.0
    overlay: dict[str, Any] = {"type": obj.get("type")}
    if obj.get("decorative"):
        overlay["decorative"] = True
    if obj.get("type") == "polygon":
        overlay["points"] = obj["points"]
        clip = {"shape": "polygon", "args": {"points": obj["points"]}}
    elif obj.get("type") == "path":
        overlay["d"] = obj["d"]
        clip = {"shape": "path", "args": {"d": obj["d"]}}
    else:
        return None
    style = dict(obj.get("style") or {})
    style["clip_path"] = clip
    overlay["style"] = style
    overlay["stroke"] = paint
    overlay["stroke_style"] = {"stroke_width": round(width_img / mag, 2)}
    return overlay


def apply_gradient_fills(
    objects: "list[dict[str, Any]]",
    image,
    *,
    min_pixels: int | None = None,
    erode_px: int = 2,
    geometry: str = "user",
    bands: int = 1,
) -> dict[str, Any]:
    """Replace flat fills with gradients fitted from ``image`` (in place).

    ``image`` is a PIL RGB image in the SAME coordinate space as the objects'
    geometry (the vectorize page space). Every polygon/path object is
    re-painted from the source: a linear/radial gradient when the fit beats
    flat (``frameforge_vision.domain.gradient_fit.fit_paint``), else the
    shape's sampled mean colour. Returns the paint summary
    ``{"fill_mode", "fitted", "flat", "skipped"}``.

    ``geometry="user"`` (default) emits the EXACT A1 form — linear ``line`` /
    radial px ``at``+``radius`` — converted into each object's local space via
    :func:`_paint_to_local`, so the ramp lands back on the sampled pixels
    precisely. ``"bbox"`` keeps the legacy angle/fraction approximation.

    ``bands`` ≥ 2 (A2 shape-conforming shading) decomposes each deep-enough
    shape by distance-to-boundary: thresholds at interior-distance quantiles,
    the CORE band re-fitting the object's own fill and each RIM band emitted
    as a self-clipped inner-stroke overlay of the same geometry carrying that
    band's fitted paint (see :func:`_band_overlay`). ``bands=1`` is the
    previous behaviour, byte-identical.
    """
    import numpy as np
    from PIL import ImageFilter

    from ..domain.gradient_fit import DEFAULT_MIN_PIXELS, fit_paint

    n_bands = max(1, int(bands))
    floor = DEFAULT_MIN_PIXELS if min_pixels is None else int(min_pixels)
    src = np.asarray(image.convert("RGB"), dtype=np.float64)
    fitted = flat = skipped = banded = 0
    insertions: list[tuple[int, list[dict[str, Any]]]] = []

    def _fit(sel_ys, sel_xs, flipped, obj):
        pts = np.stack([sel_xs, sel_ys], axis=1).astype(np.float64)
        bbox = (float(sel_xs.min()), float(sel_ys.min()),
                float(sel_xs.max()), float(sel_ys.max()))
        out = fit_paint(pts, src[sel_ys, sel_xs], bbox=bbox, y_flipped=flipped,
                        min_pixels=floor, geometry=geometry)
        fill_val = out["fill"]
        if geometry == "user" and isinstance(fill_val, dict):
            fill_val = _paint_to_local(fill_val, obj)
        return out["family"], fill_val

    for index, obj in enumerate(objects):
        if "fill" not in obj:
            skipped += 1
            continue
        shaped = _shape_mask(obj, image.size)
        if shaped is None:
            skipped += 1
            continue
        mask, flipped = shaped
        eroded = mask
        for _ in range(max(0, int(erode_px))):
            eroded = eroded.filter(ImageFilter.MinFilter(3))
        m = np.asarray(eroded)
        if int((m > 0).sum()) < max(3, floor // 2):
            m = np.asarray(mask)  # slivers: keep the un-eroded mask
        ys, xs = np.nonzero(m > 0)
        if len(ys) == 0:
            skipped += 1
            continue

        overlays: list[dict[str, Any]] = []
        if n_bands >= 2 and len(ys) >= floor:
            dist = _chamfer_distance(np.asarray(mask))
            interior = dist[dist > 0]
            if interior.size and float(interior.max()) >= _MIN_BAND_DEPTH:
                qs = [k / n_bands for k in range(1, n_bands)]
                ts = [float(t) for t in np.quantile(interior, qs)]
                ts = sorted({t for t in ts if t >= 1.0})
                core_sel = (dist >= ts[-1]) & (m > 0) if ts else None
                if ts and int(core_sel.sum()) >= 12:
                    cys, cxs = np.nonzero(core_sel)
                    family, obj["fill"] = _fit(cys, cxs, flipped, obj)
                    lo = 0.0
                    ring_paints: list[tuple[float, Any]] = []
                    for t in ts:
                        sel = (dist > lo) & (dist <= t) & (m > 0)
                        rys, rxs = np.nonzero(sel)
                        if len(rys) >= 12:
                            _, ring_fill = _fit(rys, rxs, flipped, obj)
                            ring_paints.append((t, ring_fill))
                        lo = t
                    # widest (innermost ring) paints first; the narrower outer
                    # rims overpaint it toward the boundary
                    for t, paint in sorted(ring_paints, key=lambda p: -p[0]):
                        overlay = _band_overlay(obj, paint, 2.0 * t)
                        if overlay is not None:
                            overlays.append(overlay)
                    if overlays:
                        banded += 1
                        fitted += 1
                        insertions.append((index, overlays))
                        continue

        family, obj["fill"] = _fit(ys, xs, flipped, obj)
        if family == "flat":
            flat += 1
        else:
            fitted += 1

    for index, overlays in sorted(insertions, key=lambda p: -p[0]):
        objects[index + 1:index + 1] = overlays

    summary: dict[str, Any] = {
        "fill_mode": "gradient" if n_bands == 1 else "shading",
        "fitted": fitted, "flat": flat, "skipped": skipped,
    }
    if n_bands > 1:
        summary["bands"] = n_bands
        summary["banded"] = banded
    return summary
