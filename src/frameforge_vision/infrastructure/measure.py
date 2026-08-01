"""Automatic measurement + annotation layer for raster references.

A rasterized image is only a reliable reference for *vector reconstruction* if the
model reading it knows where things are. A bare PNG gives pixels; the model has to
guess coordinates. This adapter overlays a measurement layer — a grid, edge rulers
labelled in source coordinates, an explicit coordinate system, region boxes with
stable IDs, and landmark crosshairs — and emits the exact numbers as structured
spatial metadata a caller can extract coordinates from.

Two invariants make it a *reliable* reference:

1. **Coordinate identity.** The full overlay keeps the source's pixel dimensions,
   so a pixel at ``(px, py)`` in the source is at ``(px, py)`` in the overlay — the
   rulers read true.
2. **Zoom-awareness.** A crop is rendered *enlarged* (so detail is legible), but
   its rulers stay labelled in the ORIGINAL image's coordinates and the crop's
   ``origin`` + ``scale`` transform is reported. Any coordinate read off a zoomed
   crop maps straight back to source space via
   ``source_px = crop.origin_px + read_px / crop.scale``.

⚠ ARCHITECTURAL CONTRACT (PALS's LAW). The returned ``spatial`` numbers for the
coordinate system, grid, rulers, explicit regions, and *structural* landmarks
(corners / edge midpoints / centre) are exact deterministic geometry — trust them.
*Detected* landmarks and regions from the optional CV detectors are UNVERIFIED
guesses: they are hints for anchoring, not ground truth. The overlay image is a
drawing aid; the JSON is the source of truth.

Pillow is imported lazily (via the ``image_compare`` helpers) so
``import frameforge_vision`` stays dependency-free; the backend is only touched
when a measurement is actually built.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .image_compare import Region, _font, _pil, load_rgb

# The coordinate authority is the pure domain module; this infra layer keeps only the
# PIL rendering and re-exports the frame maths so existing callers/tests keep working.
from ..domain.coordinates import (  # noqa: F401  (re-exported for back-compat)
    CoordinateSystem,
    CropTransform,
    Landmark,
    MeasuredRegion,
    crop_transform,
    denorm_box,
    denorm_point,
    measured_regions,
    nice_step,
    normalize_point,
    point_frames,
    resolve_point_spec,
    structural_landmarks,
)

# ─────────────────────────────────────────────────────────────
# palette (translucent so the layer reads over dark and light grounds alike)
# ─────────────────────────────────────────────────────────────
_GRID_MINOR = (64, 132, 220, 46)
_GRID_MAJOR = (64, 132, 220, 96)
_RULER_TICK = (18, 20, 26, 210)
_RULER_LABEL_BG = (18, 20, 26, 190)
_RULER_LABEL_FG = (244, 244, 240)
_STRUCT_LM = (214, 52, 168)      # magenta — exact structural anchors
_DETECT_LM = (24, 176, 196)      # cyan — unverified detected anchors
_CROP_BOUND = (240, 150, 24)     # amber — zoom crop bounds
_REGION_PALETTE = (
    (36, 148, 78), (198, 92, 24), (150, 72, 196), (24, 130, 200),
    (196, 44, 92), (120, 128, 20), (0, 150, 150), (176, 96, 0),
)


# The CoordinateSystem, CropTransform, MeasuredRegion and Landmark value objects, and
# the frame maths (measured_regions / structural_landmarks / crop_transform /
# resolve_point_spec / point_frames / nice_step / _clamp01), now live in
# frameforge_vision.domain.coordinates and are imported above. Only the PIL-bearing
# Measurement container and the drawing/orchestration stay in this infra module.
@dataclass
class Measurement:
    """The overlay image, any zoom crops, and the exact spatial metadata."""

    overlay: Any                       # PIL.Image.Image (source-sized)
    spatial: dict[str, Any]
    crops: list[tuple[str, Any]] = field(default_factory=list)   # (name, PIL image)


# ─────────────────────────────────────────────────────────────
# geometry — pure frame maths moved to frameforge_vision.domain.coordinates
# (nice_step / measured_regions / structural_landmarks / crop_transform); only the
# CV-backed detected_landmarks (needs OpenCV/PIL) stays in this infra module.
# ─────────────────────────────────────────────────────────────
def detected_landmarks(image_bytes: bytes, cs: CoordinateSystem, *,
                       max_landmarks: int = 16) -> list[Landmark]:
    """Optional CV-detected anchors: flat-colour region centroids + shape corners.

    UNVERIFIED (PALS's LAW). Returns ``[]`` when numpy/OpenCV are absent or nothing
    salient is found, so the layer degrades to the exact structural anchors alone.
    """
    try:
        from ..domain.observation import RasterImage
        from .opencv_detectors import ColorRegionDetector, ShapeDetector
        import numpy as np
    except ImportError:
        return []
    try:
        from PIL import Image
        from io import BytesIO
        rgb = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = RasterImage(width=rgb.width, height=rgb.height,
                          pixels=np.asarray(rgb, dtype="uint8"))
    except Exception:  # pragma: no cover - decode already succeeded upstream
        return []

    out: list[Landmark] = []
    n = 0
    for det in (ColorRegionDetector(), ShapeDetector()):
        if not det.available():
            continue
        try:
            observations = det.detect(img)
        except Exception:  # pragma: no cover - detector-internal failure is non-fatal
            continue
        for obs in observations:
            if n >= max_landmarks:
                break
            if not obs.bbox:
                continue
            bx, by, bw, bh = obs.bbox
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            n += 1
            out.append(Landmark(
                id=f"L{n:02d}",
                kind=f"{obs.detector}-centroid",
                x_px=cx, y_px=cy, cs=cs.to_cs(cx, cy),
                source=obs.detector, confidence=float(obs.confidence),
            ))
    return out


# ─────────────────────────────────────────────────────────────
# drawing
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _Viewport:
    """Maps source pixels onto a render surface (identity for the full overlay)."""

    origin_x: float
    origin_y: float
    scale: float

    def to_render(self, px: float, py: float) -> tuple[float, float]:
        return ((px - self.origin_x) * self.scale, (py - self.origin_y) * self.scale)


def _rounded_label(draw, x: float, y: float, text: str, *, font, anchor: str = "la"):
    """Draw ``text`` with a legible translucent chip behind it."""
    try:
        bbox = draw.textbbox((x, y), text, font=font, anchor=anchor)
    except TypeError:  # pragma: no cover - very old Pillow without anchor
        bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                   fill=_RULER_LABEL_BG)
    draw.text((x, y), text, font=font, fill=_RULER_LABEL_FG, anchor=anchor)


def _draw_grid(draw, cs: CoordinateSystem, vp: _Viewport, render_size: tuple[int, int],
               step: int, label_every: int):
    """Draw minor/major vertical + horizontal grid lines across the render surface."""
    rw, rh = render_size
    k = 0
    x = 0
    while x <= cs.width:
        rx, _ = vp.to_render(x, 0)
        if 0 <= rx <= rw:
            major = (k % max(1, label_every)) == 0
            draw.line([(rx, 0), (rx, rh)], fill=_GRID_MAJOR if major else _GRID_MINOR,
                      width=2 if major else 1)
        x += step
        k += 1
    k = 0
    y = 0
    while y <= cs.height:
        _, ry = vp.to_render(0, y)
        if 0 <= ry <= rh:
            major = (k % max(1, label_every)) == 0
            draw.line([(0, ry), (rw, ry)], fill=_GRID_MAJOR if major else _GRID_MINOR,
                      width=2 if major else 1)
        y += step
        k += 1


def _draw_rulers(draw, cs: CoordinateSystem, vp: _Viewport, render_size: tuple[int, int],
                 step: int, label_every: int, *, tick_font):
    """Draw edge ticks + coordinate labels (top + left edges) in CS units."""
    rw, rh = render_size
    # top edge: x ticks
    k = 0
    x = 0
    while x <= cs.width:
        rx, _ = vp.to_render(x, 0)
        if 0 <= rx <= rw:
            draw.line([(rx, 0), (rx, 12 if (k % max(1, label_every)) == 0 else 7)],
                      fill=_RULER_TICK, width=2)
            if (k % max(1, label_every)) == 0:
                cval, _ = cs.to_cs(x, 0)
                _rounded_label(draw, min(rx + 3, rw - 2), 3, f"{cval:g}",
                               font=tick_font, anchor="la")
        x += step
        k += 1
    # left edge: y ticks
    k = 0
    y = 0
    while y <= cs.height:
        _, ry = vp.to_render(0, y)
        if 0 <= ry <= rh:
            draw.line([(0, ry), (12 if (k % max(1, label_every)) == 0 else 7, ry)],
                      fill=_RULER_TICK, width=2)
            if (k % max(1, label_every)) == 0:
                _, cval = cs.to_cs(0, y)
                _rounded_label(draw, 3, min(ry + 2, rh - 2), f"{cval:g}",
                               font=tick_font, anchor="la")
        y += step
        k += 1


def _draw_regions(draw, regions: Sequence[MeasuredRegion], vp: _Viewport, *, font):
    for i, r in enumerate(regions):
        color = _REGION_PALETTE[i % len(_REGION_PALETTE)]
        x, y, w, h = r.bbox_px
        r0 = vp.to_render(x, y)
        r1 = vp.to_render(x + w, y + h)
        draw.rectangle([r0[0], r0[1], r1[0], r1[1]], outline=color, width=3)
        badge = f"{r.id}"
        try:
            tb = draw.textbbox((r0[0] + 2, r0[1] + 2), badge, font=font, anchor="la")
        except TypeError:  # pragma: no cover
            tb = draw.textbbox((r0[0] + 2, r0[1] + 2), badge, font=font)
        draw.rectangle([tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2], fill=color)
        draw.text((r0[0] + 2, r0[1] + 2), badge, font=font, fill=(255, 255, 255), anchor="la")


def _draw_landmarks(draw, landmarks: Sequence[Landmark], vp: _Viewport,
                    render_size: tuple[int, int], *, font):
    rw, rh = render_size
    for lm in landmarks:
        rx, ry = vp.to_render(lm.x_px, lm.y_px)
        if not (0 <= rx <= rw and 0 <= ry <= rh):
            continue
        color = _STRUCT_LM if lm.kind == "structural" or lm.source == "structural" else _DETECT_LM
        arm = 7
        draw.line([(rx - arm, ry), (rx + arm, ry)], fill=color, width=2)
        draw.line([(rx, ry - arm), (rx, ry + arm)], fill=color, width=2)
        draw.ellipse([rx - 2, ry - 2, rx + 2, ry + 2], outline=color, width=1)
        lx = rx + arm + 2 if rx < rw - 40 else rx - arm - 2
        anchor = "lm" if rx < rw - 40 else "rm"
        _rounded_label(draw, lx, ry, lm.id, font=font, anchor=anchor)


def _draw_crop_bounds(draw, crops: Sequence[CropTransform], vp: _Viewport, *, font):
    for c in crops:
        x, y = c.origin_px
        w, h = c.size_px
        r0 = vp.to_render(x, y)
        r1 = vp.to_render(x + w, y + h)
        draw.rectangle([r0[0], r0[1], r1[0], r1[1]], outline=_CROP_BOUND, width=3)
        _rounded_label(draw, r0[0] + 3, r0[1] + 3, f"ZOOM {c.name}", font=font, anchor="la")


def annotate(render_img, cs: CoordinateSystem, vp: _Viewport, *,
             step: int, label_every: int,
             grid: bool = True, rulers: bool = True,
             regions: Sequence[MeasuredRegion] = (),
             landmarks: Sequence[Landmark] = (),
             crops: Sequence[CropTransform] = ()):
    """Composite the measurement layer onto ``render_img`` (kept at its own size)."""
    Image, _, ImageDraw, _, _ = _pil()
    base = render_img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    render_size = base.size
    tick_font = _font(15)
    label_font = _font(16, bold=True)

    if grid:
        _draw_grid(draw, cs, vp, render_size, step, label_every)
    if crops:
        _draw_crop_bounds(draw, crops, vp, font=label_font)
    if regions:
        _draw_regions(draw, regions, vp, font=label_font)
    if landmarks:
        _draw_landmarks(draw, landmarks, vp, render_size, font=tick_font)
    if rulers:
        _draw_rulers(draw, cs, vp, render_size, step, label_every, tick_font=tick_font)

    return Image.alpha_composite(base, layer).convert("RGB")


# ─────────────────────────────────────────────────────────────
# orchestration
# ─────────────────────────────────────────────────────────────
def build_measurement(image_bytes: bytes, *,
                      regions: Sequence[Region] = (),
                      origin: str = "top-left",
                      grid: bool = True,
                      grid_step: int | None = None,
                      rulers: bool = True,
                      label_every: int = 2,
                      landmarks: bool = True,
                      detect_landmarks: bool = True,
                      zooms: Sequence[Region] = (),
                      zoom_long_edge: int = 1024) -> Measurement:
    """Build the overlay + zoom crops + exact spatial metadata for one image."""
    img = load_rgb(image_bytes)
    W, H = img.size
    cs = CoordinateSystem(origin=origin, width=W, height=H)
    step = int(grid_step) if grid_step and grid_step > 0 else nice_step(max(W, H))
    label_every = max(1, int(label_every))

    region_objs = measured_regions(regions, cs)
    lm_objs: list[Landmark] = []
    if landmarks:
        lm_objs.extend(structural_landmarks(cs))
        if detect_landmarks:
            lm_objs.extend(detected_landmarks(image_bytes, cs))

    crop_xforms = [crop_transform(z.name, z.box, cs, render_long_edge=zoom_long_edge)
                   for z in zooms]

    identity = _Viewport(0.0, 0.0, 1.0)
    overlay = annotate(
        img, cs, identity, step=step, label_every=label_every,
        grid=grid, rulers=rulers, regions=region_objs, landmarks=lm_objs,
        crops=crop_xforms,
    )

    crops_out: list[tuple[str, Any]] = []
    for xform in crop_xforms:
        crops_out.append((xform.name, _render_crop(img, cs, xform, step, label_every,
                                                   grid, rulers, lm_objs)))

    spatial: dict[str, Any] = {
        "image": {"width_px": W, "height_px": H},
        "coordinate_system": cs.describe(),
        "grid": {"step_px": step, "label_every": label_every,
                 "cols": W // step + 1, "rows": H // step + 1} if grid else None,
        "rulers": {"step_px": step, "label_every": label_every,
                   "edges": ["top", "left"], "units": "pixels"} if rulers else None,
        "regions": [r.to_dict() for r in region_objs],
        "landmarks": [lm.to_dict() for lm in lm_objs],
        "crops": [x.to_dict() for x in crop_xforms],
        "reconstruction_hint": (
            "Read coordinates off the rulers (source pixels in the stated coordinate "
            "system). For a zoomed crop, map read pixels back with "
            "source_px = crop.origin_px + read_px / crop.scale. Anchor elements to the "
            "structural landmarks (A1..A9, exact); detected landmarks (L*) are hints."
        ),
    }
    return Measurement(overlay=overlay, spatial=spatial, crops=crops_out)


# ─────────────────────────────────────────────────────────────
# point marking ("aim + click") — coordinates in every frame
# ─────────────────────────────────────────────────────────────
_MARK_FG = (222, 28, 44)          # red crosshair
_MARK_PATH = (222, 28, 44, 150)   # translucent connecting path


# resolve_point_spec + point_frames (the any-frame ⇄ image-px funnels) now live in
# frameforge_vision.domain.coordinates and are imported above.


def _draw_points(draw, points: Sequence[tuple[float, float, str]], vp: _Viewport,
                 *, connect: bool, font):
    """Draw numbered crosshairs at ``points`` (image px), optionally connected."""
    rpts = [(vp.to_render(px, py), label) for (px, py, label) in points]
    if connect and len(rpts) >= 2:
        draw.line([p for (p, _) in rpts], fill=_MARK_PATH, width=3, joint="curve")
    for i, ((rx, ry), label) in enumerate(rpts, start=1):
        arm = 10
        draw.line([(rx - arm, ry), (rx + arm, ry)], fill=_MARK_FG, width=2)
        draw.line([(rx, ry - arm), (rx, ry + arm)], fill=_MARK_FG, width=2)
        draw.ellipse([rx - 4, ry - 4, rx + 4, ry + 4], outline=_MARK_FG, width=2)
        tag = label or f"P{i}"
        _rounded_label(draw, rx + arm + 2, ry - arm - 2, tag, font=font, anchor="la")


def draw_points_overlay(img, cs: CoordinateSystem,
                        points: Sequence[tuple[float, float, str]], *,
                        viewport: CropTransform | None,
                        landmarks: Sequence[Landmark],
                        step: int, label_every: int,
                        grid: bool, rulers: bool, connect: bool):
    """Annotate ``img`` (grid/rulers/landmarks) with numbered point crosshairs, plus —
    when ``viewport`` is set — an enlarged, source-coordinate crop carrying the same
    crosshairs. Returns ``(overlay_rgb, crops)`` where ``crops`` is a list of
    ``(name, image)``.

    This is the single drawing body shared by :func:`build_marks` and
    ``frameforge_vision.infrastructure.workspace.render``; ``points`` are already in
    image pixels, so the callers keep only their differing point-resolution and
    ``spatial`` assembly.
    """
    Image, _, ImageDraw, _, _ = _pil()
    font = _font(16, bold=True)
    identity = _Viewport(0.0, 0.0, 1.0)
    base = annotate(img, cs, identity, step=step, label_every=label_every,
                    grid=grid, rulers=rulers, landmarks=landmarks,
                    crops=(viewport,) if viewport else ())
    over = base.convert("RGBA")
    layer = Image.new("RGBA", over.size, (0, 0, 0, 0))
    _draw_points(ImageDraw.Draw(layer), points, identity, connect=connect, font=font)
    overlay = Image.alpha_composite(over, layer).convert("RGB")

    crops: list[tuple[str, Any]] = []
    if viewport is not None:
        crop_view = _render_crop(img, cs, viewport, step, label_every, grid, rulers,
                                 landmarks).convert("RGBA")
        clayer = Image.new("RGBA", crop_view.size, (0, 0, 0, 0))
        cvp = _Viewport(viewport.origin_px[0], viewport.origin_px[1], viewport.scale)
        _draw_points(ImageDraw.Draw(clayer), points, cvp, connect=connect, font=font)
        crops.append((viewport.name, Image.alpha_composite(crop_view, clayer).convert("RGB")))
    return overlay, crops


def build_marks(image_bytes: bytes, point_specs: Sequence[Any], *,
                viewport_box: Region | None = None,
                origin: str = "top-left",
                grid: bool = True,
                grid_step: int | None = None,
                rulers: bool = True,
                label_every: int = 2,
                connect: bool = False,
                zoom_long_edge: int = 1024) -> Measurement:
    """Resolve + draw marked points, returning the overlay, optional zoomed view, and frames."""
    img = load_rgb(image_bytes)
    W, H = img.size
    cs = CoordinateSystem(origin=origin, width=W, height=H)
    step = int(grid_step) if grid_step and grid_step > 0 else nice_step(max(W, H))
    label_every = max(1, int(label_every))

    lms = structural_landmarks(cs)
    lms_by_id = {lm.id: lm for lm in lms}
    viewport = (crop_transform(viewport_box.name, viewport_box.box, cs,
                               render_long_edge=zoom_long_edge)
                if viewport_box is not None else None)

    resolved: list[tuple[float, float, str]] = []
    frames: list[dict[str, Any]] = []
    for i, spec in enumerate(point_specs):
        px, py = resolve_point_spec(spec, cs, lms_by_id, viewport)
        label = str(spec.get("label")) if isinstance(spec, dict) and spec.get("label") else f"P{i + 1}"
        resolved.append((px, py, label))
        entry = {"index": i + 1, "label": label}
        entry.update(point_frames(px, py, cs, viewport))
        frames.append(entry)

    overlay, crops_out = draw_points_overlay(
        img, cs, resolved, viewport=viewport, landmarks=lms, step=step,
        label_every=label_every, grid=grid, rulers=rulers, connect=connect)

    spatial = {
        "image": {"width_px": W, "height_px": H},
        "coordinate_system": cs.describe(),
        "viewport": viewport.to_dict() if viewport else None,
        "points": frames,
        "reconstruction_hint": (
            "Each point is reported in the full image (px + coordinate system + "
            "normalized) and, when a viewport is set, in viewport pixels. Points are "
            "anchored to the image, so a crosshair stays fixed as the viewport moves. "
            "Feed image_px straight into the SDK draw/geometry API to reconstruct."
        ),
    }
    return Measurement(overlay=overlay, spatial=spatial, crops=crops_out)


def _render_crop(img, cs: CoordinateSystem, xform: CropTransform, step: int,
                 label_every: int, grid: bool, rulers: bool,
                 landmarks: Sequence[Landmark]):
    """Crop ``img`` to the zoom box, enlarge it, and annotate in SOURCE coordinates."""
    Image, *_ = _pil()
    ox, oy = xform.origin_px
    sw, sh = xform.size_px
    # crop_transform snaps origin/size to whole pixels and render_px to an exact
    # integer multiple, so this crop+resize IS the transform — every viewport_px
    # reading inverts exactly (no truncated-fraction bias, no resize scale skew).
    crop = img.crop((int(ox), int(oy), int(ox + sw), int(oy + sh)))
    rw, rh = xform.render_px
    enlarged = crop.resize((rw, rh), Image.LANCZOS)
    vp = _Viewport(ox, oy, xform.scale)
    # Ruler labels come from `cs.to_cs(source_px)`, so a zoomed crop still reads in
    # original coordinates — that is what makes the crop "zoom-aware".
    return annotate(
        enlarged, cs, vp, step=step, label_every=label_every,
        grid=grid, rulers=rulers, regions=(), landmarks=landmarks, crops=(),
    )
