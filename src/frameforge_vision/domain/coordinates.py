"""The coordinate authority — pure frame maths for the raster→vector tools.

This is the single home for how the measurement/reconstruction tools express and
convert coordinates: the `CoordinateSystem` (origin top-left / bottom-left /
center), the `CropTransform` (a zoomed crop's offset+scale, both directions), the
value objects (`MeasuredRegion`, `Landmark`), and the multi-frame funnels
`resolve_point_spec` (any frame → image px), its session-free subset
`resolve_plain_point` (px/norm/landmark dicts, no CS or viewport), and
`point_frames` (image px → every frame). It is deliberately **PIL-free and
OpenCV-free** so it imports cheaply and
is exhaustively unit-testable; the rendering that draws these frames lives in
`frameforge_vision.infrastructure.measure`, which re-exports this module's names.

⚠ ARCHITECTURAL CONTRACT (PALS's LAW). The numbers this module produces for the
coordinate system, regions, structural landmarks, and crop transforms are exact
deterministic geometry — trust them. *Detected* landmarks (built elsewhere from CV)
are UNVERIFIED guesses; this module never produces those.

Invariants any caller relies on:
  * `to_cs`/`from_cs` are exact inverses for every origin.
  * `CropTransform.to_source_px` and `to_render_px` are exact inverses, and the
    transform describes the raster crop *actually taken* (`crop_transform` snaps
    origin/extent to whole pixels and the zoom to an integer — no fractional-origin
    bias, no rounded-resize scale skew).
  * points live canonically in IMAGE PIXELS (top-left, +y down); frames are only a
    boundary concern (`resolve_point_spec` in, `point_frames` out).
  * coordinates are CONTINUOUS (pixel-centre convention, shared with the SDK/SVG
    doc space and `edgesnap`/`matchscore`): pixel index `i` covers `[i, i + 1)` and
    its centre is `i + 0.5`. Structural landmarks are continuous — corner-br of a
    W×H image is `(W, H)`, never `(W-1, H-1)`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

# ─────────────────────────────────────────────────────────────
# coordinate system
# ─────────────────────────────────────────────────────────────
_ORIGINS = ("top-left", "bottom-left", "center")


@dataclass(frozen=True)
class CoordinateSystem:
    """Maps image pixels (origin top-left, +y down) to a labelled coordinate space.

    ``origin`` picks where (0, 0) sits and which way +y points:

    - ``top-left``    — (0,0) top-left, +x right, +y down (image / FrameForge page space).
    - ``bottom-left`` — (0,0) bottom-left, +x right, +y up (maths / plot convention).
    - ``center``      — (0,0) at the image centre, +x right, +y up.
    """

    origin: str
    width: int
    height: int

    def normalized(self) -> str:
        return self.origin if self.origin in _ORIGINS else "top-left"

    @property
    def y_up(self) -> bool:
        return self.normalized() in ("bottom-left", "center")

    def to_cs(self, px: float, py: float) -> tuple[float, float]:
        """Image-pixel ``(px, py)`` → coordinate-system value ``(cx, cy)``."""
        o = self.normalized()
        if o == "bottom-left":
            return (px, self.height - py)
        if o == "center":
            return (px - self.width / 2.0, self.height / 2.0 - py)
        return (px, py)

    def from_cs(self, cx: float, cy: float) -> tuple[float, float]:
        """Coordinate-system value ``(cx, cy)`` → image-pixel ``(px, py)``."""
        o = self.normalized()
        if o == "bottom-left":
            return (cx, self.height - cy)
        if o == "center":
            return (cx + self.width / 2.0, self.height / 2.0 - cy)
        return (cx, cy)

    def describe(self) -> dict[str, Any]:
        return {
            "origin": self.normalized(),
            "x_axis": "right",
            "y_axis": "up" if self.y_up else "down",
            "units": "pixels",
            "width_px": self.width,
            "height_px": self.height,
            "note": "Coordinates are in source pixels unless a crop transform says otherwise.",
        }


# ─────────────────────────────────────────────────────────────
# spatial value objects
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MeasuredRegion:
    """A region resolved to exact pixel + coordinate-system geometry."""

    id: str
    name: str
    box_norm: tuple[float, float, float, float]
    bbox_px: tuple[float, float, float, float]     # x, y, w, h (image px)
    offset_px: tuple[float, float]                 # top-left = local frame origin
    centroid_px: tuple[float, float]
    centroid_cs: tuple[float, float]
    area_px: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "box_norm": [round(v, 6) for v in self.box_norm],
            "bbox_px": [round(v, 2) for v in self.bbox_px],
            "offset_px": [round(v, 2) for v in self.offset_px],
            "centroid_px": [round(v, 2) for v in self.centroid_px],
            "centroid_cs": [round(v, 2) for v in self.centroid_cs],
            "area_px": round(self.area_px, 1),
        }


@dataclass(frozen=True)
class Landmark:
    """A named anchor point with exact coordinates.

    ``kind == 'structural'`` anchors (corners / edges / centre) are exact geometry;
    detected anchors are UNVERIFIED CV guesses (see the module contract).
    """

    id: str
    kind: str
    x_px: float
    y_px: float
    cs: tuple[float, float]
    source: str = "structural"
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "px": [round(self.x_px, 2), round(self.y_px, 2)],
            "cs": [round(self.cs[0], 2), round(self.cs[1], 2)],
            "source": self.source,
            "confidence": round(float(self.confidence), 3),
        }


@dataclass(frozen=True)
class CropTransform:
    """The offset+scale that maps a zoomed crop's render pixels back to source px."""

    name: str
    box_norm: tuple[float, float, float, float]
    origin_px: tuple[float, float]     # source px of the crop's top-left
    size_px: tuple[float, float]       # source px extent of the crop
    scale: float                       # render px per source px (zoom factor)
    render_px: tuple[int, int]

    def to_source_px(self, rx: float, ry: float) -> tuple[float, float]:
        """Crop render/viewport px → source px."""
        return (self.origin_px[0] + rx / self.scale, self.origin_px[1] + ry / self.scale)

    def to_render_px(self, px: float, py: float) -> tuple[float, float]:
        """Source px → crop render/viewport px (exact inverse of ``to_source_px``)."""
        return ((px - self.origin_px[0]) * self.scale, (py - self.origin_px[1]) * self.scale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "box_norm": [round(v, 6) for v in self.box_norm],
            "origin_px": [round(v, 2) for v in self.origin_px],
            "size_px": [round(v, 2) for v in self.size_px],
            "scale": round(self.scale, 4),
            "render_px": [int(self.render_px[0]), int(self.render_px[1])],
            "inverse": "source_px = origin_px + read_px / scale",
        }


# ─────────────────────────────────────────────────────────────
# geometry
# ─────────────────────────────────────────────────────────────
def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


_NICE_STEPS = (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000)


def nice_step(extent_px: float, *, divisions: int = 12) -> int:
    """A round tick spacing that splits ``extent_px`` into roughly ``divisions``."""
    if extent_px <= 0:
        return 10
    raw = extent_px / max(1, divisions)
    for step in _NICE_STEPS:
        if step >= raw:
            return step
    return _NICE_STEPS[-1]


# ── the single norm ⇄ px authority — every tool routes its conversions here ──
def denorm_point(nx: float, ny: float, width: float, height: float) -> tuple[float, float]:
    """Normalized (0..1) → pixels, **unclamped**.

    For POINT specs and landmark pairs: an out-of-bounds coordinate must stay out of
    bounds (unlike a box, which is trimmed to the frame — see :func:`denorm_box`).
    """
    return (float(nx) * width, float(ny) * height)


def normalize_point(px: float, py: float, width: float, height: float) -> tuple[float, float]:
    """Pixels → normalized (0..1). A zero-size dimension yields ``0.0`` (no divide)."""
    return (px / width if width else 0.0, py / height if height else 0.0)


def denorm_box(x: float, y: float, w: float, h: float,
               width: float, height: float) -> tuple[float, float, float, float]:
    """Normalized box → pixel ``(px, py, pw, ph)``, **clamped** to the unit square.

    For regions/crops: a box overrunning an edge is trimmed before denormalizing, so
    its pixel extent never exceeds the image.
    """
    px = _clamp01(x) * width
    py = _clamp01(y) * height
    pw = max(0.0, _clamp01(x + w) - _clamp01(x)) * width
    ph = max(0.0, _clamp01(y + h) - _clamp01(y)) * height
    return (px, py, pw, ph)


def measured_regions(regions: Sequence[Any], cs: CoordinateSystem) -> list[MeasuredRegion]:
    """Resolve normalized regions to exact pixel + coordinate-system geometry.

    ``regions`` is any sequence of objects exposing ``.name`` and ``.box``
    (``(x, y, w, h)`` normalized) — the infra ``Region`` value object satisfies this,
    kept duck-typed so the domain stays free of the PIL-bearing image layer.
    """
    W, H = cs.width, cs.height
    out: list[MeasuredRegion] = []
    for i, r in enumerate(regions):
        x, y, w, h = r.box
        px, py, pw, ph = denorm_box(x, y, w, h, W, H)
        cx, cy = px + pw / 2.0, py + ph / 2.0
        out.append(MeasuredRegion(
            id=f"R{i + 1}",
            name=r.name,
            box_norm=(x, y, w, h),
            bbox_px=(px, py, pw, ph),
            offset_px=(px, py),
            centroid_px=(cx, cy),
            centroid_cs=cs.to_cs(cx, cy),
            area_px=pw * ph,
        ))
    return out


def structural_landmarks(cs: CoordinateSystem) -> list[Landmark]:
    """The nine exact structural anchors: 4 corners, 4 edge midpoints, centre."""
    W, H = float(cs.width), float(cs.height)
    pts = [
        ("A1", "corner-tl", 0.0, 0.0), ("A2", "corner-tr", W, 0.0),
        ("A3", "corner-bl", 0.0, H), ("A4", "corner-br", W, H),
        ("A5", "edge-top", W / 2, 0.0), ("A6", "edge-bottom", W / 2, H),
        ("A7", "edge-left", 0.0, H / 2), ("A8", "edge-right", W, H / 2),
        ("A9", "center", W / 2, H / 2),
    ]
    return [Landmark(i, k, x, y, cs.to_cs(x, y)) for (i, k, x, y) in pts]


def crop_transform(name: str, box: Sequence[float], cs: CoordinateSystem, *,
                   render_long_edge: int = 1024) -> CropTransform:
    """Compute the offset+scale for a zoomed crop of ``box`` (normalized).

    The transform is snapped to the raster crop that will actually be taken: the
    origin is floored to the pixel the crop starts at, the extent widened to whole
    pixels covering the requested box, and the zoom factor snapped to an integer,
    so ``render_px`` is an EXACT multiple of ``size_px``. Without the snap, a
    fractional origin (e.g. 0.333·3840 = 1278.72) biased every ``to_source_px``
    reading by the truncated fraction (up to ~1 px), and rounding the render size
    of a fractional extent skewed the effective scale — both fatal to a sub-pixel
    reading instrument. ``render_long_edge`` is a target, not a contract: the
    integer zoom lands the long edge as close to it as exactness allows.
    """
    W, H = cs.width, cs.height
    x, y, w, h = box
    ox0, oy0, sw0, sh0 = denorm_box(x, y, w, h, W, H)
    ox = float(min(math.floor(ox0), max(0, W - 1)))
    oy = float(min(math.floor(oy0), max(0, H - 1)))
    sw = max(1.0, float(min(W, math.ceil(ox0 + max(1.0, sw0)))) - ox)
    sh = max(1.0, float(min(H, math.ceil(oy0 + max(1.0, sh0)))) - oy)
    scale = float(max(1, int(round(render_long_edge / max(sw, sh)))))
    return CropTransform(name, (x, y, w, h), (ox, oy), (sw, sh), scale,
                         (int(sw * scale), int(sh * scale)))


# ─────────────────────────────────────────────────────────────
# point marking ("aim + click") — coordinates in every frame
# ─────────────────────────────────────────────────────────────
def resolve_point_spec(spec: Any, cs: CoordinateSystem,
                       landmarks_by_id: dict[str, Landmark],
                       viewport: CropTransform | None) -> tuple[float, float]:
    """Resolve ONE point spec (in whatever frame) to image pixels ``(px, py)``.

    Accepts ``{"px": [x, y]}``, ``{"norm": [nx, ny]}``, ``{"cs": [cx, cy]}``,
    ``{"landmark": id, "dx"?, "dy"?}``, or ``{"viewport_px": [vx, vy]}``.
    """
    if not isinstance(spec, dict):
        raise ValueError("each point must be an object")
    if "px" in spec:
        x, y = spec["px"]
        return float(x), float(y)
    if "norm" in spec:
        nx, ny = spec["norm"]
        return denorm_point(nx, ny, cs.width, cs.height)
    if "cs" in spec:
        cx, cy = spec["cs"]
        return cs.from_cs(float(cx), float(cy))
    if "landmark" in spec:
        lm = landmarks_by_id.get(str(spec["landmark"]))
        if lm is None:
            raise ValueError(f"unknown landmark {spec['landmark']!r}")
        return lm.x_px + float(spec.get("dx", 0.0)), lm.y_px + float(spec.get("dy", 0.0))
    if "viewport_px" in spec:
        if viewport is None:
            raise ValueError("viewport_px given but no viewport is set")
        vx, vy = spec["viewport_px"]
        return viewport.to_source_px(float(vx), float(vy))
    raise ValueError("point needs one of: px, norm, cs, landmark, viewport_px")


def resolve_plain_point(spec: Any, *, width: float | None = None,
                        height: float | None = None,
                        anchors: dict[str, tuple[float, float]] | None = None,
                        ) -> tuple[float, float]:
    """Resolve a self-describing point dict to image pixels — the session-free grammar.

    The subset of :func:`resolve_point_spec` that needs no ``CoordinateSystem`` or
    viewport: ``{"px": [x, y]}``, ``{"norm": [nx, ny]}`` (resolved against
    ``width``/``height``, which must then be given), and — only when an ``anchors``
    map ``{id: (x_px, y_px)}`` is supplied — ``{"landmark": id, "dx"?, "dy"?}``.
    Session-scoped frames (``cs`` / ``viewport_px``) are deliberately NOT accepted;
    every error names exactly the forms the call site accepts.
    """
    forms = "px, norm" + (", landmark" if anchors is not None else "")
    if not isinstance(spec, dict):
        raise ValueError(f"point {spec!r} must be an object with one of: {forms}")
    if "px" in spec:
        x, y = spec["px"]
        return float(x), float(y)
    if "norm" in spec:
        if not (width and height):
            raise ValueError(f"norm point {spec['norm']!r} needs image dims to "
                             "resolve — pass width/height")
        nx, ny = spec["norm"]
        return denorm_point(nx, ny, width, height)
    if anchors is not None and "landmark" in spec:
        key = str(spec["landmark"])
        if key not in anchors:
            raise ValueError(f"unknown pin/landmark {key!r}")
        ax, ay = anchors[key]
        return ax + float(spec.get("dx", 0.0)), ay + float(spec.get("dy", 0.0))
    raise ValueError(f"point {spec!r} needs one of: {forms}")


def point_frames(px: float, py: float, cs: CoordinateSystem,
                 viewport: CropTransform | None) -> dict[str, Any]:
    """Coordinates of image-pixel ``(px, py)`` in every reference frame."""
    frames: dict[str, Any] = {
        "image_px": [round(px, 2), round(py, 2)],
        "image_cs": [round(v, 2) for v in cs.to_cs(px, py)],
        "normalized": [round(v, 6) for v in normalize_point(px, py, cs.width, cs.height)],
    }
    if viewport is not None:
        vx, vy = viewport.to_render_px(px, py)
        rw, rh = viewport.render_px
        frames["viewport"] = {
            "name": viewport.name,
            "viewport_px": [round(vx, 2), round(vy, 2)],
            "viewport_norm": [round(vx / rw, 6) if rw else 0.0,
                              round(vy / rh, 6) if rh else 0.0],
            "inside": bool(0 <= vx <= rw and 0 <= vy <= rh),
        }
    return frames
