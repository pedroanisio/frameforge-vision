"""Landmark-driven overlay alignment + coordinate-offset extraction.

Given a base (source) image and an overlay image plus matched landmark pairs, this
fits the scale+translation that best maps overlay pixels onto base pixels, reports
the per-pair coordinate offset and the post-fit residual, and composes an aligned
overlay so the fit is visible. It is the alignment complement to ``image_compare``
(which crops matching regions) and ``measure`` (which measures one image): here the
question is *how far apart are these two images at each landmark, and what single
transform brings them together?*

Scope (honest): the fit is a **similarity without rotation** — uniform scale + 2D
translation. It is exact for images that differ only in position and size (the
common raster-reference case) and a least-squares best-fit otherwise; rotation and
shear are NOT modelled, so a rotated overlay will show up as large residuals rather
than being silently "corrected". One pair yields a pure translation.

Pillow is imported lazily (via ``image_compare`` helpers) so ``import
frameforge_vision`` stays dependency-free.
"""
from __future__ import annotations

from typing import Any, Sequence

from .image_compare import _font, _pil, load_rgb

# The fit maths (Similarity + fit_similarity + landmark_offsets + rms_residual) is pure
# and lives in the domain layer; this infra module keeps the PIL compositing and
# re-exports those names so existing callers/tests keep working.
from ..domain.coordinates import denorm_point, resolve_plain_point
from ..domain.fitting import (  # noqa: F401  (re-exported for back-compat)
    Similarity,
    fit_similarity,
    landmark_offsets,
    rms_residual,
)

_BASE_LM = (36, 168, 84)      # green — base landmarks
_OVER_LM = (24, 176, 220)     # cyan — mapped overlay landmarks


def _to_pairs(landmarks: Sequence[dict[str, Any]], base_size, overlay_size):
    """Normalize the landmark-pair payload into pixel ((base),(overlay)) tuples.

    Each side is a legacy ``[x, y]`` list — source pixels, or 0..1 fractions when
    the pair-level ``"norm"`` flag is set — or a self-describing point dict
    (``{"px": [x, y]}`` / ``{"norm": [nx, ny]}``) resolved against THAT side's
    image size; dict entries ignore the pair-level flag. Session-scoped specs
    (``cs`` / ``landmark`` / ``viewport_px``) have no meaning here and raise.
    """
    bw, bh = base_size
    ow, oh = overlay_size

    def _side(pt: Any, w: float, h: float, norm_flag: bool, i: int) -> tuple[float, float]:
        if isinstance(pt, dict):
            try:
                return resolve_plain_point(pt, width=w, height=h)
            except ValueError as exc:
                raise ValueError(f"landmark {i}: {exc}") from None
        if len(pt) != 2:
            raise ValueError(f"landmark {i} points must be [x, y]")
        if norm_flag:
            return denorm_point(pt[0], pt[1], w, h)
        return (float(pt[0]), float(pt[1]))

    pairs = []
    for i, lm in enumerate(landmarks or []):
        if not isinstance(lm, dict) or "base" not in lm or "overlay" not in lm:
            raise ValueError(f"landmark {i} needs both 'base' and 'overlay' points")
        norm_flag = bool(lm.get("norm"))
        pairs.append((_side(lm["base"], bw, bh, norm_flag, i),
                      _side(lm["overlay"], ow, oh, norm_flag, i)))
    if not pairs:
        raise ValueError("need at least one landmark pair")
    return pairs


def build_overlay(base_bytes: bytes, overlay_bytes: bytes, *,
                  landmarks: Sequence[dict[str, Any]],
                  opacity: float = 0.5,
                  rotation: bool = False):
    """Align overlay→base by landmarks; return (composite_image, spatial_dict).

    ``rotation=True`` opts into the full-similarity model (2D Procrustes) and a
    rotated composite; the default stays the rotation-free contract.
    """
    Image, _, ImageDraw, _, _ = _pil()
    base = load_rgb(base_bytes)
    overlay = load_rgb(overlay_bytes)
    pairs = _to_pairs(landmarks, base.size, overlay.size)
    if rotation:
        from frameforge_vision.domain.fitting import fit_similarity_rotation
        transform = fit_similarity_rotation(pairs)
    else:
        transform = fit_similarity(pairs)
    offsets = landmark_offsets(pairs, transform)

    canvas = base.convert("RGBA")
    if rotation:
        # PIL's AFFINE takes the INVERSE map (output→input): o = (1/s)·R⁻¹·(b − t)
        import math as _math
        th = _math.radians(transform.rotation_deg)
        c, sn = _math.cos(th), _math.sin(th)
        s = max(1e-3, transform.scale)
        coeffs = (c / s, sn / s, (-c * transform.tx - sn * transform.ty) / s,
                  -sn / s, c / s, (sn * transform.tx - c * transform.ty) / s)
        layer = overlay.convert("RGBA").transform(
            canvas.size, Image.AFFINE, coeffs, resample=Image.BICUBIC,
            fillcolor=(0, 0, 0, 0))
    else:
        # place the scaled overlay so overlay(0,0) lands at the transform's translation
        s = max(1e-3, transform.scale)
        scaled = overlay.resize((max(1, round(overlay.width * s)),
                                 max(1, round(overlay.height * s))), Image.LANCZOS).convert("RGBA")
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer.paste(scaled, (int(round(transform.tx)), int(round(transform.ty))))
    op = 0.0 if opacity < 0 else 1.0 if opacity > 1 else float(opacity)
    if op < 1.0:
        alpha = layer.getchannel("A").point(lambda v: int(v * op))
        layer.putalpha(alpha)
    comp = Image.alpha_composite(canvas, layer)

    draw = ImageDraw.Draw(comp)
    font = _font(15, bold=True)
    for i, ((bx, by), (ox, oy)) in enumerate(pairs, start=1):
        mx, my = transform.apply(ox, oy)
        _cross(draw, bx, by, _BASE_LM, f"b{i}", font)
        _cross(draw, mx, my, _OVER_LM, f"o{i}", font)
        draw.line([(mx, my), (bx, by)], fill=_OVER_LM + (180,), width=1)
    comp = comp.convert("RGB")

    spatial = {
        "base": {"width_px": base.width, "height_px": base.height},
        "overlay": {"width_px": overlay.width, "height_px": overlay.height},
        "alignment": transform.to_dict(),
        "rms_residual_px": rms_residual(offsets),
        "landmark_offsets": offsets,
        "reconstruction_hint": (
            "offset_px is the raw base−overlay gap at each landmark; residual_px is "
            "what remains after the best-fit scale+translation. Large residuals with "
            "few pairs usually mean rotation/shear (not modelled) — add pairs or "
            "correct orientation first."
        ),
    }
    return comp, spatial


def _cross(draw, x: float, y: float, color, label: str, font):
    arm = 8
    draw.line([(x - arm, y), (x + arm, y)], fill=color, width=2)
    draw.line([(x, y - arm), (x, y + arm)], fill=color, width=2)
    draw.text((x + arm + 2, y - arm - 2), label, font=font, fill=color)
