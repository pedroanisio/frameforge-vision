"""Font matching — rank resolvable families by shape similarity to a reference.

Reconstruction needs a stand-in typeface chosen on evidence, not priors: given
a reference crop showing type and the text it shows, each candidate family is
rendered (fontconfig resolves the file, PIL rasterizes) and scored against the
reference by normalized cross-correlation on ink-cropped, height-normalized
bitmaps, with an aspect-ratio penalty (condensed vs wide is exactly what NCC
on normalized heights can miss). Heuristic ranking — verify the winner in a
real render (PALS's Law).
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Any, Sequence

__all__ = ["match_font_ranking"]

_SAMPLE_HEIGHT = 64


def _require_deps():
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover — vision extras absent
        raise RuntimeError(f"match_font needs numpy+Pillow: {exc}") from exc
    if shutil.which("fc-match") is None:
        raise RuntimeError("match_font needs fontconfig's fc-match on PATH")
    return np, Image, ImageDraw, ImageFont


def _font_file(family: str) -> tuple[str, bool]:
    """Resolve a family via fontconfig; returns (file, exact-ish match?)."""
    out = subprocess.run(["fc-match", "-f", "%{file}\n%{family}", family],
                         capture_output=True, text=True, timeout=10)
    lines = (out.stdout or "").splitlines() + ["", ""]
    file, resolved = lines[0].strip(), lines[1].strip()
    exact = bool(file) and family.strip().lower() in resolved.lower()
    return file, exact


def _ink_crop(np, arr, threshold=128.0):
    ink = arr < threshold
    if not ink.any():
        return None
    ys, xs = ink.nonzero()
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _render_sample(np, Image, ImageDraw, ImageFont, file: str, text: str):
    font = ImageFont.truetype(file, _SAMPLE_HEIGHT)
    pad = _SAMPLE_HEIGHT
    img = Image.new("L", (int(_SAMPLE_HEIGHT * 0.9 * max(len(text), 1)) + 2 * pad,
                          _SAMPLE_HEIGHT * 3), 255)
    ImageDraw.Draw(img).text((pad, pad), text, font=font, fill=0)
    return _ink_crop(np, np.asarray(img, dtype=float))


def _normalized(np, Image, arr, height=48):
    src = Image.fromarray(arr.astype("uint8"), mode="L")
    width = max(8, round(src.width * height / src.height))
    return np.asarray(src.resize((width, height), Image.LANCZOS), dtype=float)


def _ncc_same_height(np, a, b):
    """NCC of two height-normalized strips, overlapped at the common width."""
    w = min(a.shape[1], b.shape[1])
    pa, pb = a[:, :w] - a[:, :w].mean(), b[:, :w] - b[:, :w].mean()
    na, nb = float(np.sqrt((pa * pa).sum())), float(np.sqrt((pb * pb).sum()))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((pa * pb).sum()) / (na * nb)


def match_font_ranking(reference_bytes: bytes, text: str,
                       candidates: Sequence[str],
                       *, box: Sequence[float] | None = None) -> list[dict[str, Any]]:
    """Rank ``candidates`` by similarity of their rendered ``text`` to the reference.

    ``box`` optionally crops the reference first (normalized [x, y, w, h]).
    Unresolvable families are reported with ``resolved: false`` and no score.
    """
    import io

    np, Image, ImageDraw, ImageFont = _require_deps()
    ref_img = Image.open(io.BytesIO(reference_bytes)).convert("L")
    if box:
        x, y, w, h = box
        ref_img = ref_img.crop((round(x * ref_img.width), round(y * ref_img.height),
                                round((x + w) * ref_img.width),
                                round((y + h) * ref_img.height)))
    ref_arr = _ink_crop(np, np.asarray(ref_img, dtype=float))
    if ref_arr is None:
        raise ValueError("reference contains no ink to match against")
    ref_norm = _normalized(np, Image, ref_arr)
    ref_aspect = ref_arr.shape[1] / ref_arr.shape[0]

    ranking: list[dict[str, Any]] = []
    for family in candidates:
        file, exact = _font_file(family)
        if not file or not exact:
            ranking.append({"family": family, "resolved": False,
                            "note": "fontconfig does not resolve this family here"})
            continue
        sample = _render_sample(np, Image, ImageDraw, ImageFont, file, text)
        if sample is None:
            ranking.append({"family": family, "resolved": False,
                            "note": "rendered sample produced no ink"})
            continue
        cand_norm = _normalized(np, Image, sample)
        ncc = _ncc_same_height(np, ref_norm, cand_norm)
        import math
        aspect = sample.shape[1] / sample.shape[0]
        aspect_delta = abs(math.log(max(aspect, 1e-6) / max(ref_aspect, 1e-6)))
        ranking.append({
            "family": family,
            "resolved": True,
            "file": file,
            "ncc": round(ncc, 4),
            "aspect_delta": round(aspect_delta, 4),
            "score": round(ncc - 0.5 * aspect_delta, 4),
        })
    ranking.sort(key=lambda r: r.get("score", float("-inf")), reverse=True)
    return ranking
