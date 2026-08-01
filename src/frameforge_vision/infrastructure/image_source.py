"""Image loading adapter (Pillow + optional numpy)."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from ..domain.observation import RasterImage


class DefaultImageSource:
    """Load an image reference into a :class:`RasterImage`.

    Uses Pillow for dimensions and a normalised PNG byte payload (consumed by the
    VLM lane), and numpy for an RGB pixel array (consumed by pixel detectors).
    Pillow is required; numpy is optional — without it ``pixels`` is ``None`` and
    only the VLM lane can run.
    """

    def load(self, ref: "str | bytes", *, is_base64: bool = False) -> RasterImage:
        raw = self._raw_bytes(ref, is_base64=is_base64)
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - Pillow is in the vision group
            raise RuntimeError(
                "image loading requires Pillow; install the `vision` dependency group"
            ) from exc

        with Image.open(BytesIO(raw)) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size
            buffer = BytesIO()
            rgb.save(buffer, format="PNG")
            pixels = self._pixels(rgb)

        return RasterImage(
            width=int(width),
            height=int(height),
            pixels=pixels,
            encoded=buffer.getvalue(),
            media_type="image/png",
        )

    @staticmethod
    def _raw_bytes(ref: "str | bytes", *, is_base64: bool) -> bytes:
        if isinstance(ref, (bytes, bytearray)):
            return bytes(ref)
        if not isinstance(ref, str) or not ref:
            raise ValueError("image ref must be a non-empty path string or raw bytes")
        if is_base64:
            return base64.b64decode(ref)
        return Path(ref).expanduser().read_bytes()

    @staticmethod
    def _pixels(rgb):
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - exercised only without numpy
            return None
        return np.asarray(rgb, dtype="uint8")
