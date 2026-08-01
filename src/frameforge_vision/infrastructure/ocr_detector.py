"""Optional OCR detector (pytesseract → Tesseract).

Words become ``text`` observations boxed to their detected bounds. OCR is
opportunistic: when pytesseract or the Tesseract binary is missing the detector
reports ``available() is False`` and the proposer skips it.
"""
from __future__ import annotations

from typing import Sequence

from ..domain.observation import Observation, RasterImage


class TextDetector:
    name = "text"

    def __init__(self, *, min_confidence: float = 40.0) -> None:
        self._min_confidence = float(min_confidence)

    def availability(self) -> "tuple[bool, str | None]":
        """``(available, reason)`` — the reason names the missing piece, so the MCP
        layer can relay *why* OCR degraded ('no text found' is a different signal
        than a missing dependency; PALS's Law)."""
        try:
            import pytesseract
        except ImportError:
            return False, ("pytesseract is not installed — install the `vision` "
                           "dependency group")
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:
            return False, f"the Tesseract binary is missing or broken: {exc}"
        return True, None

    def available(self) -> bool:
        return self.availability()[0]

    def unavailable_reason(self) -> str:
        reason = self.availability()[1]
        return reason or ("OCR backend unavailable; install the `vision` group's "
                          "pytesseract plus the Tesseract binary")

    def detect(self, image: RasterImage) -> Sequence[Observation]:
        import pytesseract
        from PIL import Image

        if image.encoded is None:
            return []
        from io import BytesIO

        with Image.open(BytesIO(image.encoded)) as pil_image:
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

        observations: list[Observation] = []
        count = len(data.get("text", []))
        for i in range(count):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < self._min_confidence:
                continue
            x, y = float(data["left"][i]), float(data["top"][i])
            w, h = float(data["width"][i]), float(data["height"][i])
            observations.append(
                Observation(
                    kind="text",
                    bbox=(x, y, w, h),
                    text=text,
                    color="#111111",
                    confidence=round(conf / 100.0, 3),
                    detector=self.name,
                )
            )
        return observations
