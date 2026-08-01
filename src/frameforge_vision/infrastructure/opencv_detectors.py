"""Classical computer-vision detectors.

- :class:`ColorRegionDetector` needs only numpy: it finds the dominant flat
  colours and proposes them as fills / blocks. It runs wherever numpy is present.
- :class:`ShapeDetector` and :class:`LineDetector` use OpenCV (contours, Hough).
  When OpenCV is absent they report ``available() is False`` and are skipped.

Each is a thin adapter over its backend; the geometry→FrameForge translation is
the mapper's job, keeping detection and mapping independently testable.
"""
from __future__ import annotations

import math
from typing import Sequence

from ..domain.observation import Observation, RasterImage

_GRAY_STROKE = "#333333"


def _hex(rgb) -> str:
    r, g, b = (int(max(0, min(255, round(float(v))))) for v in rgb[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


class _NumpyMixin:
    @staticmethod
    def _numpy():
        import numpy as np  # noqa: F401

        return np

    def available(self) -> bool:
        try:
            self._numpy()
            return True
        except ImportError:
            return False

    def unavailable_reason(self) -> str:
        return "numpy is not installed; install the `vision` dependency group"


class _OpenCvMixin(_NumpyMixin):
    @staticmethod
    def _cv2():
        import cv2  # noqa: F401

        return cv2

    def available(self) -> bool:
        try:
            self._cv2()
            self._numpy()
            return True
        except ImportError:
            return False

    def unavailable_reason(self) -> str:
        return "OpenCV is not installed; install the `vision` dependency group (opencv-python-headless)"

    def _rgb_array(self, image: RasterImage):
        np = self._numpy()
        if image.pixels is None:
            raise RuntimeError("pixel detectors require an image with a numpy pixel array")
        return np.ascontiguousarray(image.pixels)

    def _gray(self, image: RasterImage):
        cv2 = self._cv2()
        return cv2.cvtColor(self._rgb_array(image), cv2.COLOR_RGB2GRAY)


class ColorRegionDetector(_NumpyMixin):
    """Propose flat colour regions as fills (numpy only, no OpenCV needed).

    Quantises colours, then for each colour covering at least ``min_fraction`` of
    the canvas emits its pixel bounding box as a ``fill`` (large/background) or
    ``rect`` (smaller block). Rough but real: every screenshot has a background
    and a few solid panels, so this gives the proposer non-empty output even when
    OpenCV is unavailable.
    """

    name = "color_region"

    def __init__(self, *, quantize: int = 24, min_fraction: float = 0.04, max_regions: int = 8) -> None:
        self._quantize = max(1, int(quantize))
        self._min_fraction = float(min_fraction)
        self._max_regions = int(max_regions)

    def detect(self, image: RasterImage) -> Sequence[Observation]:
        np = self._numpy()
        arr = image.pixels
        if arr is None:
            return []
        arr = np.asarray(arr)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return []
        height, width = arr.shape[:2]
        total = float(height * width)
        if total <= 0:
            return []

        step = self._quantize
        quantized = ((arr[:, :, :3].astype("int32") // step) * step).astype("int32")
        flat = quantized.reshape(-1, 3)
        colors, counts = np.unique(flat, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1][: self._max_regions]

        observations: list[Observation] = []
        for rank, idx in enumerate(order):
            fraction = float(counts[idx]) / total
            if fraction < self._min_fraction:
                break
            color = colors[idx]
            mask = np.all(quantized == color, axis=2)
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox = (float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1))
            observations.append(
                Observation(
                    kind="fill" if (rank == 0 or fraction >= 0.4) else "rect",
                    bbox=bbox,
                    color=_hex(color),
                    confidence=round(min(0.95, fraction), 3),
                    detector=self.name,
                    meta={"coverage": round(fraction, 3)},
                )
            )
        return observations


class ShapeDetector(_OpenCvMixin):
    """Propose rectangles, ellipses, and closed paths from edge contours (OpenCV)."""

    name = "shape"

    def __init__(self, *, min_area: float = 120.0, max_shapes: int = 80) -> None:
        self._min_area = float(min_area)
        self._max_shapes = int(max_shapes)

    def detect(self, image: RasterImage) -> Sequence[Observation]:
        cv2 = self._cv2()
        gray = self._gray(image)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, None, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        observations: list[Observation] = []
        for contour in contours:
            if len(observations) >= self._max_shapes:
                break
            area = float(cv2.contourArea(contour))
            if area < self._min_area:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            x, y, w, h = (float(v) for v in cv2.boundingRect(contour))
            rect_fill = (area / (w * h)) if (w > 0 and h > 0) else 0.0
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)

            if len(approx) == 4 and rect_fill > 0.8:
                observations.append(
                    Observation("rect", bbox=(x, y, w, h), stroke_color=_GRAY_STROKE,
                                stroke_width=1.0, confidence=0.6, detector=self.name)
                )
            elif circularity > 0.7:
                observations.append(
                    Observation("ellipse", bbox=(x, y, w, h), stroke_color=_GRAY_STROKE,
                                stroke_width=1.0, confidence=0.55, detector=self.name)
                )
            else:
                pts = tuple((float(p[0][0]), float(p[0][1])) for p in approx)
                if len(pts) >= 2:
                    observations.append(
                        Observation("path", points=pts, stroke_color=_GRAY_STROKE, stroke_width=1.0,
                                    confidence=0.4, detector=self.name, meta={"closed": True})
                    )
        return observations


class LineDetector(_OpenCvMixin):
    """Propose straight line segments via the probabilistic Hough transform (OpenCV)."""

    name = "line"

    def __init__(self, *, threshold: int = 80, min_length: float = 40.0, max_gap: float = 8.0,
                 max_lines: int = 60) -> None:
        self._threshold = int(threshold)
        self._min_length = float(min_length)
        self._max_gap = float(max_gap)
        self._max_lines = int(max_lines)

    def detect(self, image: RasterImage) -> Sequence[Observation]:
        cv2 = self._cv2()
        gray = self._gray(image)
        edges = cv2.Canny(gray, 60, 180)
        segments = cv2.HoughLinesP(
            edges, 1, math.pi / 180, self._threshold,
            minLineLength=self._min_length, maxLineGap=self._max_gap,
        )
        if segments is None:
            return []
        observations: list[Observation] = []
        for segment in segments[: self._max_lines]:
            x0, y0, x1, y1 = (float(v) for v in segment[0])
            observations.append(
                Observation("line", points=((x0, y0), (x1, y1)), stroke_color=_GRAY_STROKE,
                            stroke_width=1.0, confidence=0.5, detector=self.name)
            )
        return observations
