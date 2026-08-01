"""Infrastructure adapters for the vision context.

Every adapter lazily imports its heavy backend (OpenCV, pytesseract, PyMuPDF) so
``import frameforge_vision`` stays cheap and dependency-free; the backend is only
touched inside ``available()`` / ``detect()`` / ``render_page()``.
"""
from __future__ import annotations

from .image_source import DefaultImageSource
from .ocr_detector import TextDetector
from .opencv_detectors import ColorRegionDetector, LineDetector, ShapeDetector
from .pdf_source import PdfDocumentSource
from .regions import DetectedRegion, detect_regions
from .vlm_detector import HttpVlmClient, VlmDetector

__all__ = [
    "DefaultImageSource",
    "ColorRegionDetector",
    "ShapeDetector",
    "LineDetector",
    "TextDetector",
    "VlmDetector",
    "HttpVlmClient",
    "PdfDocumentSource",
    "DetectedRegion",
    "detect_regions",
]
