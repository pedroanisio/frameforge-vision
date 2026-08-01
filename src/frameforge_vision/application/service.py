"""Composition root for the vision context.

Wires the default detector fleet (OpenCV/numpy shapes, OCR, the VLM lane) to the
proposer, and exposes the two entry points the MCP tools call. Everything is
injectable so callers/tests can swap in fakes (Dependency Inversion in practice).
"""
from __future__ import annotations

from typing import Iterable, Sequence

from ..domain.observation import Proposal, RasterImage
from ..domain.ports import Detector, DocumentSource, ImageSource
from ..domain.services.proposer import Proposer
from .mapper import DefaultObservationMapper


def default_detectors() -> list[Detector]:
    """The default fleet, in draw order. None is a hard dependency of the import:
    each adapter degrades to ``available() is False`` when its backend is absent.
    """
    from ..infrastructure.opencv_detectors import ColorRegionDetector, LineDetector, ShapeDetector
    from ..infrastructure.ocr_detector import TextDetector
    from ..infrastructure.vlm_detector import HttpVlmClient, VlmDetector

    return [
        ColorRegionDetector(),  # numpy-only: backgrounds + solid blocks → fills
        ShapeDetector(),        # OpenCV: contours → rect / ellipse / path
        LineDetector(),         # OpenCV: Hough segments → line
        TextDetector(),         # OCR (optional): words → text
        VlmDetector(HttpVlmClient()),  # optional VLM lane: semantic layout
    ]


def build_default_proposer(detectors: Sequence[Detector] | None = None) -> Proposer:
    fleet = list(detectors) if detectors is not None else default_detectors()
    return Proposer(fleet, DefaultObservationMapper())


def propose_from_image(
    ref: "str | bytes",
    *,
    is_base64: bool = False,
    title: str = "Proposed from image",
    detector_names: Iterable[str] | None = None,
    image_source: ImageSource | None = None,
    proposer: Proposer | None = None,
) -> Proposal:
    """Load an image and propose a draft FrameForge document from it."""
    if image_source is None:
        from ..infrastructure.image_source import DefaultImageSource

        image_source = DefaultImageSource()
    image = image_source.load(ref, is_base64=is_base64)
    return (proposer or build_default_proposer()).propose(image, title=title, detector_names=detector_names)


def propose_from_document(
    path: str,
    *,
    page: int = 1,
    dpi: int = 144,
    title: str | None = None,
    detector_names: Iterable[str] | None = None,
    document_source: DocumentSource | None = None,
    proposer: Proposer | None = None,
) -> Proposal:
    """Rasterise one document page and propose a draft document from it.

    v1 routes documents through the same CV pipeline as images (page → raster →
    detectors). The repo's deterministic vector extractor for born-digital PDFs
    lives separately in ``tooling/pdf_to_frameforge_yml.py``.
    """
    if document_source is None:
        from ..infrastructure.pdf_source import PdfDocumentSource

        document_source = PdfDocumentSource()
    if not document_source.available():
        raise RuntimeError(document_source.unavailable_reason())
    image: RasterImage = document_source.render_page(path, page, dpi=dpi)
    resolved_title = title or f"Proposed from {path} (page {page})"
    return (proposer or build_default_proposer()).propose(
        image, title=resolved_title, detector_names=detector_names
    )
