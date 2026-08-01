"""Value objects for the vision (image/document → FrameForge) bounded context.

These are intentionally backend-agnostic: nothing here imports OpenCV, numpy,
pytesseract, PyMuPDF, or any model SDK. Detectors (infrastructure) produce
:class:`Observation` values; the proposer lowers them into a draft document.

⚠ ARCHITECTURAL CONTRACT (PALS's LAW) — CV/VLM OUTPUT IS UNVERIFIED BY DEFAULT.
Every :class:`Observation` is a *proposal*: a statistical guess from a classical
CV heuristic or a vision model. It is mapped to a FrameForge object and then
re-validated and re-rendered by the forward pipeline before any caller may trust
it. The proposer never commits; it proposes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

Bbox = tuple[float, float, float, float]            # x, y, w, h (page space, +y down)
PointSeq = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RasterImage:
    """A loaded raster image.

    ``pixels`` is an opaque backend array (an RGB ``numpy.ndarray`` when a pixel
    backend loaded it) used by pixel-level detectors; the domain never inspects
    it. ``encoded`` keeps the normalised image bytes so detectors that talk to a
    remote model (the VLM lane) can forward the image without a pixel backend.
    """

    width: int
    height: int
    pixels: Any = None
    encoded: bytes | None = None
    media_type: str = "image/png"


@dataclass(frozen=True)
class Observation:
    """A single backend-agnostic detection. Maps to at most one object.

    Conventions used by :class:`frameforge_vision.application.mapper`:

    - ``color`` is the *fill* paint (closed shapes / regions).
    - ``stroke_color`` / ``stroke_width`` are the outline paint + geometry.
    - ``bbox`` drives ``rect`` / ``ellipse`` / ``text``; ``points`` drive
      ``line`` / ``polyline`` / ``path`` (``meta['d']`` overrides with raw SVG).
    """

    kind: str  # "rect" | "ellipse" | "line" | "polyline" | "path" | "text" | "fill"
    bbox: Bbox | None = None
    points: PointSeq = ()
    color: str | None = None
    stroke_color: str | None = None
    stroke_width: float | None = None
    text: str | None = None
    confidence: float = 1.0
    detector: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkippedDetector:
    """A detector that could not run, and why (graceful degradation, not error)."""

    name: str
    reason: str


@dataclass(frozen=True)
class Proposal:
    """The result of a propose run: a draft document plus full provenance."""

    document: Mapping[str, Any]
    observations: Sequence[Observation]
    detectors_run: Sequence[str]
    detectors_skipped: Sequence[SkippedDetector]
