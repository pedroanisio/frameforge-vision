"""Ports (hexagonal boundaries) for the vision context.

All abstractions the proposer depends on live here, in the domain. Concrete
adapters — OpenCV/numpy detectors, OCR, the VLM lane, the PDF page rasteriser —
implement these Protocols in :mod:`frameforge_vision.infrastructure`, so the
domain depends on abstractions only (Dependency Inversion) and new detectors are
added without editing the proposer (Open/Closed).
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .observation import Observation, RasterImage


@runtime_checkable
class Detector(Protocol):
    """Detect primitives in a raster image.

    ``available()`` reports whether the backend can run *here* (import present,
    binary installed, endpoint configured); when False the proposer records the
    detector as skipped with ``unavailable_reason()`` and moves on.
    """

    name: str

    def available(self) -> bool: ...

    def unavailable_reason(self) -> str: ...

    def detect(self, image: RasterImage) -> Sequence[Observation]: ...


@runtime_checkable
class ImageSource(Protocol):
    """Load a reference (path or bytes) into a :class:`RasterImage`."""

    def load(self, ref: "str | bytes", *, is_base64: bool = False) -> RasterImage: ...


@runtime_checkable
class DocumentSource(Protocol):
    """Rasterise one page of a document (e.g. a PDF) into a :class:`RasterImage`."""

    name: str

    def available(self) -> bool: ...

    def unavailable_reason(self) -> str: ...

    def render_page(self, path: str, page: int, *, dpi: int = 144) -> RasterImage: ...


@runtime_checkable
class VlmClient(Protocol):
    """Talk to a vision model and return its raw text response.

    Kept as a port so the VLM lane is provider-agnostic: any open-weights server
    with an OpenAI-compatible vision endpoint (llama.cpp, Ollama, vLLM, …) can be
    wired in behind it without touching the detector.
    """

    def available(self) -> bool: ...

    def unavailable_reason(self) -> str: ...

    def infer(self, image: RasterImage, prompt: str) -> str: ...


@runtime_checkable
class ObservationMapper(Protocol):
    """Lower one :class:`Observation` to one FrameForge object dict (or ``None``)."""

    def to_object(self, observation: Observation, index: int) -> "Mapping[str, Any] | None": ...
