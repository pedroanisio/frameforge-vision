"""frameforge-vision — measure images, and propose FrameForge documents from them.

The inverse of the renderer: instead of FrameForge → pixels, this reads pixels
and reports what is there. Classical OpenCV/numpy detectors and an optional VLM
lane each implement one :class:`Detector` port; the proposer lowers their
observations into a draft document via the authoring SDK.

⚠ PALS's LAW: every proposal is unverified CV/VLM output. Callers must run it
through the forward validate+render pipeline (the MCP ``propose_*`` tools do this
automatically) before trusting it.

TWO HALVES, AND ONLY ONE OF THEM NEEDS FRAMEFORGE
-------------------------------------------------
*Measuring* an image — coordinates, primitive fitting, region detection, image
comparison, vectorisation — needs numpy and OpenCV and nothing from FrameForge.
*Emitting* a document from those measurements needs the authoring SDK.

So this package imports `frameforge_sdk` and `frameforge_render` **lazily, at
the call sites that build or rasterise a document** — never at module scope. The
measurement half therefore works with `frameforge` absent, and the reverse edge
is lazy too, so neither distribution hard-depends on the other. Install
``frameforge-vision[author]`` when you want the document-emitting half.

THE PUBLIC SURFACE IS TIERED
----------------------------
* **This module** is the stable API: ports, value objects, and the proposer
  composition above. Prefer it.
* **The submodules** (``frameforge_vision.infrastructure.measure``,
  ``.image_compare``, ``.vectorize``, ``.domain.coordinates``, …) are a real but
  *wider* surface. The MCP server reaches into them for the coordinate-workspace
  tools, so they are supported — but they are where breaking changes will land
  first, and narrowing them is tracked work rather than a promise already kept.
"""
from __future__ import annotations

from .application import (
    DefaultObservationMapper,
    build_default_proposer,
    default_detectors,
    propose_from_document,
    propose_from_image,
)
from .domain import (
    Detector,
    DocumentSource,
    ImageSource,
    Observation,
    ObservationMapper,
    Proposal,
    Proposer,
    RasterImage,
    SkippedDetector,
    VlmClient,
)

__all__ = [
    # value objects
    "Observation",
    "RasterImage",
    "Proposal",
    "SkippedDetector",
    # ports
    "Detector",
    "ImageSource",
    "DocumentSource",
    "VlmClient",
    "ObservationMapper",
    # services / composition
    "Proposer",
    "DefaultObservationMapper",
    "default_detectors",
    "build_default_proposer",
    "propose_from_image",
    "propose_from_document",
]
