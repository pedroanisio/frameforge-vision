"""Application layer for the vision context: mapper + composition service."""
from __future__ import annotations

from .mapper import DefaultObservationMapper
from .service import (
    build_default_proposer,
    default_detectors,
    propose_from_document,
    propose_from_image,
)

__all__ = [
    "DefaultObservationMapper",
    "default_detectors",
    "build_default_proposer",
    "propose_from_image",
    "propose_from_document",
]
