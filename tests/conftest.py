"""Shared markers for the vision suite.

This package has two halves and they have different install shapes, so the
suite has to as well: the measurement tests must run on a bare
`pip install frameforge-vision`, while the handful that round-trip an
observation into a rendered FrameForge document need the optional `author`
extra.

Marking them explicitly — rather than letting them fail, or skipping the whole
module — keeps both facts visible: a bare install is genuinely green, and the
cross-package seam is genuinely covered when the extra is present.

    uv run pytest                      # measurement only; seam tests skip
    uv sync --extra author && pytest   # everything, including the seam
"""
from __future__ import annotations

import importlib.util

import pytest

#: True when the FrameForge authoring SDK is importable (the `author` extra).
HAS_AUTHOR = importlib.util.find_spec("frameforge") is not None

#: Decorator for tests that build or render a FrameForge document. These are the
#: seam tests — they are the ones that would catch this package drifting away
#: from the SDK it emits into, so they should be RUN in CI with the extra, not
#: quietly deleted because they were inconvenient after the split.
needs_author = pytest.mark.skipif(
    not HAS_AUTHOR,
    reason="needs the `author` extra (the FrameForge authoring SDK): "
           "uv sync --extra author",
)
