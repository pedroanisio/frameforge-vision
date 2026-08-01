"""Public-surface contract for the vision region-analysis adapter."""

from __future__ import annotations

import importlib


def test_regions_star_exports_match_the_infrastructure_boundary() -> None:
    regions = importlib.import_module("frameforge_vision.infrastructure.regions")
    infrastructure = importlib.import_module("frameforge_vision.infrastructure")

    assert regions.__all__ == ["DetectedRegion", "detect_regions"]
    assert all(getattr(regions, name) is getattr(infrastructure, name) for name in regions.__all__)

    namespace: dict[str, object] = {}
    exec("from frameforge_vision.infrastructure.regions import *", namespace)
    assert {name for name in namespace if not name.startswith("__")} == set(regions.__all__)


def test_regions_internal_helpers_remain_available_by_explicit_import() -> None:
    regions = importlib.import_module("frameforge_vision.infrastructure.regions")

    for name in ("RegionAnalysis", "detect_closed_regions", "consensus_smooth_regions"):
        assert hasattr(regions, name)
        assert name not in regions.__all__
