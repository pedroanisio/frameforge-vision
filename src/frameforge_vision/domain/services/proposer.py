"""The proposer: orchestrate detectors and lower observations into a draft doc.

Depends only on the :class:`Detector` and :class:`ObservationMapper` ports. The
FrameForge envelope is assembled with the SDK's own authoring API
(:class:`frameforge.sdk.author.DocumentBuilder`) rather than a hand-rolled dict,
so the document contract lives in one place (SDK reuse, not a fork).
"""
from __future__ import annotations

from typing import Iterable, Sequence


from ..observation import Observation, Proposal, RasterImage, SkippedDetector
from ..ports import Detector, ObservationMapper


class Proposer:
    """Run available detectors over an image and build a draft FrameForge document."""

    def __init__(
        self,
        detectors: Sequence[Detector],
        mapper: ObservationMapper,
        *,
        profile: str = "diagram",
        lang: str = "en",
    ) -> None:
        self._detectors = tuple(detectors)
        self._mapper = mapper
        self._profile = profile
        self._lang = lang

    def propose(
        self,
        image: RasterImage,
        *,
        title: str = "Proposed (vision)",
        detector_names: Iterable[str] | None = None,
    ) -> Proposal:
        observations: list[Observation] = []
        run: list[str] = []
        skipped: list[SkippedDetector] = []
        for detector in self._select(detector_names):
            if detector.available():
                observations.extend(detector.detect(image))
                run.append(detector.name)
            else:
                skipped.append(SkippedDetector(detector.name, detector.unavailable_reason()))
        document = self._assemble(title, image, observations)
        return Proposal(
            document=document,
            observations=tuple(observations),
            detectors_run=tuple(run),
            detectors_skipped=tuple(skipped),
        )

    def _select(self, detector_names: Iterable[str] | None) -> tuple[Detector, ...]:
        if detector_names is None:
            return self._detectors
        wanted = set(detector_names)
        return tuple(detector for detector in self._detectors if detector.name in wanted)

    def _assemble(self, title: str, image: RasterImage, observations: Sequence[Observation]):
        # Deferred, and deliberately so: this is the DOMAIN layer, and a domain
        # importing the authoring SDK at module scope is backwards — it made the
        # pure measurement core (2k LOC, numpy only) drag in a 24k-line SDK.
        # Import it where a document is actually built.
        from frameforge.sdk.author import DocumentBuilder
        builder = DocumentBuilder(title=title, profile=self._profile, lang=self._lang)
        page = builder.page(
            "proposed",
            canvas={"size": [image.width, image.height], "units": "px"},
            coordinate_mode="absolute",
        ).layer("proposed")
        for index, observation in enumerate(observations):
            obj = self._mapper.to_object(observation, index)
            if obj is not None:
                page.add(dict(obj))
        return builder.build_dict()
