"""Default mapper: one :class:`Observation` → one FrameForge object dict.

Stroke geometry goes through the SDK's :func:`frameforge_sdk.paint.stroke`
helper, which emits the P3-correct inline ``stroke_style`` bundle (paint in
``stroke``, geometry in ``stroke_style``). No bespoke stroke-token registry —
the SDK already owns that contract.
"""
from __future__ import annotations

from typing import Any, Mapping


from ..domain.observation import Observation

_FILLABLE = {"rect", "ellipse", "fill"}
_STROKED = {"line", "polyline", "path"}


class DefaultObservationMapper:
    """Lower observations to objects, dropping anything geometrically degenerate."""

    def to_object(self, observation: Observation, index: int) -> "Mapping[str, Any] | None":
        kind = observation.kind
        oid = f"{kind}_{index}"

        if kind in ("rect", "fill"):
            box = self._box(observation)
            if box is None:
                return None
            obj: dict[str, Any] = {"type": "rect", "id": oid, "box": box}
            self._apply_fill(obj, observation)
            self._apply_stroke(obj, observation)
            return obj

        if kind == "ellipse":
            box = self._box(observation)
            if box is None:
                return None
            x, y, w, h = box
            obj = {"type": "ellipse", "id": oid, "center": [x + w / 2, y + h / 2], "rx": w / 2, "ry": h / 2}
            self._apply_fill(obj, observation)
            self._apply_stroke(obj, observation)
            return obj

        if kind == "line":
            ends = self._endpoints(observation)
            if ends is None:
                return None
            (x0, y0), (x1, y1) = ends
            obj = {"type": "line", "id": oid, "from": [x0, y0], "to": [x1, y1]}
            self._apply_stroke(obj, observation, default_width=1.0)
            return obj

        if kind == "polyline":
            pts = self._clean_points(observation.points)
            if len(pts) < 2:
                return None
            obj = {"type": "polyline", "id": oid, "points": [[px, py] for px, py in pts]}
            self._apply_stroke(obj, observation, default_width=1.0)
            return obj

        if kind == "path":
            d = observation.meta.get("d") or self._points_to_d(self._clean_points(observation.points),
                                                               closed=bool(observation.meta.get("closed")))
            if not d:
                return None
            obj = {"type": "path", "id": oid, "d": d}
            self._apply_fill(obj, observation)
            self._apply_stroke(obj, observation, default_width=1.0)
            return obj

        if kind == "text":
            box = self._box(observation)
            if box is None or not (observation.text and observation.text.strip()):
                return None
            obj = {"type": "text", "id": oid, "text": observation.text, "box": box}
            if observation.color:
                obj["style"] = {"fill": observation.color}
            return obj

        return None

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _box(observation: Observation) -> "list[float] | None":
        if observation.bbox is None:
            return None
        x, y, w, h = (float(v) for v in observation.bbox)
        if not (w > 0 and h > 0) or any(v != v for v in (x, y, w, h)):  # w/h>0 and not NaN
            return None
        return [x, y, w, h]

    @staticmethod
    def _apply_fill(obj: dict[str, Any], observation: Observation) -> None:
        if observation.color and observation.kind in _FILLABLE:
            obj["fill"] = observation.color

    @staticmethod
    def _apply_stroke(obj: dict[str, Any], observation: Observation, *, default_width: "float | None" = None) -> None:
        color = observation.stroke_color
        if color is None and observation.kind in _STROKED:
            color = observation.color
        width = observation.stroke_width if observation.stroke_width is not None else default_width
        if width is None and color is None:
            return
        if width is None:
            obj["stroke"] = color
            return
        # Deferred: emitting a FrameForge document is the OPTIONAL half of this
        # package. Importing the authoring SDK at module scope would make
        # `frameforge-vision` unusable without it, when measuring an image needs
        # no authoring at all.
        from frameforge_sdk.paint import stroke as stroke_fields
        obj.update(stroke_fields(float(width), color=color))

    @staticmethod
    def _endpoints(observation: Observation):
        pts = DefaultObservationMapper._clean_points(observation.points)
        if len(pts) >= 2:
            return pts[0], pts[-1]
        box = DefaultObservationMapper._box(observation)
        if box is not None:
            x, y, w, h = box
            return (x, y), (x + w, y + h)
        return None

    @staticmethod
    def _clean_points(points) -> list[tuple[float, float]]:
        cleaned: list[tuple[float, float]] = []
        for pt in points or ():
            try:
                px, py = float(pt[0]), float(pt[1])
            except (TypeError, ValueError, IndexError):
                continue
            if px == px and py == py:  # drop NaN
                cleaned.append((px, py))
        return cleaned

    @staticmethod
    def _points_to_d(points: list[tuple[float, float]], *, closed: bool = False) -> "str | None":
        if len(points) < 2:
            return None
        segs = [f"M{_fmt(points[0][0])} {_fmt(points[0][1])}"]
        segs += [f"L{_fmt(px)} {_fmt(py)}" for px, py in points[1:]]
        if closed:
            segs.append("Z")
        return " ".join(segs)


def _fmt(value: float) -> str:
    rounded = round(float(value), 2)
    return str(int(rounded)) if rounded == int(rounded) else str(rounded)
