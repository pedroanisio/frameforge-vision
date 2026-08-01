"""Optional VLM lane: a vision model proposes semantic layout observations.

The lane is just another :class:`Detector`, so it composes with the classical
detectors under the same proposer (Open/Closed). It is provider-agnostic: any
open-weights server exposing an OpenAI-compatible vision chat endpoint
(llama.cpp, Ollama, vLLM, …) works behind :class:`HttpVlmClient`, which uses only
the standard library — no extra Python dependency.

⚠ PALS's LAW: the model's response is untrusted text. It is parsed defensively
into observations, mapped to objects, then re-validated and re-rendered by the
forward pipeline before anything is trusted.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any, Sequence

from ..domain.observation import Observation, RasterImage
from ..domain.ports import VlmClient

_PROMPT = (
    "You are a layout digitiser. Look at the image and return ONLY a JSON object "
    '{"observations": [...]} with no prose. Each observation is one of:\n'
    '{"kind":"rect","bbox":[x,y,w,h],"color":"#RRGGBB"}\n'
    '{"kind":"ellipse","bbox":[x,y,w,h],"color":"#RRGGBB"}\n'
    '{"kind":"line","points":[[x0,y0],[x1,y1]],"stroke_color":"#RRGGBB"}\n'
    '{"kind":"text","bbox":[x,y,w,h],"text":"..."}\n'
    "Coordinates are pixels from the top-left, +y down, matching the image size. "
    "Be conservative; include a confidence in [0,1] when unsure."
)

_ALLOWED_KINDS = {"rect", "ellipse", "fill", "line", "polyline", "path", "text"}


class VlmDetector:
    """Turn a vision model's JSON answer into observations."""

    name = "vlm"

    def __init__(self, client: VlmClient, *, prompt: str = _PROMPT) -> None:
        self._client = client
        self._prompt = prompt

    def available(self) -> bool:
        return self._client.available()

    def unavailable_reason(self) -> str:
        return self._client.unavailable_reason()

    def detect(self, image: RasterImage) -> Sequence[Observation]:
        raw = self._client.infer(image, self._prompt)
        payload = _loads_lenient(raw)
        if payload is None:
            return []
        items = payload.get("observations") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return []
        observations: list[Observation] = []
        for item in items:
            observation = self._coerce(item)
            if observation is not None:
                observations.append(observation)
        return observations

    def _coerce(self, item: Any) -> "Observation | None":
        if not isinstance(item, dict):
            return None
        kind = str(item.get("kind", "")).lower()
        if kind not in _ALLOWED_KINDS:
            return None
        bbox = _coerce_bbox(item.get("bbox"))
        points = _coerce_points(item.get("points"))
        text = item.get("text")
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return Observation(
            kind=kind,
            bbox=bbox,
            points=points,
            color=_coerce_color(item.get("color")),
            stroke_color=_coerce_color(item.get("stroke_color")),
            stroke_width=_coerce_float(item.get("stroke_width")),
            text=str(text) if isinstance(text, str) else None,
            confidence=max(0.0, min(1.0, confidence)),
            detector=self.name,
        )


class HttpVlmClient:
    """OpenAI-compatible vision chat client over the standard library.

    Configure via environment:
      ``FRAMEFORGE_VISION_VLM_URL``   chat/completions endpoint (required)
      ``FRAMEFORGE_VISION_VLM_MODEL`` model id (required)
      ``FRAMEFORGE_VISION_VLM_KEY``   bearer token (optional)
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._url = url if url is not None else os.environ.get("FRAMEFORGE_VISION_VLM_URL")
        self._model = model if model is not None else os.environ.get("FRAMEFORGE_VISION_VLM_MODEL")
        self._api_key = api_key if api_key is not None else os.environ.get("FRAMEFORGE_VISION_VLM_KEY")
        self._timeout = float(timeout)

    def available(self) -> bool:
        return bool(self._url and self._model)

    def unavailable_reason(self) -> str:
        return (
            "VLM lane not configured; set FRAMEFORGE_VISION_VLM_URL and "
            "FRAMEFORGE_VISION_VLM_MODEL (and optionally FRAMEFORGE_VISION_VLM_KEY)"
        )

    def infer(self, image: RasterImage, prompt: str) -> str:
        if not self.available():
            raise RuntimeError(self.unavailable_reason())
        if image.encoded is None:
            raise RuntimeError("VLM lane requires the encoded image bytes")
        data_url = f"data:{image.media_type};base64,{base64.b64encode(image.encoded).decode('ascii')}"
        body = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                "temperature": 0,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise RuntimeError(f"VLM request failed: {exc}") from exc
        return _extract_message(parsed)


def _extract_message(parsed: Any) -> str:
    try:
        return parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def _loads_lenient(raw: str) -> Any:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except ValueError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except ValueError:
            return None
    return None


def _coerce_bbox(value: Any):
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return tuple(float(v) for v in value)
        except (TypeError, ValueError):
            return None
    return None


def _coerce_points(value: Any):
    if not isinstance(value, (list, tuple)):
        return ()
    points = []
    for pt in value:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                points.append((float(pt[0]), float(pt[1])))
            except (TypeError, ValueError):
                continue
    return tuple(points)


def _coerce_color(value: Any):
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_float(value: Any):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
