"""Local vision-language describer — an ADVISORY 'what does this render look like?'.

A small, CPU-runnable VLM (default ``HuggingFaceTB/SmolVLM-256M-Instruct``) that
looks at a rendered page and answers in words: what it depicts, whether it reads,
whether it matches an intent. It closes the coach loop — the render/silhouette
tools already produce the pixels; this turns them back into language the calling
model can act on.

⚠ ARCHITECTURAL CONTRACT (PALS's LAW) — VLM OUTPUT IS UNVERIFIED BY DEFAULT.
A description is a *statistical opinion about pixels*, not a measurement and not
ground truth. It hallucinates, miscounts, and misnames. Any caller that treats
this as a verdict is introducing an architectural omission. Use it to steer, then
verify with the deterministic tools (compare_images NCC/RMSE, score_reconstruction,
the validator) — never as the check itself.

Heavy deps (torch / transformers) are optional and imported lazily, so importing
this module costs nothing until :func:`describe_image` runs. ``available()`` lets
callers degrade gracefully (return an install hint) when the ``vlm`` group is absent.

Boundary: stdlib + optional external ML libs only (no ``tooling``).
"""
from __future__ import annotations

import io
import os
from typing import Any, Optional, Union

DEFAULT_MODEL = os.environ.get("FG_VLM_MODEL", "HuggingFaceTB/SmolVLM-256M-Instruct")
_CACHE: dict[str, Any] = {}


def available() -> bool:
    """True if the optional ``vlm`` group (torch + transformers + PIL) is importable."""
    try:
        import PIL.Image  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def _to_pil(image: Union[str, bytes, "os.PathLike[str]", Any]):
    from PIL import Image
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(image))).convert("RGB")
    return Image.open(os.fspath(image)).convert("RGB")


def _load(model_id: str):
    if model_id in _CACHE:
        return _CACHE[model_id]
    import torch
    from transformers import AutoProcessor
    try:                                          # transformers v5
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except ImportError:                           # transformers v4
        from transformers import AutoModelForVision2Seq as _AutoVLM
    proc = AutoProcessor.from_pretrained(model_id)
    mdl = _AutoVLM.from_pretrained(model_id, dtype=torch.float32)
    mdl.eval()
    _CACHE[model_id] = (proc, mdl)
    return proc, mdl


def describe_image(
    image: Union[str, bytes, "os.PathLike[str]", Any],
    prompt: str = "Describe this image in one or two sentences.",
    *,
    model: Optional[str] = None,
    max_new_tokens: int = 96,
) -> str:
    """Return the VLM's free-text answer to ``prompt`` about ``image`` (CPU).

    ``image`` is a path, raw bytes, or a PIL image. Raises ``RuntimeError`` with an
    install hint if the ``vlm`` group is not available. The result is advisory
    (see the module contract) — never a measurement.
    """
    if not available():
        raise RuntimeError(
            "the local VLM needs the 'vlm' group: "
            "`uv pip install torch transformers pillow accelerate` "
            "(CPU is fine; a small model like SmolVLM-256M is the default)."
        )
    import torch
    proc, mdl = _load(model or DEFAULT_MODEL)
    pil = _to_pil(image)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=text, images=[pil], return_tensors="pt")
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = proc.batch_decode(out, skip_special_tokens=True)[0]
    return decoded.split("Assistant:")[-1].strip()


__all__ = ["available", "describe_image", "DEFAULT_MODEL"]
