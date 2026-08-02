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

import importlib.util
import io
import os
from typing import Any, Optional, Union

DEFAULT_MODEL = os.environ.get("FG_VLM_MODEL", "HuggingFaceTB/SmolVLM-256M-Instruct")
_CACHE: dict[str, Any] = {}


#: Every module the lane needs at RUNTIME, not just to import this file.
#:
#: ``torchvision`` is the non-obvious one and the reason this list exists.
#: transformers 5.x split image processors into `pil` and `torchvision`
#: backends; for Idefics3 — SmolVLM, the default model — BOTH generated classes
#: declare ``_backends = ["torchvision"]``, so a torch-only environment loads
#: the model and then fails inside ``AutoProcessor.from_pretrained``. Probing it
#: here is what keeps :func:`available` an honest gate.
BACKEND_MODULES: tuple[str, ...] = ("torch", "transformers", "PIL", "torchvision")


def install_hint() -> str:
    """One string every caller can show when the lane is unusable."""
    return (
        "install the optional `vlm` extra: "
        "`pip install 'frameforge-vision[vlm]'` (or `uv sync --extra vlm`) — "
        "it carries torch, transformers, Pillow and torchvision (the image-processor "
        "backend transformers 5.x requires). CPU is fine; the default "
        f"{DEFAULT_MODEL} is ~0.5GB and downloads on first use."
    )


def missing_backends() -> tuple[str, ...]:
    """The :data:`BACKEND_MODULES` that cannot be imported, in declaration order.

    Resolved with :func:`importlib.util.find_spec`, so asking the question never
    pays torch's import cost. A ``find_spec`` that raises (a half-installed or
    namespace-shadowed package) counts as missing: unusable is absent.
    """
    missing: list[str] = []
    for module in BACKEND_MODULES:
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError, AttributeError):
            present = False
        if not present:
            missing.append(module)
    return tuple(missing)


def available() -> bool:
    """True if the optional ``vlm`` extra is installed AND usable.

    "Usable" is the point: this gate exists so callers can return an install
    hint instead of a traceback, which it cannot do if it answers yes to an
    environment where the processor will not load.
    """
    return not missing_backends()


def _to_pil(image: Union[str, bytes, "os.PathLike[str]", Any]):
    from PIL import Image
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(image))).convert("RGB")
    return Image.open(os.fspath(image)).convert("RGB")


def _load_processor_and_model(model_id: str):
    """Load (and cache) the processor + model for ``model_id``."""
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


def _load(model_id: str):
    """:func:`_load_processor_and_model`, with backend gaps translated.

    :data:`BACKEND_MODULES` cannot foresee every split — another model family,
    a future transformers release — so the load is wrapped as well as gated. A
    third-party ``ValueError`` about a missing dependency is re-raised as a
    ``RuntimeError`` carrying both the original text and the install command,
    because the caller's contract is "advisory answer or actionable error",
    never "a traceback from inside somebody else's `from_pretrained`".
    """
    try:
        return _load_processor_and_model(model_id)
    except (ValueError, ImportError, OSError) as exc:
        raise RuntimeError(
            f"the local VLM could not load {model_id!r}: {exc} — {install_hint()}"
        ) from exc


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
    missing = missing_backends()
    if missing:
        raise RuntimeError(
            f"the local VLM lane is missing {', '.join(missing)} — {install_hint()}"
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


__all__ = [
    "BACKEND_MODULES",
    "DEFAULT_MODEL",
    "available",
    "describe_image",
    "install_hint",
    "missing_backends",
]
