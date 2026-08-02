"""``available()`` must mean *usable*, not *some of the imports resolve*.

The defect: `vlm.available()` returned True whenever torch, transformers and
Pillow imported. Under transformers 5.x that is not sufficient — image
processors were split into `pil` and `torchvision` backends, and for Idefics3
(SmolVLM, the default model) BOTH classes declare `_backends = ['torchvision']`.
So a torch-only environment passed the gate, and `describe_image` then died
deep inside `AutoProcessor.from_pretrained` with

    ValueError: Could not load any image processor class for
    HuggingFaceTB/SmolVLM-256M-Instruct ... Missing optional dependencies:
    torchvision.

A gate that answers "yes" and then raises somewhere else is worse than no gate:
every caller of `available()` — the MCP's `describe_render` among them — exists
precisely to return an install hint instead of a traceback.
"""
from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 has no stdlib tomllib
    import tomli as tomllib  # type: ignore[no-redefine]

from frameforge_vision import vlm

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = tomllib.loads(
    (ROOT / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["optional-dependencies"]


def _resolve_all_but(monkeypatch, absent: set[str]) -> None:
    """Answer `find_spec` from a fixture, not from this checkout's venv.

    The vision test environment does not install the `vlm` extra, so probing the
    real interpreter would make every assertion here depend on which optional
    packages happen to be present.
    """
    sentinel = object()
    monkeypatch.setattr(
        vlm.importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name in absent else sentinel,
    )


def test_available_reports_false_when_the_image_processor_backend_is_absent(monkeypatch):
    """The reported failure: torch + transformers + PIL present, torchvision not."""
    _resolve_all_but(monkeypatch, {"torchvision"})

    assert vlm.missing_backends() == ("torchvision",)
    assert vlm.available() is False


def test_missing_backends_is_empty_when_every_backend_resolves(monkeypatch):
    _resolve_all_but(monkeypatch, set())

    assert vlm.missing_backends() == ()
    assert vlm.available() is True


def test_torchvision_is_probed_because_transformers_5_requires_it():
    """Pins the specific module the gap was about, so a future edit cannot drop
    it back to the torch/transformers/PIL triple that under-reported."""
    assert "torchvision" in vlm.BACKEND_MODULES


def test_missing_backends_names_every_absent_module_not_just_the_first(monkeypatch):
    monkeypatch.setattr(vlm.importlib.util, "find_spec", lambda name, *a, **k: None)

    missing = vlm.missing_backends()
    assert set(missing) == {"torch", "transformers", "PIL", "torchvision"}


def test_a_half_installed_backend_counts_as_missing(monkeypatch):
    """`find_spec` raises for a broken/namespace-shadowed package."""

    def explode(name, *args, **kwargs):
        raise ValueError(f"{name}.__spec__ is None")

    monkeypatch.setattr(vlm.importlib.util, "find_spec", explode)
    assert vlm.available() is False


def test_describe_image_refuses_with_an_actionable_hint_when_a_backend_is_missing(monkeypatch):
    monkeypatch.setattr(vlm, "missing_backends", lambda: ("torchvision",))

    with pytest.raises(RuntimeError) as excinfo:
        vlm.describe_image(b"not even an image", "what is this?")

    message = str(excinfo.value)
    assert "torchvision" in message, "the message must name what is missing"
    assert "vlm" in message and "extra" in message
    assert "--group" not in message, "this distribution ships extras, not groups"


def test_a_processor_load_failure_becomes_a_runtime_error_naming_the_fix(monkeypatch):
    """Defence in depth: a backend gap the probe cannot foresee (a different
    model, a future transformers split) must still not surface as a raw
    third-party ValueError from inside `from_pretrained`."""
    pytest.importorskip("transformers")

    def explode(model_id):
        raise ValueError(
            "Could not load any image processor class for X. "
            "Missing optional dependencies: torchvision."
        )

    monkeypatch.setattr(vlm, "_load_processor_and_model", explode)
    monkeypatch.setattr(vlm, "missing_backends", lambda: ())
    monkeypatch.setattr(vlm, "_to_pil", lambda image: image)

    with pytest.raises(RuntimeError) as excinfo:
        vlm.describe_image(b"bytes", "what is this?")

    message = str(excinfo.value)
    assert "torchvision" in message
    assert "install" in message.lower()


def test_the_install_hint_is_one_string_every_caller_can_show():
    hint = vlm.install_hint()

    assert "torchvision" in hint
    assert "frameforge-vision[vlm]" in hint
    assert "--group" not in hint


# --------------------------------------------------------------------------- #
#  The declaration must install what the lane probes                           #
# --------------------------------------------------------------------------- #


def _declared(extra: str) -> set[str]:
    """Distribution names declared by ``extra``, lowercased and de-specified."""
    return {
        requirement.split("[")[0].split(";")[0]
        .split(">")[0].split("<")[0].split("=")[0].split("!")[0].split("~")[0]
        .strip().lower()
        for requirement in OPTIONAL[extra]
    }


def test_the_vlm_extra_declares_every_backend_the_lane_probes():
    """A probe with no installer is a lane that can never become available.

    `BACKEND_MODULES` is what the code needs; this extra is the only way a user
    gets it. torchvision was probed by nothing and declared by nothing, so
    `pip install 'frameforge-vision[vlm]'` produced an environment where the
    model loads and its processor does not.
    """
    # import name -> the distribution that provides it
    providers = {
        "torch": "torch",
        "transformers": "transformers",
        "PIL": "pillow",
        "torchvision": "torchvision",
    }
    declared = _declared("vlm")
    missing = [
        f"{module} (provided by {providers[module]})"
        for module in vlm.BACKEND_MODULES
        if providers[module] not in declared
    ]
    assert not missing, (
        "the `vlm` extra does not install: " + ", ".join(missing) +
        f" — declared: {sorted(declared)}"
    )


def test_the_install_hint_promises_only_what_the_extra_delivers():
    """The hint names torchvision; the extra must actually carry it, or the
    instruction we hand the user does not fix their environment."""
    assert "torchvision" in vlm.install_hint()
    assert "torchvision" in _declared("vlm")
