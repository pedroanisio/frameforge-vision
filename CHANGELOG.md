# frameforge-vision — CHANGELOG

*The package version is this wheel's own release line. It is not the FrameForge
document-format revision (`frameforge_api.HEAD_VERSION`, 2.8.x) — this package
reads pixels, and its releases follow its detectors and API.*

## 1.1.0 — the vlm extra installs a lane that runs (2026-08-02)

### Fixed — the `vlm` extra installed a lane that could not run

`pip install 'frameforge-vision[vlm]'` gave you torch, transformers, Pillow,
accelerate and num2words — and a lane that raised on first use:

```
ValueError: Could not load any image processor class for
HuggingFaceTB/SmolVLM-256M-Instruct. The model configuration resolves to the
following image processor classes: pil: Idefics3ImageProcessorPil, torchvision:
Idefics3ImageProcessor. None of these classes could be imported. Missing
optional dependencies: torchvision.
```

transformers 5.x split image processors into `pil` and `torchvision` backends.
For Idefics3 — SmolVLM-256M, this lane's default model — **both** generated
classes declare `_backends = ["torchvision"]`, so the `pil` name is no escape
hatch. Nothing in this package imports torchvision, which is exactly why it was
never declared; the model does, at load time. `torchvision>=0.17` is now part of
the extra.

`available()` was wrong in the same way, and worse, because callers trust it:
it returned True whenever torch, transformers and Pillow imported, which is a
gate that says yes and then lets the caller crash. It now answers *usable*.

- **`vlm.BACKEND_MODULES`** — the modules the lane needs at runtime, probed with
  `importlib.util.find_spec` so asking never pays torch's import cost.
- **`vlm.missing_backends()`** — exactly which of them are absent, in order, so
  a caller can name the fix instead of guessing. A `find_spec` that raises (a
  half-installed or namespace-shadowed package) counts as missing.
- **`vlm.install_hint()`** — one string every consumer can show. The MCP
  server's `describe_render` returns it as the `hint` of an `ok: false`
  envelope, where it used to let a transformers `ValueError` escape.
- **`describe_image`** refuses with that hint when a backend is missing, and
  translates any load failure the probe could not foresee (another model family,
  a future split) into a `RuntimeError` naming the fix — never a raw
  third-party exception from inside `from_pretrained`.

Upgrading: re-install the extra (`uv sync --extra vlm`, or
`pip install --upgrade 'frameforge-vision[vlm]'`). Nothing else changed —
`available()` and `describe_image` keep their signatures.

### Dependency pins are bounded, consistent, and gated

A pin audit across the eight family repositories found defects that a local run
can never surface, because siblings resolve by path in development and by
version only once published. `tests/test_dependency_pins.py` (new) now reads
`pyproject.toml` and `uv.lock` directly and fails on:

- a family requirement with no floor or **no cap** — a future major accepted
  without review;
- a cap that is not the next major (`<2` for a 1.x floor, `<1` for a 0.x one);
- an **extra restating a lower floor** than the base requirement for the same
  package — harmless while both are present, since the resolver intersects
  them, but a real defect the moment the base pin is edited;
- a **lockfile recording a version the sibling checkout no longer has**, which
  `uv sync` would faithfully reproduce.

- `frameforge>=2.8` in the `raster` extra was **uncapped**; now `>=2.8,<3`.
- **The lockfile was three contract revisions stale**: it recorded
  `frameforge-api` 1.0.0 (`HEAD_VERSION 2.8.2`) and `frameforge-render` 1.0.0
  against checkouts at 1.3.1 (`HEAD_VERSION 2.11.0`) and 1.9.0. Refreshed.

The gate reads `pyproject.toml` through the family's guarded `tomllib` /
`tomli` idiom, so it runs on the Python 3.10 floor the package claims; `tomli`
is declared as a dev dependency, marked to be a no-op on 3.11+.

## 1.0.0 — extracted from the frameforge monorepo (2026-08-01)

First standalone release. The code is unchanged apart from import paths and two
deferred imports; this is a move, not a rewrite.

- **`frameforge_vision`** — the vision context, moved from
  `frameforge/src/frameforge/vision`: 36 modules, ~8,900 LOC across
  `domain` (measurement maths), `application` (proposer composition) and
  `infrastructure` (OpenCV/PIL detectors, vectorise, compare, measure).
- **The domain layer no longer imports the authoring SDK.**
  `domain/services/proposer.py` imported `frameforge.sdk.author` at module
  scope, which made a 2,000-line numpy-only measurement core drag in a
  24,000-line SDK. Deferred to the one function that builds a document; the same
  was done for `application/mapper.py`.
- **No hard dependency in either direction.** Every import of FrameForge from
  this package is lazy, and every import of this package from FrameForge was
  already lazy. So the two distributions are optional extras of each other
  (`frameforge-vision[author]`, `frameforge[vision]`) and there is no cycle.
  Verified: `import frameforge_vision` and the whole measurement surface load
  with `frameforge` absent from `sys.modules`.
- **numpy is the only unconditional dependency.** OpenCV, Pillow, pytesseract,
  torch and the authoring SDK are extras (`cv`, `ocr`, `vlm`, `author`), all
  imported lazily, so an absent extra fails at the call site with a clear
  message rather than on import.
- **Seam tests are marked, not deleted.** The three tests that round-trip an
  observation into a rendered FrameForge document are the ones that would catch
  this package drifting from the SDK it emits into. They skip on a bare install
  and run under `make seam`.
