# frameforge-vision — CHANGELOG

*The package version is this wheel's own release line. It is not the FrameForge
document-format revision (`frameforge_api.HEAD_VERSION`, 2.8.x) — this package
reads pixels, and its releases follow its detectors and API.*

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
