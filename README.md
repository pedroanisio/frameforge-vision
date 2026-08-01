# frameforge-vision

**Read pixels and report what is there.** The inverse of a renderer: instead of
FrameForge → pixels, this measures pixels and — optionally — proposes a
FrameForge document from them.

```bash
pip install frameforge-vision            # measurement core (numpy only)
pip install "frameforge-vision[cv]"      # + the OpenCV detector lane
pip install "frameforge-vision[author]"  # + emit FrameForge documents
```

---

## Two halves, and only one of them needs FrameForge

This is the shape of the package, and it is deliberate:

| half | what it does | needs |
|---|---|---|
| **measure** | coordinates, primitive/spine fitting, region detection, image comparison, vectorisation, font matching | numpy (+ OpenCV for the detector lane) |
| **author** | lower those observations into a draft FrameForge document | the `frameforge` authoring SDK |

Measuring an image has nothing to do with document authoring, so the authoring
SDK is imported **lazily, at the call sites that build or rasterise a document**
— never at module scope. The measurement core (2,000 lines, numpy only) works
with `frameforge` absent.

The reverse edge is lazy too: FrameForge imports this package only inside the
functions that need it. **Neither distribution hard-depends on the other**, so
they can be installed, versioned and released independently without a cycle.

---

## What it does

```python
from frameforge_vision import propose_from_image

proposal = propose_from_image("screenshot.png")     # needs [cv] and [author]
```

⚠ **PALS's LAW.** Every proposal is unverified CV/VLM output. Omissions,
hallucinated primitives and silent misses are properties of the model class, not
edge cases. Run a proposal through a forward validate+render pipeline and
*compare against the source pixels* before trusting it. The FrameForge MCP
`propose_*` tools do this round trip automatically.

The measurement half carries no such caveat — it reports what it measured, and
its numbers are reproducible.

---

## The public surface is tiered

- **`frameforge_vision`** is the stable API: the `Detector` / `ImageSource` /
  `ObservationMapper` ports, the `Observation` / `Proposal` / `RasterImage`
  value objects, and the proposer composition. Prefer it.
- **The submodules** — `infrastructure.measure`, `.image_compare`, `.vectorize`,
  `.regions`, `.refine`, `.fontmatch`, `domain.coordinates`, `domain.fitting` —
  are a real but wider surface. The FrameForge MCP server reaches into them for
  its coordinate-workspace tools, so they are supported. They are also where
  breaking changes will land first; narrowing them is tracked work, not a
  promise already kept.

---

## Layers

```
domain/          2,066 LOC   pure measurement maths — numpy only
application/       249 LOC   proposer composition, observation → object mapping
infrastructure/  6,414 LOC   OpenCV/PIL detectors, vectorise, compare, measure
```

`domain` imports nothing from FrameForge. That was not true before extraction —
`domain/services/proposer.py` imported the authoring SDK at module scope, which
made the pure core drag in a 24,000-line dependency. The import is now deferred
to the one function that builds a document.

---

## Optional extras

| extra | pulls | for |
|---|---|---|
| `cv` | opencv-python-headless, pillow | the classical detector lane |
| `ocr` | pytesseract | OCR-assisted text detection |
| `author` | frameforge | emitting FrameForge documents |
| `vlm` | torch, transformers, accelerate | the local VLM lane |

Nothing heavier than numpy is imported at module scope, so an absent extra
produces a clear message at the call site rather than an `ImportError` on
`import frameforge_vision`.

---

## Versioning

The package version (`1.x`) is this wheel's own release line — it tracks the
detectors and the API, **not** the FrameForge document-format revision
(`frameforge_api.HEAD_VERSION`, `2.8.x`). The two move independently and are
deliberately different numbers.

---

## Development

```bash
uv sync --all-groups
uv run pytest
```

## Related

- [`frameforge`](https://github.com/pedroanisio/frameforge) — the engine, SDK and MCP server
- [`frameforge-api`](https://github.com/pedroanisio/frameforge-api) — the document contract
- [`frameforge-fonts`](https://github.com/pedroanisio/frameforge-fonts) — font discovery and shaping

## License

MIT.
