"""A stateful, session-persisted coordinate workspace — the AI's precise "mouse".

``measure`` and ``mark_points`` are stateless: each call is a pure function of its
inputs. Reconstructing a raster into vectors, though, is *iterative* — pin a few
anchors, look, nudge one 0.01 to the left, pin more, look again, refine over several
passes. That needs memory. This module holds that memory: a workspace bound to one
image, persisted as ``workspace.json`` in the MCP session dir, carrying a set of
named **pins** (anchor points, stored canonically in image pixels) and the current
**viewport** (a crop + zoom). Pins survive across tool calls, so they can be reused,
nudged, pinned in groups, and adjusted together or independently until the
reconstruction is pixel-accurate.

Everything geometric is delegated to :mod:`frameforge_vision.infrastructure.measure`
(coordinate systems, rulers/grid, zoom-aware crops, multi-frame point resolution), so
a pin reads out in every frame — full image, coordinate system, and viewport — exactly
like a measured point. Pins are anchored to the image, so a pin's coordinates never
change when the viewport pans or zooms: that is the "fixed aim / coordinate
continuity" the workspace guarantees.

⚠ PALS's LAW: pin coordinates are exact (they are just numbers the caller set or
nudged); the *overlay* is a drawing aid. Detected landmarks used as anchors are the
same UNVERIFIED hints as elsewhere.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .image_compare import load_rgb
from .measure import (
    CoordinateSystem,
    CropTransform,
    Landmark,
    Measurement,
    crop_transform,
    denorm_point,
    draw_points_overlay,
    nice_step,
    normalize_point,
    point_frames,
    resolve_point_spec,
    structural_landmarks,
)

WORKSPACE_FILE = "workspace.json"
_TRUTHY_UNITS = ("norm", "px", "viewport")


@dataclass
class Pin:
    """A persisted anchor point, stored canonically in image pixels."""

    id: str
    x: float
    y: float
    group: str = ""
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "x": self.x, "y": self.y, "group": self.group, "label": self.label}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Pin":
        return cls(str(d["id"]), float(d["x"]), float(d["y"]),
                   str(d.get("group", "")), str(d.get("label", "")))


@dataclass
class WorkspaceState:
    """The persisted workspace: image binding, pins, viewport, id counter."""

    image_ref: str
    width: int
    height: int
    origin: str = "top-left"
    pins: list[Pin] = field(default_factory=list)
    viewport: tuple[float, float, float, float] | None = None
    viewport_name: str = "viewport"
    seq: int = 0
    checkpoints: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_ref": self.image_ref,
            "width": self.width,
            "height": self.height,
            "origin": self.origin,
            "pins": [p.to_dict() for p in self.pins],
            "viewport": list(self.viewport) if self.viewport else None,
            "viewport_name": self.viewport_name,
            "seq": self.seq,
            "checkpoints": self.checkpoints,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkspaceState":
        vp = d.get("viewport")
        return cls(
            image_ref=str(d["image_ref"]),
            width=int(d["width"]),
            height=int(d["height"]),
            origin=str(d.get("origin", "top-left")),
            pins=[Pin.from_dict(p) for p in d.get("pins", [])],
            viewport=tuple(float(v) for v in vp) if vp else None,
            viewport_name=str(d.get("viewport_name", "viewport")),
            seq=int(d.get("seq", 0)),
            checkpoints=list(d.get("checkpoints", [])),
        )

    def snapshot(self) -> dict[str, Any]:
        """A restorable snapshot of the mutable state (pins + viewport + id counter)."""
        return {
            "pins": [p.to_dict() for p in self.pins],
            "viewport": list(self.viewport) if self.viewport else None,
            "viewport_name": self.viewport_name,
            "seq": self.seq,
        }

    def restore(self, snap: dict[str, Any]) -> None:
        self.pins = [Pin.from_dict(p) for p in snap.get("pins", [])]
        vp = snap.get("viewport")
        self.viewport = tuple(float(v) for v in vp) if vp else None
        self.viewport_name = snap.get("viewport_name", "viewport")
        self.seq = int(snap.get("seq", self.seq))

    # -- coordinate helpers --------------------------------------------------
    def cs(self) -> CoordinateSystem:
        return CoordinateSystem(self.origin, self.width, self.height)

    def viewport_transform(self) -> CropTransform | None:
        if not self.viewport:
            return None
        return crop_transform(self.viewport_name, self.viewport, self.cs())

    def anchors(self) -> dict[str, Landmark]:
        """Structural anchors (A1..A9) plus every pin, keyed by id, for spec resolution."""
        cs = self.cs()
        out: dict[str, Landmark] = {lm.id: lm for lm in structural_landmarks(cs)}
        for p in self.pins:
            out[p.id] = Landmark(p.id, "pin", p.x, p.y, cs.to_cs(p.x, p.y), source="pin")
        return out


def load_state(session_dir: Path) -> WorkspaceState | None:
    path = session_dir / WORKSPACE_FILE
    if not path.is_file():
        return None
    return WorkspaceState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_state(session_dir: Path, state: WorkspaceState) -> None:
    (session_dir / WORKSPACE_FILE).write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# mutations
# ─────────────────────────────────────────────────────────────
def _select(state: WorkspaceState, select: Any) -> list[Pin]:
    """Resolve a selector to pins: None/'all' → all; {'ids': [...]}; {'group': g}."""
    if select in (None, "all", "*"):
        return list(state.pins)
    if isinstance(select, dict):
        if "ids" in select:
            want = {str(i) for i in select["ids"]}
            return [p for p in state.pins if p.id in want]
        if "group" in select:
            g = str(select["group"])
            return [p for p in state.pins if p.group == g]
    raise ValueError("select must be null/'all', {'ids': [...]}, or {'group': str}")


def add_pins(state: WorkspaceState, specs: Sequence[Any]) -> list[Pin]:
    """Resolve point specs (any frame; may reference existing pins) and add them."""
    cs = state.cs()
    vp = state.viewport_transform()
    added: list[Pin] = []
    for spec in specs:
        anchors = state.anchors()  # rebuilt so a spec can reference a just-added pin
        px, py = resolve_point_spec(spec, cs, anchors, vp)
        if isinstance(spec, dict) and spec.get("id"):
            pid = str(spec["id"])
        else:
            state.seq += 1
            pid = f"P{state.seq}"
        pin = Pin(pid, px, py,
                  str(spec.get("group", "")) if isinstance(spec, dict) else "",
                  str(spec.get("label", "")) if isinstance(spec, dict) else "")
        # replacing an existing id updates it in place (reuse / re-pin)
        state.pins = [p for p in state.pins if p.id != pid]
        state.pins.append(pin)
        added.append(pin)
    return added


def _delta_px(state: WorkspaceState, dx: float, dy: float, unit: str) -> tuple[float, float]:
    """Convert a nudge delta in the chosen unit to image pixels."""
    unit = unit if unit in _TRUTHY_UNITS else "norm"
    if unit == "px":
        return float(dx), float(dy)
    if unit == "viewport":
        vp = state.viewport_transform()
        if vp is None:
            raise ValueError("unit 'viewport' needs a viewport; set one first")
        # a viewport-fraction step, expressed in source pixels
        return denorm_point(dx, dy, vp.size_px[0], vp.size_px[1])
    return denorm_point(dx, dy, state.width, state.height)  # norm


def nudge_pins(state: WorkspaceState, select: Any, dx: float, dy: float, unit: str) -> list[Pin]:
    """Move selected pins by a delta (the AI 'mouse': e.g. 0.01 norm left = dx=-0.01)."""
    ddx, ddy = _delta_px(state, dx, dy, unit)
    moved = _select(state, select)
    for p in moved:
        p.x += ddx
        p.y += ddy
    return moved


def move_pins(state: WorkspaceState, select: Any, to: Any) -> list[Pin]:
    """Set selected pins' absolute position to a single point spec."""
    cs = state.cs()
    vp = state.viewport_transform()
    px, py = resolve_point_spec(to, cs, state.anchors(), vp)
    moved = _select(state, select)
    for p in moved:
        p.x, p.y = px, py
    return moved


def remove_pins(state: WorkspaceState, select: Any) -> int:
    victims = {p.id for p in _select(state, select)}
    before = len(state.pins)
    state.pins = [p for p in state.pins if p.id not in victims]
    return before - len(state.pins)


def set_viewport(state: WorkspaceState, box: Sequence[float] | None, *, name: str = "viewport") -> None:
    if box is None:
        state.viewport = None
        return
    if len(box) != 4:
        raise ValueError("viewport box must be [x, y, w, h] normalized 0..1")
    state.viewport = tuple(float(v) for v in box)
    state.viewport_name = name


def pan_viewport(state: WorkspaceState, dnx: float, dny: float) -> None:
    if not state.viewport:
        raise ValueError("no viewport to pan; set one first")
    x, y, w, h = state.viewport
    state.viewport = (x + float(dnx), y + float(dny), w, h)


def zoom_viewport(state: WorkspaceState, factor: float, aim: Any = None) -> None:
    """Zoom the viewport by ``factor`` about an aim point (default: viewport centre).

    The aim is any point spec (a pin id via {"landmark": "P1"}, norm, px, ...); it
    stays centred, so the target the AI is inspecting does not drift — fixed aim.
    """
    if not state.viewport:
        raise ValueError("no viewport to zoom; set one first")
    if factor <= 0:
        raise ValueError("zoom factor must be > 0")
    x, y, w, h = state.viewport
    if aim is not None:
        ax, ay = resolve_point_spec(aim, state.cs(), state.anchors(), state.viewport_transform())
        cx, cy = normalize_point(ax, ay, state.width, state.height)
    else:
        cx, cy = x + w / 2, y + h / 2
    nw, nh = w / factor, h / factor
    state.viewport = (cx - nw / 2, cy - nh / 2, nw, nh)


# ─────────────────────────────────────────────────────────────
# rendering
# ─────────────────────────────────────────────────────────────
def render(image_bytes: bytes, state: WorkspaceState, *,
           grid: bool = True, rulers: bool = True, connect: bool = False) -> Measurement:
    """Render the workspace: overlay with pins, an optional viewport crop, and spatial."""
    img = load_rgb(image_bytes)
    cs = state.cs()
    step = nice_step(max(state.width, state.height))
    lms = structural_landmarks(cs)
    vp_xform = state.viewport_transform()
    pts = [(p.x, p.y, p.label or p.id) for p in state.pins]

    overlay, crops = draw_points_overlay(
        img, cs, pts, viewport=vp_xform, landmarks=lms, step=step,
        label_every=2, grid=grid, rulers=rulers, connect=connect)

    pin_records = []
    for p in state.pins:
        rec = {"id": p.id, "group": p.group, "label": p.label}
        rec.update(point_frames(p.x, p.y, cs, vp_xform))
        pin_records.append(rec)

    spatial = {
        "image": {"width_px": state.width, "height_px": state.height},
        "coordinate_system": cs.describe(),
        "viewport": vp_xform.to_dict() if vp_xform else None,
        "pin_count": len(state.pins),
        "pins": pin_records,
        "reconstruction_hint": (
            "Pins persist across calls (reuse by id/group). Nudge in norm/px/viewport "
            "units to refine over passes; pins are image-anchored, so their coordinates "
            "hold as the viewport moves. Feed pins into construct_vectors to draw."
        ),
    }
    return Measurement(overlay=overlay, spatial=spatial, crops=crops)


def snap_pins(state: WorkspaceState, select: Any, image_bytes: bytes, *,
              to: str = "bright", radius: int = 4) -> list[Pin]:
    """Snap selected pins to the nearest salient pixel — pixel-accurate refinement.

    ``to``: ``bright`` / ``dark`` (extreme luminance in the window), ``edge`` (max
    gradient magnitude), or ``centroid`` (intensity-weighted centre). ``radius`` is
    the search window in pixels around each pin.
    """
    import numpy as np
    from io import BytesIO

    from PIL import Image

    if to not in ("bright", "dark", "edge", "centroid"):
        raise ValueError("snap `to` must be one of: bright, dark, edge, centroid")
    r = max(1, int(radius))
    gray = np.asarray(Image.open(BytesIO(image_bytes)).convert("L"), dtype=float)
    H, W = gray.shape
    snapped = _select(state, select)
    for p in snapped:
        cx, cy = int(round(p.x)), int(round(p.y))
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(W, cx + r + 1), min(H, cy + r + 1)
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        if to == "centroid":
            w = patch - patch.min()
            s = float(w.sum())
            if s > 0:
                ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
                p.x = float(x0 + (xs * w).sum() / s)
                p.y = float(y0 + (ys * w).sum() / s)
            continue
        if to == "edge":
            gy, gx = np.gradient(patch)
            field = np.hypot(gx, gy)
            idx = int(np.argmax(field))
        else:
            idx = int(np.argmax(patch) if to == "bright" else np.argmin(patch))
        py, px = np.unravel_index(idx, patch.shape)
        p.x, p.y = float(x0 + px), float(y0 + py)
    return snapped


# ─────────────────────────────────────────────────────────────
# geometry-constrained refinement (sub-pixel edges + symmetry/collinearity)
#
# The coarse ``snap_pins`` above is an integer-pixel argmax in a window; these place
# pins by the *right* method for a geometric mark — fit lines to edges (sub-pixel),
# intersect them for corners, and enforce the rigid constraints (bilateral symmetry,
# collinear edges) a luminance diff is blind to. Exact maths lives in
# :mod:`frameforge_vision.domain.geometry`; sub-pixel edge sampling in
# :mod:`frameforge_vision.infrastructure.edgesnap`.
# ─────────────────────────────────────────────────────────────
def fit_edge_pins(state: WorkspaceState, select: Any, image_bytes: bytes, *,
                  band: float = 6.0, step: float = 2.0,
                  min_strength: float = 6.0) -> tuple[list[Pin], dict[str, Any]]:
    """Refine selected pins onto one sub-pixel edge line, re-projecting each onto it.

    The first and last selected pins are the rough segment across the edge; the edge
    is located to sub-pixel precision and every selected pin is projected onto the
    fitted line — making them collinear AND edge-accurate. Falls back to a pure
    best-fit line through the pins when no confident edge is found (PALS: never trust
    a bad snap).
    """
    from ..domain.geometry import fit_line
    from . import edgesnap
    pins = _select(state, select)
    if len(pins) < 2:
        raise ValueError("action 'fit_edge' needs >=2 selected pins")
    res = edgesnap.refine_edge_line(image_bytes, (pins[0].x, pins[0].y),
                                    (pins[-1].x, pins[-1].y),
                                    band=band, step=step, min_strength=min_strength)
    if res.get("ok"):
        line = res["_line"]
        info = {"source": "subpixel-edge", "n_crossings": res["n_crossings"],
                "rms_residual_px": res["rms_residual_px"], "line": res["line"]}
    else:
        line = fit_line([(p.x, p.y) for p in pins])
        info = {"source": "collinear-only (no confident edge — PALS)",
                "reason": res.get("reason"), "line": line.to_dict()}
    for p in pins:
        p.x, p.y = line.project((p.x, p.y))
    return pins, info


def collinear_pins(state: WorkspaceState, select: Any) -> tuple[list[Pin], dict[str, Any]]:
    """Project selected pins onto their best-fit line — pure geometry, no image (≥2 pins)."""
    from ..domain.geometry import collinearity_residual, fit_line
    pins = _select(state, select)
    if len(pins) < 2:
        raise ValueError("action 'collinear' needs >=2 selected pins")
    pts = [(p.x, p.y) for p in pins]
    before = collinearity_residual(pts)
    line = fit_line(pts)
    for p in pins:
        p.x, p.y = line.project((p.x, p.y))
    return pins, {"residual_before": before, "line": line.to_dict()}


def symmetrize_pins(state: WorkspaceState, pairs: Sequence[Sequence[str]], *,
                    axis: float | None = None) -> tuple[list[tuple[Pin, Pin]], dict[str, Any]]:
    """Enforce bilateral symmetry over ``(leftId, rightId)`` pin pairs about a vertical axis.

    Snaps each pair symmetric about the consensus axis (median of pair mid-x, or a
    supplied ``axis``) and equalises their y. Returns the *pre-snap* symmetry report so
    the caller sees which pair was the outlier the metric could not.
    """
    from ..domain.geometry import reflect_across_vertical, symmetry_axis_x, symmetry_report
    if not pairs:
        raise ValueError("action 'symmetrize' needs 'pairs': [[leftId, rightId], ...]")
    byid = {p.id: p for p in state.pins}
    resolved: list[tuple[Pin, Pin]] = []
    pt_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for pr in pairs:
        if len(pr) != 2:
            raise ValueError(f"each pair must be [leftId, rightId]; got {pr!r}")
        lid, rid = str(pr[0]), str(pr[1])
        if lid not in byid or rid not in byid:
            raise ValueError(f"symmetrize: unknown pin id in pair {pr!r}")
        L, R = byid[lid], byid[rid]
        resolved.append((L, R))
        pt_pairs.append(((L.x, L.y), (R.x, R.y)))
    ax = float(axis) if axis is not None else symmetry_axis_x(pt_pairs)
    report = symmetry_report(pt_pairs, axis=ax)
    for L, R in resolved:
        rl = reflect_across_vertical((R.x, R.y), ax)      # mirror R into left space
        nlx, nly = (L.x + rl[0]) / 2.0, (L.y + rl[1]) / 2.0
        L.x, L.y = nlx, nly
        R.x, R.y = reflect_across_vertical((nlx, nly), ax)
    return resolved, {"axis_x": round(ax, 3), "report": report}


def intersect_to_pin(state: WorkspaceState, image_bytes: bytes, *,
                     edge1: Sequence[str], edge2: Sequence[str], target: str,
                     band: float = 6.0, step: float = 2.0,
                     min_strength: float = 6.0) -> tuple[tuple[float, float], dict[str, Any]]:
    """Set/create pin ``target`` at the corner where two edges meet (sub-pixel).

    ``edge1``/``edge2`` are lists of ≥2 existing pin ids roughly on each edge; each is
    refined to a sub-pixel line (falling back to the pins' best-fit line with no
    confident edge) and the two lines are intersected. This is the right way to place a
    corner: intersect two well-fit edges rather than eyeball the ill-conditioned tip.
    """
    from ..domain.geometry import fit_line
    from ..domain.geometry import intersect as _intersect
    from . import edgesnap
    byid = {p.id: p for p in state.pins}

    def _fit(ids: Sequence[str]) -> tuple[Any, dict[str, Any]]:
        pins = [byid[str(i)] for i in ids if str(i) in byid]
        if len(pins) < 2:
            raise ValueError("each edge needs >=2 existing pins")
        r = edgesnap.refine_edge_line(image_bytes, (pins[0].x, pins[0].y),
                                      (pins[-1].x, pins[-1].y),
                                      band=band, step=step, min_strength=min_strength)
        if r.get("ok"):
            return r["_line"], {"source": "subpixel-edge", "n_crossings": r["n_crossings"],
                                "rms_residual_px": r["rms_residual_px"]}
        return fit_line([(p.x, p.y) for p in pins]), {"source": "pins best-fit (no edge — PALS)"}

    l1, i1 = _fit(edge1)
    l2, i2 = _fit(edge2)
    cx, cy = _intersect(l1, l2)
    tid = str(target)
    existing = [p for p in state.pins if p.id == tid]
    if existing:
        existing[0].x, existing[0].y = cx, cy
    else:
        state.pins.append(Pin(tid, cx, cy))
    return (cx, cy), {"corner_px": [round(cx, 3), round(cy, 3)], "target": tid,
                      "edge1": i1, "edge2": i2}


def snap_pins_subpixel(state: WorkspaceState, select: Any, image_bytes: bytes, *,
                       band: float = 8.0, search_dir: Sequence[float] | None = None,
                       min_strength: float = 6.0) -> tuple[list[Pin], list[dict[str, Any]]]:
    """Slide selected pins onto the nearest edge to sub-pixel precision (edgesnap).

    ``search_dir`` is the axis to search along (the edge normal); omit it to snap
    perpendicular to the local image gradient. Pins with no confident edge in range are
    left untouched (reported ``ok=False``).
    """
    from . import edgesnap
    pins = _select(state, select)
    info: list[dict[str, Any]] = []
    sd = tuple(float(v) for v in search_dir) if search_dir else None
    for p in pins:
        res = edgesnap.snap_point_to_edge(image_bytes, (p.x, p.y), search_dir=sd,
                                          band=band, min_strength=min_strength)
        if res.get("ok"):
            p.x, p.y = float(res["point_px"][0]), float(res["point_px"][1])
        info.append({"id": p.id, **{k: res.get(k) for k in ("ok", "moved_px", "strength")}})
    return pins, info


def transform_pins(state: WorkspaceState, select: Any, *,
                   translate: Sequence[float] = (0.0, 0.0), scale: float = 1.0,
                   rotate: float = 0.0, about: Any = None) -> list[Pin]:
    """Translate/scale/rotate selected pins together about a pivot (default: their centroid).

    Adjusts several pins as a rigid+scale group — for correcting proportions,
    alignment, and local distortion. ``translate`` is in image pixels, ``rotate`` in
    degrees; ``about`` is any point spec (else the selection centroid).
    """
    import math

    pts = _select(state, select)
    if not pts:
        return pts
    if about is not None:
        ax, ay = resolve_point_spec(about, state.cs(), state.anchors(), state.viewport_transform())
    else:
        ax = sum(p.x for p in pts) / len(pts)
        ay = sum(p.y for p in pts) / len(pts)
    th = math.radians(float(rotate))
    co, si = math.cos(th), math.sin(th)
    s = float(scale)
    tx, ty = float(translate[0]), float(translate[1])
    for p in pts:
        dx, dy = (p.x - ax) * s, (p.y - ay) * s
        p.x = ax + (dx * co - dy * si) + tx
        p.y = ay + (dx * si + dy * co) + ty
    return pts


def checkpoint_state(state: WorkspaceState, tag: str | None = None) -> int:
    """Push a full snapshot of the pins/viewport; returns its index for revert."""
    state.checkpoints.append({"tag": tag, **state.snapshot()})
    return len(state.checkpoints) - 1


def revert_state(state: WorkspaceState, index: int = -1) -> dict[str, Any]:
    """Restore a checkpoint (default: latest) and drop it + everything pushed after."""
    if not state.checkpoints:
        raise ValueError("no checkpoints to revert to")
    i = index if index >= 0 else len(state.checkpoints) + index
    if i < 0 or i >= len(state.checkpoints):
        raise ValueError(f"no checkpoint at index {index}")
    snap = state.checkpoints[i]
    state.restore(snap)
    del state.checkpoints[i:]
    return {"reverted_to": i, "tag": snap.get("tag")}
