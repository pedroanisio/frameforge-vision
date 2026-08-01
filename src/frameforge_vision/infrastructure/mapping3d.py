"""2D↔3D coordinate mapping: homography rectification, plane lift, and projection.

Raster reconstruction is not always flat. This module transposes measured 2D image
coordinates into other frames so a raster can seed perspective correction and 3D
reconstruction:

- ``homography`` — fit a projective transform from >=4 point correspondences and
  apply it to points. Rectifies a perspective-distorted plane (e.g. a photographed
  facade) back to a fronto-parallel view, or maps points from source→reference.
- ``to_3d`` — lift 2D image points onto a 3D plane defined by an origin + two basis
  vectors (default: the z=0 plane), turning a flat drawing into a spatial reference.
- ``project`` — project 3D points to 2D through the SDK :class:`Camera` (the same
  math the 3D renderer uses), returning both NDC and pixel coordinates.

Honest scope: ``homography`` is a plane-to-plane projective map (no lens distortion
model); ``project`` is a pinhole camera. Full multi-view calibration is out of scope.
numpy backs the homography least-squares fit (present in the ``vision`` group).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

_NUMPY_HINT = ("3D/homography mapping needs numpy; install the `vision` dependency group.")


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is in the vision group
        raise RuntimeError(_NUMPY_HINT) from exc
    return np


# ─────────────────────────────────────────────────────────────
# homography (2D projective)
# ─────────────────────────────────────────────────────────────
def fit_homography(pairs: Sequence[tuple[Sequence[float], Sequence[float]]]):
    """Least-squares 3x3 homography mapping src→dst from >=4 point pairs."""
    np = _np()
    if len(pairs) < 4:
        raise ValueError("homography needs >= 4 point pairs")
    A: list[list[float]] = []
    b: list[float] = []
    for (s, d) in pairs:
        sx, sy = float(s[0]), float(s[1])
        dx, dy = float(d[0]), float(d[1])
        A.append([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy])
        b.append(dx)
        A.append([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy])
        b.append(dy)
    h, *_ = np.linalg.lstsq(np.array(A, dtype="float64"), np.array(b, dtype="float64"), rcond=None)
    return [[float(h[0]), float(h[1]), float(h[2])],
            [float(h[3]), float(h[4]), float(h[5])],
            [float(h[6]), float(h[7]), 1.0]]


def apply_homography(H, x: float, y: float) -> tuple[float, float]:
    wx = H[0][0] * x + H[0][1] * y + H[0][2]
    wy = H[1][0] * x + H[1][1] * y + H[1][2]
    w = H[2][0] * x + H[2][1] * y + H[2][2]
    if abs(w) < 1e-12:
        raise ValueError("homography maps a point to infinity (w≈0)")
    return wx / w, wy / w


def warp_image(image_bytes: bytes, H, out_size: Sequence[int]):
    """Rectify/dewarp an image by homography ``H`` into an ``out_size`` (w, h) canvas.

    Returns a PIL image. Needs OpenCV (the ``vision`` group). Use it to flatten a
    perspective-distorted plane (e.g. a photographed sign) before tracing it.
    """
    from io import BytesIO

    import cv2
    from PIL import Image

    np = _np()
    src = np.asarray(Image.open(BytesIO(image_bytes)).convert("RGB"))
    Hm = np.array(H, dtype="float64")
    w, h = int(out_size[0]), int(out_size[1])
    out = cv2.warpPerspective(src, Hm, (w, h), flags=cv2.INTER_LINEAR,
                              borderValue=(255, 255, 255))
    return Image.fromarray(out)


def homography_map(pairs, points) -> dict[str, Any]:
    H = fit_homography(pairs)
    mapped = [list(apply_homography(H, float(p[0]), float(p[1]))) for p in points]
    residuals = []
    for (s, d) in pairs:
        mx, my = apply_homography(H, float(s[0]), float(s[1]))
        residuals.append(round(math.hypot(mx - float(d[0]), my - float(d[1])), 4))
    rms = round(math.sqrt(sum(r * r for r in residuals) / len(residuals)), 4) if residuals else 0.0
    return {
        "mode": "homography",
        "matrix": [[round(v, 8) for v in row] for row in H],
        "mapped_points": [[round(v, 3) for v in p] for p in mapped],
        "fit_residuals_px": residuals,
        "rms_residual_px": rms,
        "note": "Projective plane-to-plane map (no lens distortion). Apply: [x',y',w']=H·[x,y,1]; result=(x'/w', y'/w').",
    }


# ─────────────────────────────────────────────────────────────
# 2D -> 3D plane lift
# ─────────────────────────────────────────────────────────────
def lift_to_plane(points, *, origin=(0.0, 0.0, 0.0), u=(1.0, 0.0, 0.0),
                  v=(0.0, 1.0, 0.0)) -> dict[str, Any]:
    """Map 2D image points onto a 3D plane: P = origin + x·u + y·v."""
    ox, oy, oz = (float(c) for c in origin)
    ux, uy, uz = (float(c) for c in u)
    vx, vy, vz = (float(c) for c in v)
    out = []
    for p in points:
        x, y = float(p[0]), float(p[1])
        out.append([round(ox + x * ux + y * vx, 4),
                    round(oy + x * uy + y * vy, 4),
                    round(oz + x * uz + y * vz, 4)])
    return {
        "mode": "to_3d",
        "plane": {"origin": [ox, oy, oz], "u": [ux, uy, uz], "v": [vx, vy, vz]},
        "points_3d": out,
        "note": "Image (x, y) lifted onto the plane spanned by u, v at origin. Default is the z=0 plane.",
    }


# ─────────────────────────────────────────────────────────────
# 3D -> 2D projection (pinhole camera)
# ─────────────────────────────────────────────────────────────
def project_points(points_3d, *, camera: dict[str, Any] | None = None,
                   width: int | None = None, height: int | None = None) -> dict[str, Any]:
    """Project 3D points to 2D through the SDK Camera; return NDC and (optional) pixels."""
    from frameforge.sdk.geometry import Camera, Vec3

    cam_kw = dict(camera or {})
    def _vec(key, default):
        val = cam_kw.get(key)
        return Vec3(*(float(c) for c in val)) if val else default

    cam = Camera(
        eye=_vec("eye", Vec3(3.0, 2.5, 4.0)),
        target=_vec("target", Vec3(0.0, 0.0, 0.0)),
        up=_vec("up", Vec3(0.0, 1.0, 0.0)),
        fov=float(cam_kw.get("fov", 45.0)),
        aspect=float(cam_kw.get("aspect", (width / height) if (width and height) else 1.0)),
        near=float(cam_kw.get("near", 0.1)),
        far=float(cam_kw.get("far", 100.0)),
    )
    ndc = []
    pixels = []
    for p in points_3d:
        v = cam.project(Vec3(float(p[0]), float(p[1]), float(p[2])))
        ndc.append([round(v.x, 5), round(v.y, 5)])
        if width and height:
            pixels.append([round((v.x + 1) / 2 * width, 3), round((v.y + 1) / 2 * height, 3)])
    result: dict[str, Any] = {
        "mode": "project",
        "camera": {"eye": list(cam.eye.tuple()), "target": list(cam.target.tuple()),
                   "up": list(cam.up.tuple()), "fov": cam.fov, "aspect": round(cam.aspect, 4)},
        "points_ndc": ndc,
        "note": "NDC in [-1, 1]. pixels map NDC to the given width/height (y already down).",
    }
    if pixels:
        result["points_px"] = pixels
    return result
