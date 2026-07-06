"""Stage 2 — rectify a raw EL module image to a straight-on rectangle.

The module fills most of an EL frame but sits on a dark border and is usually
slightly rotated/skewed. We isolate the bright module region, fit its minimum-
area rectangle, and perspective-warp it to an upright rectangle. This is more
robust (and has fewer magic numbers) than Hough-line edge fitting for these
near-axis-aligned, well-framed images.

Typical use::

    from rdstemplate.el import rectify
    rectify.rectify_dataset(
        "data/workshop/images/01_input",
        "data/workshop/images/02_module_rectified",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class RectifyResult:
    """Diagnostics for one rectified image."""

    path: str
    ok: bool
    area_frac: float          # module area / frame area (detection confidence)
    angle: float              # minAreaRect angle (deg)
    out_size: tuple[int, int] # (width, height) of the rectified image
    note: str = ""


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def find_module_corners(gray: np.ndarray, *, min_area_frac: float = 0.4):
    """Return the module's 4 corners (tl, tr, br, bl) or ``None`` if not found.

    Otsu-thresholds the (bright) module against the dark frame, closes/opens to a
    solid blob, takes the largest contour, and fits its minimum-area rectangle.
    """
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((25, 25), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)
    area_frac = float(cv2.contourArea(c) / (gray.shape[0] * gray.shape[1]))
    rect = cv2.minAreaRect(c)
    corners = _order_corners(cv2.boxPoints(rect))
    if area_frac < min_area_frac:
        return None, area_frac, float(rect[2])
    return corners, area_frac, float(rect[2])


def warp_to_rect(gray: np.ndarray, corners: np.ndarray, *, inset: int = 0) -> np.ndarray:
    """Perspective-warp ``gray`` so ``corners`` map to an upright rectangle.

    ``inset`` trims that many pixels inward on every side to drop frame bleed.
    """
    tl, tr, br, bl = corners
    width = int(max(np.hypot(*(br - bl)), np.hypot(*(tr - tl))))
    height = int(max(np.hypot(*(tr - br)), np.hypot(*(tl - bl))))
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(gray, M, (width, height))
    if inset > 0 and height > 2 * inset and width > 2 * inset:
        warped = warped[inset:height - inset, inset:width - inset]
    return warped


def rectify_file(src: str | Path, dst: str | Path, *, inset: int = 0,
                 min_area_frac: float = 0.4) -> RectifyResult:
    """Rectify one EL image ``src`` and write it to ``dst``."""
    src, dst = Path(src), Path(dst)
    gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return RectifyResult(str(src), False, 0.0, 0.0, (0, 0), "unreadable")
    corners, area_frac, angle = find_module_corners(gray, min_area_frac=min_area_frac)
    if corners is None:
        return RectifyResult(str(src), False, area_frac, angle, (0, 0),
                             "module not found")
    warped = warp_to_rect(gray, corners, inset=inset)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), warped)
    return RectifyResult(str(src), True, area_frac, angle,
                         (warped.shape[1], warped.shape[0]))


def rectify_dataset(input_root: str | Path, output_root: str | Path, *,
                    inset: int = 0, min_area_frac: float = 0.4) -> list[RectifyResult]:
    """Rectify every ``<module>/*.png`` under ``input_root`` into ``output_root``.

    Mirrors the ``<module>/<filename>.png`` layout. Returns per-image diagnostics.
    """
    input_root, output_root = Path(input_root), Path(output_root)
    results: list[RectifyResult] = []
    for module_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        for src in sorted(module_dir.glob("*.png")):
            dst = output_root / module_dir.name / src.name
            results.append(rectify_file(src, dst, inset=inset,
                                        min_area_frac=min_area_frac))
    return results
