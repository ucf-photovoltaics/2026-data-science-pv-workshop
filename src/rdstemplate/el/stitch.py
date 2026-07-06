"""Stage 5 — re-stitch masked cells back into a full module image.

Reads a folder of ``cell_r##_c##.png`` overlays (stage 4 output), places them on
their grid (inferred from the filenames), and writes one module image. Cells are
resized to a common size so the grid is clean regardless of small per-cell
rounding differences from the even split.

Typical use::

    from rdstemplate.el import stitch
    stitch.stitch_dataset(
        "data/workshop/images/04_cell_masks",
        "data/workshop/images/05_module_restitched")
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

_RC = re.compile(r"cell_r(\d+)_c(\d+)")


def stitch_cells(cell_dir: str | Path) -> np.ndarray | None:
    """Assemble ``cell_r##_c##.png`` files in ``cell_dir`` into one module image."""
    cell_dir = Path(cell_dir)
    cells: dict[tuple[int, int], np.ndarray] = {}
    for f in cell_dir.glob("cell_r*_c*.png"):
        m = _RC.search(f.name)
        if not m:
            continue
        cells[(int(m.group(1)), int(m.group(2)))] = cv2.imread(str(f), cv2.IMREAD_COLOR)
    if not cells:
        return None
    rows = max(r for r, _ in cells)
    cols = max(c for _, c in cells)
    # common cell size = median across cells (robust to rounding differences)
    ch = int(np.median([im.shape[0] for im in cells.values()]))
    cw = int(np.median([im.shape[1] for im in cells.values()]))
    canvas = np.zeros((rows * ch, cols * cw, 3), np.uint8)
    for (r, c), im in cells.items():
        canvas[(r - 1) * ch:r * ch, (c - 1) * cw:c * cw] = cv2.resize(im, (cw, ch))
    return canvas


def stitch_file(cell_dir: str | Path, out_path: str | Path) -> bool:
    """Stitch one cell folder and write the module image. Returns success."""
    module = stitch_cells(cell_dir)
    if module is None:
        return False
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), module)
    return True


def stitch_dataset(input_root: str | Path, output_root: str | Path) -> list[str]:
    """Re-stitch every ``<module>/<image>/`` cell folder into ``output_root``.

    Writes ``output_root/<module>/<image>.png``. Returns the written paths.
    """
    input_root, output_root = Path(input_root), Path(output_root)
    written: list[str] = []
    for module_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        for image_dir in sorted(p for p in module_dir.iterdir() if p.is_dir()):
            out = output_root / module_dir.name / f"{image_dir.name}.png"
            if stitch_file(image_dir, out):
                written.append(str(out))
    return written
