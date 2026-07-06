"""Stage 3 — split a rectified module into labeled individual cell images.

After rectification the module is an upright rectangle, so cells fall on a regular
grid and an even split recovers them. Each cell is written as
``cell_r{row}_c{col}.png`` (1-indexed), which makes stage-5 re-stitching trivial.

Grids are per module (from the module's cell count / standard layout):

    60-cell -> 6 x 10      72-cell -> 6 x 12      96-cell -> 8 x 12

Typical use::

    from rdstemplate.el import cells
    cells.crop_dataset(
        "data/workshop/images/02_module_rectified",
        "data/workshop/images/03_cells",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# module serial -> (rows, cols). Verified against the rectified images
# (SolarCity 8x12 lands exactly on the cell gaps; 60-cell aspects match 6x10).
GRID: dict[str, tuple[int, int]] = {
    "2617127550699": (6, 10),      # Axitec, 60-cell
    "609K3ZH1A4NA": (6, 10),       # LG, 60-cell
    "321617132567500337": (6, 12), # Q-Cells, 72-cell
    "H5NE30994": (8, 12),          # SolarCity, 96-cell
}


@dataclass
class CropResult:
    path: str
    module: str
    rows: int
    cols: int
    n_cells: int


def split_cells(gray: np.ndarray, rows: int, cols: int, *, inset: int = 0):
    """Yield ``(row, col, cell_image)`` for an even ``rows x cols`` split.

    ``row``/``col`` are 1-indexed. ``inset`` trims that many pixels off each side
    of the module first, to drop any residual frame bleed.
    """
    h, w = gray.shape[:2]
    if inset > 0:
        gray = gray[inset:h - inset, inset:w - inset]
        h, w = gray.shape[:2]
    ys = np.linspace(0, h, rows + 1).round().astype(int)
    xs = np.linspace(0, w, cols + 1).round().astype(int)
    for r in range(rows):
        for c in range(cols):
            cell = gray[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            yield r + 1, c + 1, cell


def crop_file(src: str | Path, out_dir: str | Path, rows: int, cols: int,
              *, inset: int = 0) -> int:
    """Split one rectified image into cell PNGs under ``out_dir``. Returns count."""
    src, out_dir = Path(src), Path(out_dir)
    gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for r, c, cell in split_cells(gray, rows, cols, inset=inset):
        cv2.imwrite(str(out_dir / f"cell_r{r:02d}_c{c:02d}.png"), cell)
        n += 1
    return n


def crop_dataset(input_root: str | Path, output_root: str | Path, *,
                 grid: dict[str, tuple[int, int]] | None = None,
                 inset: int = 0) -> list[CropResult]:
    """Split every rectified module under ``input_root`` into ``output_root``.

    Layout: ``output_root/<module>/<image_stem>/cell_r##_c##.png`` — one folder of
    cells per source image. Returns per-image diagnostics.
    """
    grid = grid or GRID
    input_root, output_root = Path(input_root), Path(output_root)
    results: list[CropResult] = []
    for module_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        module = module_dir.name
        if module not in grid:
            raise KeyError(f"No cell grid known for module {module!r}; add it to GRID.")
        rows, cols = grid[module]
        for src in sorted(module_dir.glob("*.png")):
            out_dir = output_root / module / src.stem
            n = crop_file(src, out_dir, rows, cols, inset=inset)
            results.append(CropResult(str(src), module, rows, cols, n))
    return results
