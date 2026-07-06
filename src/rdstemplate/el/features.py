"""Aggregate per-cell defect masks into per-image EL features and merge them
onto the tidy dataframe.

``segment.segment_dataset`` returns one record per cell. Here we roll those up to
one row per module image (matching ``metadata.csv``'s grain) and join on the EL
image filename, producing the ``el_*`` feature columns the model will use.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFECT_CLASSES = ["crack", "contact", "interconnect", "corrosion"]
CRACKED_CELL_THRESHOLD = 0.01  # a cell counts as "cracked" above this crack fraction


def aggregate_cells(cell_records: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Roll per-cell records up to one row per (module, image).

    Produces, per module image:
      - ``el_<class>_frac_mean``  mean fraction of each defect class over cells
      - ``el_defect_frac_total``  mean total defective-pixel fraction
      - ``el_crack_frac_max``     worst single cell's crack fraction
      - ``el_frac_cells_cracked`` share of cells above the crack threshold
      - ``el_n_cells``            cells in the image
    """
    df = pd.DataFrame(cell_records)
    if df.empty:
        return df
    df["_total"] = df[DEFECT_CLASSES].sum(axis=1)
    df["_cracked"] = df["crack"] > CRACKED_CELL_THRESHOLD

    g = df.groupby(["module", "image"])
    out = g[DEFECT_CLASSES].mean().rename(columns={c: f"el_{c}_frac_mean" for c in DEFECT_CLASSES})
    out["el_defect_frac_total"] = g["_total"].mean()
    out["el_crack_frac_max"] = g["crack"].max()
    out["el_frac_cells_cracked"] = g["_cracked"].mean()
    out["el_n_cells"] = g.size()
    return out.reset_index()


def merge_features(metadata: str | Path | pd.DataFrame,
                   image_features: pd.DataFrame,
                   *, image_col: str = "image") -> pd.DataFrame:
    """Left-join per-image EL features onto ``metadata`` by EL image filename.

    ``image_features``' ``image`` column holds the filename stem; metadata's
    ``el_image_filename`` holds ``stem.png``. Rows with no segmentation yet keep
    NaN EL features (outer-join-safe, like the rest of the pipeline).
    """
    meta = pd.read_csv(metadata) if not isinstance(metadata, pd.DataFrame) else metadata.copy()
    feats = image_features.copy()
    feats["el_image_filename"] = feats[image_col].astype(str) + ".png"
    feats = feats.drop(columns=[c for c in ("module", image_col) if c in feats.columns])
    return meta.merge(feats, on="el_image_filename", how="left")


def build_and_merge(cell_records: list[dict] | pd.DataFrame,
                    metadata: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Convenience: aggregate cells then merge onto metadata in one call."""
    return merge_features(metadata, aggregate_cells(cell_records))
