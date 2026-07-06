"""Sample metadata loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rdstemplate.config import MetadataConfig


def load_metadata(cfg: MetadataConfig) -> pd.DataFrame:
    """Load the metadata CSV and validate required columns.

    Returns a DataFrame with at minimum columns [sample_id_col, exposure_step_col].
    Row grain: one row per (sample_id, exposure_step) observation.
    """
    path = Path(cfg.file)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)

    for col in (cfg.sample_id_col, cfg.exposure_step_col):
        if col not in df.columns:
            raise ValueError(
                f"Metadata file {path} is missing required column '{col}'. "
                f"Available columns: {list(df.columns)}"
            )

    # Normalise column names to canonical names used throughout the package.
    df = df.rename(
        columns={
            cfg.sample_id_col: "sample_id",
            cfg.exposure_step_col: "exposure_step",
        }
    )

    dupes = df.duplicated(subset=["sample_id", "exposure_step"])
    if dupes.any():
        raise ValueError(
            f"Duplicate (sample_id, exposure_step) rows found in metadata:\n"
            f"{df[dupes][['sample_id', 'exposure_step']].head()}"
        )

    return df


def exposure_steps_for_sample(metadata: pd.DataFrame, sample_id: str) -> list:
    """Return the ordered list of exposure steps for a given sample."""
    rows = metadata[metadata["sample_id"] == sample_id]
    return rows["exposure_step"].tolist()
