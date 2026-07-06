"""Modality-specific file readers.

Each loader accepts a directory path (or s3:// prefix after sync) and returns
raw data in a format suitable for the corresponding FeatureExtractor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_curve(directory: Path | str, exposure_step) -> pd.DataFrame | None:
    """Load a curve CSV for one exposure step.

    Expected filename: ``<exposure_step>.csv`` with columns [x, y].
    Returns None if the file does not exist (modality not measured at this step).
    """
    path = Path(directory) / f"{exposure_step}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_spectrum(directory: Path | str, exposure_step) -> pd.DataFrame | None:
    """Load a spectrum CSV for one exposure step.

    Expected filename: ``<exposure_step>.csv`` with columns [wavelength, intensity].
    """
    path = Path(directory) / f"{exposure_step}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_image(directory: Path | str, exposure_step) -> np.ndarray | None:
    """Load an image for one exposure step as an RGBA/RGB numpy array.

    Tries ``<exposure_step>.png`` then ``<exposure_step>.jpg``.
    """
    from PIL import Image  # noqa: PLC0415

    for ext in (".png", ".jpg", ".jpeg"):
        path = Path(directory) / f"{exposure_step}{ext}"
        if path.exists():
            return np.array(Image.open(path))
    return None


def load_timeseries(directory: Path | str, exposure_step) -> pd.DataFrame | None:
    """Load a timeseries CSV for one exposure step.

    Expected filename: ``<exposure_step>.csv`` with columns [time, value].
    """
    path = Path(directory) / f"{exposure_step}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)
