"""Shared pytest fixtures."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def sample_data_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "sample"


@pytest.fixture(scope="session")
def example_config_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "example.yaml"


@pytest.fixture
def simple_metadata() -> pd.DataFrame:
    """3 samples × 4 steps — the canonical test grain."""
    rows = []
    for sid in ["S1", "S2", "S3"]:
        for step in [1, 2, 3, 4]:
            rows.append({"sample_id": sid, "exposure_step": step, "outcome": step * 2.0})
    return pd.DataFrame(rows)


@pytest.fixture
def curve_df() -> pd.DataFrame:
    x = np.linspace(0, 10, 20)
    y = np.sin(x)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture
def spectrum_df() -> pd.DataFrame:
    wl = np.linspace(400, 700, 40)
    intensity = np.exp(-((wl - 550) ** 2) / 1000)
    return pd.DataFrame({"wavelength": wl, "intensity": intensity})


@pytest.fixture
def image_array() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.uniform(0, 255, (8, 8, 3))).astype(np.uint8)


@pytest.fixture
def timeseries_df() -> pd.DataFrame:
    t = np.linspace(0, 1, 20)
    v = np.sin(2 * np.pi * t)
    return pd.DataFrame({"time": t, "value": v})
