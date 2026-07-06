"""Tests: registered extractor for each modality runs on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdstemplate.features.curves import CurveAUC
from rdstemplate.features.images import ImageBasicStats
from rdstemplate.features.spectra import SpectraPeaks
from rdstemplate.features.timeseries import TimeseriesSummary
from rdstemplate.features.base import EXTRACTOR_REGISTRY


class TestRegistry:
    def test_all_builtins_registered(self):
        for name in ("curve_auc", "spectra_peaks", "image_basic_stats", "timeseries_summary"):
            assert name in EXTRACTOR_REGISTRY, f"'{name}' not in registry"


class TestCurveAUC:
    def test_returns_expected_keys(self, curve_df):
        result = CurveAUC().extract("S1", 1, curve_df)
        assert set(result) == {
            "curve_auc__auc",
            "curve_auc__y_max",
            "curve_auc__y_mean",
            "curve_auc__n_points",
        }

    def test_auc_numeric(self, curve_df):
        result = CurveAUC().extract("S1", 1, curve_df)
        assert isinstance(result["curve_auc__auc"], float)

    def test_none_data_returns_empty(self):
        assert CurveAUC().extract("S1", 1, None) == {}

    def test_empty_df_returns_empty(self):
        assert CurveAUC().extract("S1", 1, pd.DataFrame()) == {}


class TestSpectraPeaks:
    def test_returns_expected_keys(self, spectrum_df):
        result = SpectraPeaks().extract("S1", 1, spectrum_df)
        assert set(result) == {
            "spectra_peaks__n_peaks",
            "spectra_peaks__dominant_wavelength",
            "spectra_peaks__intensity_max",
            "spectra_peaks__intensity_mean",
        }

    def test_dominant_wavelength_near_peak(self, spectrum_df):
        result = SpectraPeaks().extract("S1", 1, spectrum_df)
        # Our synthetic spectrum peaks near 550 nm.
        assert abs(result["spectra_peaks__dominant_wavelength"] - 550) < 10

    def test_none_data_returns_empty(self):
        assert SpectraPeaks().extract("S1", 1, None) == {}


class TestImageBasicStats:
    def test_returns_rgb_keys(self, image_array):
        result = ImageBasicStats().extract("S1", 1, image_array)
        for channel in ("r", "g", "b"):
            assert f"image_basic_stats__{channel}_mean" in result
            assert f"image_basic_stats__{channel}_std" in result
        assert "image_basic_stats__brightness" in result

    def test_none_data_returns_empty(self):
        assert ImageBasicStats().extract("S1", 1, None) == {}

    def test_grayscale_array(self):
        arr = np.full((4, 4), 128, dtype=np.uint8)
        result = ImageBasicStats().extract("S1", 1, arr)
        assert "image_basic_stats__brightness" in result


class TestTimeseriesSummary:
    def test_returns_expected_keys(self, timeseries_df):
        result = TimeseriesSummary().extract("S1", 1, timeseries_df)
        assert set(result) == {
            "timeseries_summary__mean",
            "timeseries_summary__std",
            "timeseries_summary__min",
            "timeseries_summary__max",
            "timeseries_summary__slope",
        }

    def test_none_data_returns_empty(self):
        assert TimeseriesSummary().extract("S1", 1, None) == {}
