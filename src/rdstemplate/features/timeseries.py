"""Timeseries feature extractors."""

from __future__ import annotations

import numpy as np

from rdstemplate.features.base import FeatureExtractor, register_extractor


@register_extractor("timeseries_summary")
class TimeseriesSummary(FeatureExtractor):
    """Summary statistics for a 1-D timeseries (mean, std, min, max, trend slope)."""

    def extract(self, sample_id: str, exposure_step, data) -> dict:
        if data is None or data.empty:
            return {}
        values = data.iloc[:, 1].to_numpy(dtype=float)
        times = data.iloc[:, 0].to_numpy(dtype=float)
        # Ordinary-least-squares slope: trend direction over the window.
        if len(times) > 1:
            slope = float(np.polyfit(times, values, 1)[0])
        else:
            slope = float("nan")
        return {
            "timeseries_summary__mean": float(values.mean()),
            "timeseries_summary__std": float(values.std()),
            "timeseries_summary__min": float(values.min()),
            "timeseries_summary__max": float(values.max()),
            "timeseries_summary__slope": slope,
        }
