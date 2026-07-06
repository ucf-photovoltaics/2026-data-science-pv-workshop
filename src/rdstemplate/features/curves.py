"""Curve feature extractors."""

from __future__ import annotations

import numpy as np

from rdstemplate.features.base import FeatureExtractor, register_extractor


@register_extractor("curve_auc")
class CurveAUC(FeatureExtractor):
    """Area under the curve (trapezoidal) and basic curve statistics."""

    def extract(self, sample_id: str, exposure_step, data) -> dict:
        if data is None or data.empty:
            return {}
        x = data.iloc[:, 0].to_numpy(dtype=float)
        y = data.iloc[:, 1].to_numpy(dtype=float)
        auc = float(np.trapezoid(y, x))
        return {
            "curve_auc__auc": auc,
            "curve_auc__y_max": float(y.max()),
            "curve_auc__y_mean": float(y.mean()),
            "curve_auc__n_points": len(y),
        }
