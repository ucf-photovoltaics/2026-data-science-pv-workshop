"""Image feature extractors."""

from __future__ import annotations

import numpy as np

from rdstemplate.features.base import FeatureExtractor, register_extractor


@register_extractor("image_basic_stats")
class ImageBasicStats(FeatureExtractor):
    """Per-channel mean/std and overall brightness."""

    def extract(self, sample_id: str, exposure_step, data) -> dict:
        if data is None:
            return {}
        arr = np.asarray(data, dtype=float)
        # Collapse to (H, W, C) regardless of whether alpha channel exists.
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        result: dict = {}
        for c in range(arr.shape[2]):
            channel_name = ["r", "g", "b", "a"][c] if arr.shape[2] <= 4 else str(c)
            result[f"image_basic_stats__{channel_name}_mean"] = float(arr[:, :, c].mean())
            result[f"image_basic_stats__{channel_name}_std"] = float(arr[:, :, c].std())
        result["image_basic_stats__brightness"] = float(arr[..., :3].mean())
        return result
