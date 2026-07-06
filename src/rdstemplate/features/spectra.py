"""Spectral feature extractors."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from rdstemplate.features.base import FeatureExtractor, register_extractor


@register_extractor("spectra_peaks")
class SpectraPeaks(FeatureExtractor):
    """Number of peaks, dominant peak position, and intensity statistics."""

    def extract(self, sample_id: str, exposure_step, data) -> dict:
        if data is None or data.empty:
            return {}
        intensity = data.iloc[:, 1].to_numpy(dtype=float)
        wavelength = data.iloc[:, 0].to_numpy(dtype=float)
        peaks, _ = find_peaks(intensity, height=intensity.mean())
        n_peaks = len(peaks)
        dominant_wl = float(wavelength[intensity.argmax()])
        return {
            "spectra_peaks__n_peaks": n_peaks,
            "spectra_peaks__dominant_wavelength": dominant_wl,
            "spectra_peaks__intensity_max": float(intensity.max()),
            "spectra_peaks__intensity_mean": float(intensity.mean()),
        }
