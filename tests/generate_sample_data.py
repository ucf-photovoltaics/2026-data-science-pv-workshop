"""Generate a small synthetic dataset in data/sample/ for tests and demos.

Run from the repo root:
    python tests/generate_sample_data.py

Design:
- 3 samples: S1, S2, S3
- 4 exposure steps: 1, 2, 3, 4
- Curves and timeseries: measured at ALL steps (1-4)
- Spectra: measured at steps 1 and 3 only (misaligned on purpose)
- Images: measured at steps 2 and 4 only (misaligned on purpose)
- Metadata: one row per (sample_id, exposure_step)
- outcome: synthetic linear function of exposure_step + noise
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "sample"

SAMPLES = ["S1", "S2", "S3"]
ALL_STEPS = [1, 2, 3, 4]
SPECTRA_STEPS = [1, 3]      # subset — triggers misaligned-step merge path
IMAGE_STEPS = [2, 4]        # different subset

RNG = np.random.default_rng(42)


def make_metadata() -> None:
    rows = []
    for sample_id in SAMPLES:
        base_outcome = RNG.uniform(10, 20)
        for step in ALL_STEPS:
            outcome = base_outcome + step * RNG.uniform(0.5, 1.5)
            rows.append(
                {
                    "sample_id": sample_id,
                    "exposure_step": step,
                    "outcome": round(outcome, 4),
                    "sample_group": "A" if sample_id in ("S1", "S2") else "B",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "metadata.csv", index=False)
    print(f"  metadata.csv: {len(df)} rows")


def make_curves() -> None:
    for sample_id in SAMPLES:
        d = DATA_DIR / sample_id / "curves"
        d.mkdir(parents=True, exist_ok=True)
        for step in ALL_STEPS:
            x = np.linspace(0, 10, 50)
            y = (step * 0.5 + RNG.normal(0, 0.1, 50)).cumsum()
            pd.DataFrame({"x": x, "y": y}).to_csv(d / f"{step}.csv", index=False)
    print(f"  curves: {len(SAMPLES)} samples × {len(ALL_STEPS)} steps")


def make_spectra() -> None:
    for sample_id in SAMPLES:
        d = DATA_DIR / sample_id / "spectra"
        d.mkdir(parents=True, exist_ok=True)
        for step in SPECTRA_STEPS:   # only a subset of steps
            wl = np.linspace(400, 700, 60)
            intensity = np.exp(-((wl - (500 + step * 20)) ** 2) / 1000)
            intensity += RNG.normal(0, 0.01, 60)
            pd.DataFrame({"wavelength": wl, "intensity": intensity}).to_csv(
                d / f"{step}.csv", index=False
            )
    print(f"  spectra: {len(SAMPLES)} samples × {len(SPECTRA_STEPS)} steps (SUBSET)")


def make_images() -> None:
    for sample_id in SAMPLES:
        d = DATA_DIR / sample_id / "images"
        d.mkdir(parents=True, exist_ok=True)
        for step in IMAGE_STEPS:    # only a subset of steps
            arr = (RNG.uniform(0, 1, (16, 16, 3)) * 255 * (step / 4)).astype(np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            img.save(d / f"{step}.png")
    print(f"  images: {len(SAMPLES)} samples × {len(IMAGE_STEPS)} steps (SUBSET)")


def make_timeseries() -> None:
    for sample_id in SAMPLES:
        d = DATA_DIR / sample_id / "timeseries"
        d.mkdir(parents=True, exist_ok=True)
        for step in ALL_STEPS:
            t = np.linspace(0, 1, 30)
            v = np.sin(2 * np.pi * t * step) + RNG.normal(0, 0.05, 30)
            pd.DataFrame({"time": t, "value": v}).to_csv(d / f"{step}.csv", index=False)
    print(f"  timeseries: {len(SAMPLES)} samples × {len(ALL_STEPS)} steps")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating synthetic data in {DATA_DIR} …")
    make_metadata()
    make_curves()
    make_spectra()
    make_images()
    make_timeseries()
    print("Done.")


if __name__ == "__main__":
    main()
