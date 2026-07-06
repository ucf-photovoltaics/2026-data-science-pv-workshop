"""Pipeline: orchestrates metadata → features → merge → model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rdstemplate.config import Config, ExtractorEntry
from rdstemplate.merge import build_feature_table, merge_feature_tables
from rdstemplate.metadata import load_metadata, exposure_steps_for_sample

logger = logging.getLogger(__name__)


def _import_all_extractors() -> None:
    """Ensure all built-in extractor modules are imported so they register."""
    import rdstemplate.features.curves  # noqa: F401
    import rdstemplate.features.images  # noqa: F401
    import rdstemplate.features.spectra  # noqa: F401
    import rdstemplate.features.timeseries  # noqa: F401


def _import_all_models() -> None:
    """Ensure all built-in model modules are imported so they register."""
    import rdstemplate.models.registry  # noqa: F401


class Pipeline:
    """Full data-science pipeline driven by a Config object.

    Usage::

        pipe = Pipeline(cfg)
        metadata = pipe.load_metadata()
        features = pipe.extract_features()
        df       = pipe.merge()
        results  = pipe.run_model()
    """

    def __init__(self, cfg: Config, source=None) -> None:
        """
        Parameters
        ----------
        cfg:    Validated Config object.
        source: Optional pre-built DataSource (e.g. injected from a Colab cell).
                If None, the source is built from cfg.data_source.
        """
        _import_all_extractors()
        _import_all_models()
        self.cfg = cfg
        self._metadata = None
        self._feature_tables = None
        self._tidy = None
        self._model_results = None

        if source is not None:
            self.source = source
        else:
            from rdstemplate.io.sources import source_from_config  # noqa: PLC0415
            self.source = source_from_config(cfg.data_source)

    # ------------------------------------------------------------------
    # Stage 1: metadata
    # ------------------------------------------------------------------

    def load_metadata(self) -> pd.DataFrame:
        """Load and validate sample metadata; cache on self._metadata."""
        self._metadata = load_metadata(self.cfg.metadata)
        logger.info(
            "Loaded metadata: %d samples, %d observations",
            self._metadata["sample_id"].nunique(),
            len(self._metadata),
        )
        return self._metadata

    @property
    def metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            self.load_metadata()
        return self._metadata  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Stage 2: feature extraction
    # ------------------------------------------------------------------

    def extract_features(self) -> dict[str, pd.DataFrame]:
        """Run all configured extractors; return modality → feature DataFrame.

        Each extractor is called once per (sample_id, exposure_step).
        Extractors are independent per observation — embarrassingly parallel.
        """
        from rdstemplate.features.base import EXTRACTOR_REGISTRY  # noqa: PLC0415

        modality_loaders = self._get_modality_loaders()
        tables: dict[str, pd.DataFrame] = {}

        extractor_groups = {
            "curves": self.cfg.extractors.curves,
            "spectra": self.cfg.extractors.spectra,
            "images": self.cfg.extractors.images,
            "timeseries": self.cfg.extractors.timeseries,
        }

        for modality, extractor_entries in extractor_groups.items():
            if not extractor_entries:
                continue
            records = self._extract_modality(
                modality, extractor_entries, modality_loaders[modality], EXTRACTOR_REGISTRY
            )
            if records:
                tables[modality] = build_feature_table(records, modality)

        self._feature_tables = tables
        return tables

    def _extract_modality(
        self,
        modality: str,
        entries: list[ExtractorEntry],
        loader_fn,
        registry: dict,
    ) -> list[dict]:
        """Return a list of per-(sample, step) dicts for one modality."""
        records: list[dict] = []

        for sample_id in self.metadata["sample_id"].unique():
            steps = exposure_steps_for_sample(self.metadata, sample_id)
            modality_dir = self.source.get(sample_id, modality)

            for step in steps:
                data = loader_fn(modality_dir, step)
                row: dict = {"sample_id": sample_id, "exposure_step": step}

                for entry in entries:
                    cls = registry.get(entry.name)
                    if cls is None:
                        logger.warning(
                            "Extractor '%s' not found in registry; skipping.", entry.name
                        )
                        continue
                    extractor = cls(**entry.params)
                    features = extractor.extract(sample_id, step, data)
                    row.update(features)

                records.append(row)

        return records

    def _get_modality_loaders(self) -> dict:
        from rdstemplate.io.loaders import (  # noqa: PLC0415
            load_curve,
            load_image,
            load_spectrum,
            load_timeseries,
        )

        return {
            "curves": load_curve,
            "spectra": load_spectrum,
            "images": load_image,
            "timeseries": load_timeseries,
        }

    # ------------------------------------------------------------------
    # Stage 3: merge
    # ------------------------------------------------------------------

    def merge(self) -> pd.DataFrame:
        """Outer-join all feature tables into one tidy (sample, exposure_step) dataframe."""
        if self._feature_tables is None:
            self.extract_features()
        self._tidy = merge_feature_tables(
            self._feature_tables,  # type: ignore[arg-type]
            self.metadata,
            gap_fill_policy=self.cfg.merge.gap_fill_policy,
        )
        logger.info(
            "Merged tidy dataframe: %d rows × %d cols",
            len(self._tidy),
            self._tidy.shape[1],
        )
        return self._tidy

    @property
    def tidy(self) -> pd.DataFrame:
        if self._tidy is None:
            self.merge()
        return self._tidy  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Stage 4: model
    # ------------------------------------------------------------------

    def run_model(self) -> dict[str, Any]:
        """Fit and evaluate the configured model on the tidy dataframe."""
        from rdstemplate.models.base import MODEL_REGISTRY  # noqa: PLC0415

        df = self.tidy
        target = self.cfg.model.target_col

        if target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found. Available: {list(df.columns)}"
            )

        feature_cols = [
            c for c in df.columns
            if c not in ("sample_id", "exposure_step", target)
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not feature_cols:
            raise ValueError("No numeric feature columns found after merge.")

        # Drop rows where target is missing.
        mask = df[target].notna()
        X = df.loc[mask, feature_cols].fillna(0)
        y = df.loc[mask, target]

        if len(X) == 0:
            raise ValueError("No rows with a non-null target value.")

        cls = MODEL_REGISTRY.get(self.cfg.model.name)
        if cls is None:
            raise ValueError(
                f"Model '{self.cfg.model.name}' not in registry. "
                f"Available: {list(MODEL_REGISTRY)}"
            )

        model = cls(**self.cfg.model.hyperparameters)
        model.fit(X, y)
        metrics = model.evaluate(X, y)

        self._model_results = {
            "model_name": self.cfg.model.name,
            "n_samples": len(X),
            "feature_cols": feature_cols,
            "metrics": metrics,
            "model": model,
        }

        logger.info("Model %s trained. Metrics: %s", self.cfg.model.name, metrics)
        return self._model_results

    # ------------------------------------------------------------------
    # Convenience: run full pipeline in one call
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run all four stages; return results dict."""
        self.load_metadata()
        self.extract_features()
        self.merge()
        self.run_model()
        return self._model_results  # type: ignore[return-value]
