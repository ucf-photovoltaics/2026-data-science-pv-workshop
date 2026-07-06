"""Model wrapper ABC and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Class decorator that registers a ModelWrapper under *name*."""

    def deco(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return deco


class ModelWrapper(ABC):
    """Thin wrapper giving a uniform fit/predict/evaluate interface."""

    def __init__(self, **hyperparameters) -> None:
        self.hyperparameters = hyperparameters
        self._model: Any = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelWrapper":
        """Fit the model; return self."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions."""

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Return a dict of evaluation metrics."""
        from sklearn.metrics import mean_absolute_error, r2_score  # noqa: PLC0415

        preds = self.predict(X)
        return {
            "r2": float(r2_score(y, preds)),
            "mae": float(mean_absolute_error(y, preds)),
        }
