"""Built-in model wrappers registered under config-addressable names."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rdstemplate.models.base import ModelWrapper, register_model


@register_model("random_forest_regressor")
class RandomForestRegressorWrapper(ModelWrapper):
    """scikit-learn RandomForestRegressor wrapped to the ModelWrapper interface."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestRegressorWrapper":
        from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415

        self._model = RandomForestRegressor(**self.hyperparameters)
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


@register_model("linear_regression")
class LinearRegressionWrapper(ModelWrapper):
    """scikit-learn LinearRegression wrapped to the ModelWrapper interface."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionWrapper":
        from sklearn.linear_model import LinearRegression  # noqa: PLC0415

        self._model = LinearRegression(**self.hyperparameters)
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)
