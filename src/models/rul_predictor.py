"""Random Forest Regression model for Remaining Useful Life (RUL) prediction."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src import config


class RULPredictor:
    """Predicts the remaining useful life (in cycles) of a machine component."""

    def __init__(self, n_estimators: int = 300, random_state: int = 42, max_depth: int | None = 18):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        self.features: List[str] = []
        self.scaler: StandardScaler | None = None
        self.fitted: bool = False

    # ------------------------------------------------------------------ training
    def fit(self, X: np.ndarray, y: np.ndarray, features: List[str],
            scaler: StandardScaler | None = None) -> "RULPredictor":
        self.features = list(features)
        self.scaler = scaler if scaler is not None else self.scaler
        self.model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        self.fitted = True
        return self

    # ------------------------------------------------------------------ inference
    def _to_matrix(self, X) -> np.ndarray:
        """Return a scaled feature matrix.

        A ``pandas.DataFrame`` is treated as *raw* sensor data and scaled with the stored
        scaler; a numpy array is assumed to be already scaled (the training path).
        """
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.features if c not in X.columns]
            if missing:
                raise ValueError(f"RULPredictor missing columns: {missing}")
            values = X[self.features].astype(float).values
            return self.scaler.transform(values) if self.scaler is not None else values
        return np.asarray(X, dtype=float)

    def predict(self, X) -> np.ndarray:
        """Point predictions clipped to the valid RUL range."""
        preds = self.model.predict(self._to_matrix(X))
        return np.clip(preds, 0, config.RUL_CAP)

    def predict_with_interval(self, X, confidence: float = 0.9) -> Dict[str, np.ndarray]:
        """Estimate a prediction interval from the spread across the forest's trees.

        Random Forests expose individual tree predictions, which give a practical,
        model-derived uncertainty band without extra calibration data.
        """
        matrix = self._to_matrix(X)
        tree_preds = np.stack([tree.predict(matrix) for tree in self.model.estimators_])
        lower_q = (1.0 - confidence) / 2.0
        upper_q = 1.0 - lower_q
        point = np.clip(self.model.predict(matrix), 0, config.RUL_CAP)
        lower = np.clip(np.quantile(tree_preds, lower_q, axis=0), 0, config.RUL_CAP)
        upper = np.clip(np.quantile(tree_preds, upper_q, axis=0), 0, config.RUL_CAP)
        return {"prediction": point, "lower": lower, "upper": upper}

    def predict_row(self, row: pd.Series, confidence: float = 0.9) -> Dict[str, float]:
        frame = row.to_frame().T
        result = self.predict_with_interval(frame, confidence=confidence)
        return {
            "predicted_rul": float(result["prediction"][0]),
            "lower": float(result["lower"][0]),
            "upper": float(result["upper"][0]),
        }

    def feature_importances(self) -> pd.DataFrame:
        if not self.fitted:
            return pd.DataFrame(columns=["feature", "importance"])
        return (
            pd.DataFrame({"feature": self.features, "importance": self.model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def to_dict(self) -> dict:
        return {"model": self.model, "features": self.features, "scaler": self.scaler}

    @classmethod
    def from_dict(cls, payload: dict) -> "RULPredictor":
        obj = cls()
        obj.model = payload["model"]
        obj.features = payload["features"]
        obj.scaler = payload.get("scaler")
        obj.fitted = True
        return obj
