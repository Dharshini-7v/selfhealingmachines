"""Isolation Forest based anomaly detection for machine sensor readings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src import config


@dataclass
class AnomalyResult:
    score: float                      # raw decision_function value (higher = normal)
    severity: float                   # 0-100, higher = more abnormal
    is_anomaly: bool
    explanation: str
    contributors: List[Dict[str, float]] = field(default_factory=list)


class AnomalyDetector:
    """Wraps :class:`sklearn.ensemble.IsolationForest` with explainability helpers."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42, n_estimators: int = 200):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.features: List[str] = []
        self.scaler: StandardScaler | None = None
        self.baseline_mean_: np.ndarray | None = None
        self.baseline_std_: np.ndarray | None = None
        # Severity calibration anchors derived from the reference score distribution.
        self.score_high_: float = 1.0   # ~healthy operating level
        self.score_low_: float = 0.0    # ~near-failure level
        self.fitted: bool = False

    # ------------------------------------------------------------------ training
    def fit(self, X: np.ndarray, features: List[str],
            X_reference: np.ndarray | None = None,
            scaler: StandardScaler | None = None,
            X_baseline_raw: np.ndarray | None = None) -> "AnomalyDetector":
        """Fit the Isolation Forest on *scaled* healthy data.

        ``X_reference`` is an optional (typically wider) set of scaled samples spanning
        healthy-to-degraded operation. Its decision-score distribution is used to calibrate
        the 0-100 severity scale, which keeps severity well spread instead of saturating.
        When omitted, the healthy data is used as the reference.

        ``scaler`` is the fitted StandardScaler used to produce ``X``. It is stored so that
        raw sensor frames can be transformed automatically at inference time. When not
        supplied it is derived from ``X_baseline_raw`` (if given) or left unset.
        """
        X = np.asarray(X, dtype=float)
        self.features = list(features)
        if scaler is not None:
            self.scaler = scaler
        elif X_baseline_raw is not None and X_baseline_raw.size:
            self.scaler = StandardScaler().fit(np.asarray(X_baseline_raw, dtype=float))
        self.model.fit(X)
        self.baseline_mean_ = X.mean(axis=0)
        self.baseline_std_ = np.where(X.std(axis=0) < 1e-9, 1.0, X.std(axis=0))

        ref = X if X_reference is None else np.asarray(X_reference, dtype=float)
        ref_scores = self.model.decision_function(ref)
        self.score_high_ = float(np.quantile(ref_scores, 0.95))   # healthy upper level
        self.score_low_ = float(np.quantile(ref_scores, 0.02))    # near-failure level
        self.fitted = True
        return self

    # ------------------------------------------------------------------ inference
    def _to_matrix(self, X) -> np.ndarray:
        """Return a scaled feature matrix.

        A ``pandas.DataFrame`` is treated as *raw* sensor data and is scaled with the
        stored scaler. A numpy array is assumed to be already scaled (the training path).
        """
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.features if c not in X.columns]
            if missing:
                raise ValueError(f"AnomalyDetector missing columns: {missing}")
            values = X[self.features].astype(float).values
            return self.scaler.transform(values) if self.scaler is not None else values
        return np.asarray(X, dtype=float)

    def score(self, X) -> np.ndarray:
        """Return the raw Isolation Forest decision score for each row."""
        return self.model.decision_function(self._to_matrix(X))

    def severity_from_score(self, score: float) -> float:
        """Map a decision score to a 0-100 severity using the calibrated reference range.

        The Isolation Forest decision function is *higher for normal* samples. A score at
        the healthy upper level maps to ~0 severity; a score at the near-failure level maps
        to full severity.
        """
        span = self.score_high_ - self.score_low_
        if abs(span) < 1e-9:
            return 0.0
        return 100.0 * float(np.clip((self.score_high_ - score) / span, 0.0, 1.0))

    def explain(self, row: pd.Series, top_k: int = 3) -> List[Dict[str, float]]:
        """Identify the sensors deviating most from the healthy baseline.

        Deviations are computed in the scaled feature space (matching how the model was
        trained) and reported with the original physical values.
        """
        if self.baseline_mean_ is None:
            return []
        raw = np.array([float(row.get(f, 0.0)) for f in self.features], dtype=float)
        scaled = self.scaler.transform(raw.reshape(1, -1))[0] if self.scaler is not None else raw
        z = (scaled - self.baseline_mean_) / self.baseline_std_
        order = np.argsort(-np.abs(z))[:top_k]
        contributors = []
        for idx in order:
            col = self.features[idx]
            meta = config.SENSOR_META.get(col, config.AUX_SENSOR_META.get(col, {}))
            direction = "above" if z[idx] > 0 else "below"
            contributors.append({
                "sensor": col,
                "name": meta.get("name", col),
                "unit": meta.get("unit", ""),
                "category": meta.get("category", "General"),
                "value": float(raw[idx]),
                "z_score": float(z[idx]),
                "direction": direction,
            })
        return contributors

    def predict_row(self, row: pd.Series) -> AnomalyResult:
        score = float(self.score(row.to_frame().T)[0])
        severity = self.severity_from_score(score)
        is_anomaly = severity >= config.THRESHOLDS["anomaly_severity"]
        contributors = self.explain(row) if is_anomaly else []

        if is_anomaly:
            parts = [
                f"{c['name']} {c['value']:.2f} {c['unit']} "
                f"({abs(c['z_score']):.1f}σ {c['direction']} normal)"
                for c in contributors[:2]
            ]
            explanation = (
                "Anomalous operating signature detected: " + "; ".join(parts)
                if parts else
                "Operating signature deviates from the learned healthy envelope."
            )
        else:
            explaining = self.explain(row, top_k=1)
            if explaining:
                c = explaining[0]
                explanation = (
                    f"Readings within normal operating envelope (largest deviation: "
                    f"{c['name']}, {abs(c['z_score']):.1f}σ)."
                )
            else:
                explanation = "Readings within normal operating envelope."
        return AnomalyResult(
            score=score,
            severity=severity,
            is_anomaly=is_anomaly,
            explanation=explanation,
            contributors=contributors,
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorised prediction returning score/severity/flag columns."""
        X = self._to_matrix(df)
        scores = self.model.decision_function(X)
        out = df.copy()
        out["anomaly_score"] = scores
        out["anomaly_severity"] = [self.severity_from_score(float(s)) for s in scores]
        out["is_anomaly"] = out["anomaly_severity"] >= config.THRESHOLDS["anomaly_severity"]
        return out

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "features": self.features,
            "scaler": self.scaler,
            "baseline_mean": self.baseline_mean_,
            "baseline_std": self.baseline_std_,
            "score_high": self.score_high_,
            "score_low": self.score_low_,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AnomalyDetector":
        obj = cls()
        obj.model = payload["model"]
        obj.features = payload["features"]
        obj.scaler = payload.get("scaler")
        obj.baseline_mean_ = payload["baseline_mean"]
        obj.baseline_std_ = payload["baseline_std"]
        obj.score_high_ = payload["score_high"]
        obj.score_low_ = payload["score_low"]
        obj.fitted = True
        return obj
