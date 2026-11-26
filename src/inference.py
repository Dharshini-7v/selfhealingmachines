"""Runtime inference façade.

Loads the persisted model bundle once and exposes a single :func:`predict_machine`
entry point that produces a complete, explained health assessment for one sensor snapshot.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import joblib
import pandas as pd

from src import config
from src.models.anomaly_detector import AnomalyDetector
from src.models.health_classifier import HealthAssessment, HealthClassifier
from src.models.rul_predictor import RULPredictor
from src.preprocessing import Preprocessor, add_auxiliary_sensors


@dataclass
class PredictionResult:
    status: str
    health_score: float
    anomaly_score: float
    anomaly_severity: float
    is_anomaly: bool
    anomaly_explanation: str
    predicted_rul: float
    rul_lower: float
    rul_upper: float
    explanation: str
    recommended_action: str
    contributors: list
    reasons: list

    def as_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


class ModelBundle:
    """Holds the fitted models and exposes the end-to-end prediction pipeline."""

    def __init__(self, preprocessor: Preprocessor, anomaly_detector: AnomalyDetector,
                 rul_predictor: RULPredictor, health_classifier: HealthClassifier,
                 metadata: Dict[str, object]):
        self.preprocessor = preprocessor
        self.anomaly_detector = anomaly_detector
        self.rul_predictor = rul_predictor
        self.health_classifier = health_classifier
        self.metadata = metadata

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | None = None) -> "ModelBundle":
        path = path or os.path.join(config.ARTIFACTS_DIR, "model_bundle.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Trained model bundle not found. Run `python -m src.training` (or the app's "
                "auto-bootstrap) to train the models first."
            )
        payload = joblib.load(path)
        return cls(
            preprocessor=Preprocessor.from_dict(payload["preprocessor"]),
            anomaly_detector=AnomalyDetector.from_dict(payload["anomaly_detector"]),
            rul_predictor=RULPredictor.from_dict(payload["rul_predictor"]),
            health_classifier=HealthClassifier(
                thresholds=payload["health_classifier"]["thresholds"],
                rul_cap=payload["health_classifier"]["rul_cap"],
            ),
            metadata=payload.get("metadata", {}),
        )

    # ------------------------------------------------------------------ inference
    def predict_machine(self, readings: Dict[str, float]) -> PredictionResult:
        """Predict health for a single machine given a sensor snapshot.

        ``readings`` must contain the raw sensor channels (s1..s21). Auxiliary channels
        are derived automatically if missing.
        """
        row = pd.DataFrame([readings])
        row = add_auxiliary_sensors(row).iloc[0]

        # Anomaly detection
        anomaly = self.anomaly_detector.predict_row(row)

        # RUL prediction (with tree-based uncertainty band)
        rul = self.rul_predictor.predict_row(row)

        # Rule-based health classification
        assessment: HealthAssessment = self.health_classifier.classify(
            predicted_rul=rul["predicted_rul"],
            anomaly_score=anomaly.score,
            is_anomaly=anomaly.is_anomaly,
            anomaly_severity=anomaly.severity,
            contributors=anomaly.contributors,
        )

        return PredictionResult(
            status=assessment.status,
            health_score=assessment.health_score,
            anomaly_score=float(anomaly.score),
            anomaly_severity=float(anomaly.severity),
            is_anomaly=bool(anomaly.is_anomaly),
            anomaly_explanation=anomaly.explanation,
            predicted_rul=float(rul["predicted_rul"]),
            rul_lower=float(rul["lower"]),
            rul_upper=float(rul["upper"]),
            explanation=assessment.explanation,
            recommended_action=assessment.recommended_action,
            contributors=anomaly.contributors,
            reasons=assessment.reasons,
        )


def load_metadata() -> Dict[str, object]:
    """Load the model metadata JSON (metrics, sensors, training timestamp)."""
    if os.path.exists(os.path.join(config.ARTIFACTS_DIR, "model_metadata.json")):
        with open(os.path.join(config.ARTIFACTS_DIR, "model_metadata.json"), "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


_cached_bundle: Optional[ModelBundle] = None


def get_bundle(force_reload: bool = False) -> ModelBundle:
    """Process-wide cached accessor for the model bundle."""
    global _cached_bundle
    if _cached_bundle is None or force_reload:
        _cached_bundle = ModelBundle.load()
    return _cached_bundle
