"""Model training pipeline.

Trains the anomaly detector and the RUL regressor, evaluates them on held-out data and
persists a single versioned artifact bundle that the dashboard loads at runtime.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.data_loader import DatasetBundle, load_dataset
from src.models.anomaly_detector import AnomalyDetector
from src.models.health_classifier import HealthClassifier
from src.models.rul_predictor import RULPredictor
from src.preprocessing import (
    Preprocessor,
    add_auxiliary_sensors,
    add_rul_target,
    get_informative_sensors,
    test_rul_per_cycle,
)

BUNDLE_PATH = os.path.join(config.ARTIFACTS_DIR, "model_bundle.joblib")
METADATA_PATH = os.path.join(config.ARTIFACTS_DIR, "model_metadata.json")


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The asymmetric C-MAPSS scoring function (penalises late predictions harder)."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def _healthy_mask(df: pd.DataFrame, fraction: float = config.HEALTHY_CYCLE_FRACTION) -> pd.Series:
    max_cycle = df.groupby("unit_number")["cycle"].transform("max")
    return df["cycle"] <= (max_cycle * fraction)


def train_models(bundle: DatasetBundle | None = None, verbose: bool = True) -> Dict[str, object]:
    """Full training routine. Returns an in-memory model bundle and evaluation metrics."""
    bundle = bundle or load_dataset()

    train = add_auxiliary_sensors(add_rul_target(bundle.train))
    test = add_auxiliary_sensors(bundle.test)
    # Ground-truth per-cycle RUL for the test units.
    valid_truth = bundle.rul[bundle.rul["unit_number"].isin(test["unit_number"].unique())]
    test = test_rul_per_cycle(test, valid_truth)

    sensors = get_informative_sensors(train)
    if verbose:
        print(f"[train] data source: {bundle.source} | subset: {bundle.subset}")
        print(f"[train] informative sensors ({len(sensors)}): {sensors}")

    preprocessor = Preprocessor().fit(train, sensors)

    X_train = preprocessor.transform(train)
    X_test = preprocessor.transform(test)

    # ---- Isolation Forest: fit on healthy early-life operating data, calibrate on the
    #      full healthy-to-degraded score distribution so severity spreads properly ----
    healthy_rows = _healthy_mask(train)
    X_healthy = preprocessor.transform(train[healthy_rows])
    anomaly_detector = AnomalyDetector(contamination=0.05).fit(
        X_healthy, preprocessor.features, X_reference=X_train, scaler=preprocessor.scaler
    )

    # ---- Random Forest: RUL regression ----
    rul_predictor = RULPredictor().fit(
        X_train, train["rul"].values, preprocessor.features, scaler=preprocessor.scaler
    )

    # ---- Evaluation ----
    y_pred = rul_predictor.predict(X_test)
    y_true = test["rul"].values
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Final-cycle (end of test trajectory) metrics - the classic C-MAPSS benchmark.
    last_idx = test.groupby("unit_number")["cycle"].idxmax()
    last_rows = test.loc[last_idx]
    y_pred_last = rul_predictor.predict(last_rows)
    y_true_last = last_rows["rul"].values
    rmse_last = float(np.sqrt(mean_squared_error(y_true_last, y_pred_last)))

    scored_test = anomaly_detector.predict_batch(test)
    healthy_mask = _healthy_mask(scored_test)

    def _safe_rate(mask) -> float:
        vals = scored_test.loc[mask, "is_anomaly"]
        return float(vals.mean()) if len(vals) else 0.0

    anomaly_stats = {
        "healthy_anomaly_rate": _safe_rate(healthy_mask),
        "late_life_anomaly_rate": _safe_rate(~healthy_mask),
    }

    health_classifier = HealthClassifier()

    metadata = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": bundle.source,
        "data_notes": bundle.notes,
        "subset": bundle.subset,
        "n_train_units": int(train["unit_number"].nunique()),
        "n_test_units": int(test["unit_number"].nunique()),
        "n_train_rows": int(len(train)),
        "sensors": sensors,
        "features": preprocessor.features,
        "metrics": {
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "r2": round(r2, 4),
            "rmse_final_cycle": round(rmse_last, 3),
            "nasa_score": round(nasa_score(y_true, y_pred), 2),
        },
        "anomaly_stats": {k: round(v, 4) for k, v in anomaly_stats.items()},
        "thresholds": config.THRESHOLDS,
        "rul_cap": config.RUL_CAP,
    }

    if verbose:
        print(f"[train] RUL metrics: RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.3f} "
              f"final-cycle RMSE={rmse_last:.2f}")
        print(f"[train] anomaly rates: {metadata['anomaly_stats']}")

    payload = {
        "preprocessor": preprocessor.to_dict(),
        "anomaly_detector": anomaly_detector.to_dict(),
        "rul_predictor": rul_predictor.to_dict(),
        "health_classifier": {"thresholds": health_classifier.thresholds,
                              "rul_cap": health_classifier.rul_cap},
        "metadata": metadata,
    }
    return payload


def save_bundle(payload: Dict[str, object]) -> None:
    """Persist the trained bundle to disk (atomic-ish: write temp then replace)."""
    tmp = BUNDLE_PATH + ".tmp"
    joblib.dump(payload, tmp)
    os.replace(tmp, BUNDLE_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload["metadata"], fh, indent=2)


def bundle_exists() -> bool:
    return os.path.exists(BUNDLE_PATH) and os.path.exists(METADATA_PATH)


def train_and_save(bundle: DatasetBundle | None = None, verbose: bool = True) -> Dict[str, object]:
    payload = train_models(bundle=bundle, verbose=verbose)
    save_bundle(payload)
    if verbose:
        print(f"[train] saved artifacts -> {BUNDLE_PATH}")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual entry point
    train_and_save()
