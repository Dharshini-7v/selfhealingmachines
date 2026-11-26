"""Preprocessing utilities: RUL target construction, feature selection, scaling and
engineering of auxiliary industrial channels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config


# --------------------------------------------------------------------------------------
# Remaining Useful Life target
# --------------------------------------------------------------------------------------
def add_rul_target(df: pd.DataFrame, cap: int = config.RUL_CAP) -> pd.DataFrame:
    """Add a piece-wise linear ``rul`` column to a run-to-failure training frame.

    RUL is the number of cycles remaining before failure for every row. It is capped at
    ``cap`` because near-term behaviour is what matters operationally (standard C-MAPSS
    practice) and it stabilises regression training.
    """
    out = df.copy()
    max_cycle = out.groupby("unit_number")["cycle"].transform("max")
    rul = max_cycle - out["cycle"]
    out["rul"] = np.clip(rul, 0, cap)
    return out


def test_rul_per_cycle(test: pd.DataFrame, rul_truth: pd.DataFrame, cap: int = config.RUL_CAP) -> pd.DataFrame:
    """Build a per-cycle RUL target for the test units using the supplied truth table."""
    truth = dict(zip(rul_truth["unit_number"], rul_truth["rul"]))
    out = test.copy()
    max_cycle = out.groupby("unit_number")["cycle"].transform("max")
    remaining_after = (max_cycle - out["cycle"])
    base = out["unit_number"].map(truth).fillna(0)
    out["rul"] = np.clip(base + remaining_after, 0, cap)
    return out


# --------------------------------------------------------------------------------------
# Auxiliary industrial channels
# --------------------------------------------------------------------------------------
def add_auxiliary_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Derive vibration and voltage channels from the physical C-MAPSS signals.

    These are deterministic, documented, row-wise transforms of real sensor data (so the
    numbers are reproducible and explainable), used to present a realistic instrument set
    in the monitoring UI. Being row-wise, they work identically for a single reading or a
    whole fleet batch.
    """
    out = df.copy()

    def _col(name: str, default: float):
        return out[name] if name in out.columns else default

    core_speed = _col("s9", 9060.0)
    corr_core = _col("s14", 8135.0)
    bleed = _col("s17", 393.0)

    # Positive core-speed deviation and corrected-speed deviation raise vibration energy.
    core_dev = np.clip(core_speed / 9060.0 - 1.0, 0.0, None)
    corr_dev = np.abs(corr_core / 8135.0 - 1.0)
    out["vibration"] = (0.7 + 40.0 * core_dev + 25.0 * corr_dev).clip(lower=0.2, upper=8.0)

    # Bleed-enthalpy head is used as the indicator of the regulated supply voltage.
    bleed_dev = bleed / 393.0 - 1.0
    out["voltage"] = (380.0 - 120.0 * bleed_dev).clip(lower=300.0, upper=460.0)
    return out


# --------------------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------------------
def get_informative_sensors(df: pd.DataFrame, sensor_columns: Optional[List[str]] = None,
                            min_std_ratio: float = 1e-4) -> List[str]:
    """Return sensors whose variance is meaningfully non-zero across the fleet."""
    sensor_columns = sensor_columns or config.SENSOR_COLUMNS
    informative = []
    for col in sensor_columns:
        if col not in df.columns:
            continue
        values = df[col].astype(float)
        scale = max(abs(values.mean()), 1e-6)
        if values.std() / scale > min_std_ratio and values.std() > 1e-6:
            informative.append(col)
    return informative or list(config.DEFAULT_INFORMATIVE_SENSORS)


def add_rolling_features(df: pd.DataFrame, sensors: List[str], window: int = 5) -> pd.DataFrame:
    """Add a short rolling-mean per sensor to smooth sensor noise per unit."""
    out = df.sort_values(["unit_number", "cycle"]).copy()
    for col in sensors:
        out[f"{col}_roll"] = (
            out.groupby("unit_number")[col]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
    return out


def feature_columns(sensors: List[str], with_rolling: bool = False) -> List[str]:
    cols = list(sensors)
    if with_rolling:
        cols += [f"{s}_roll" for s in sensors]
    return cols

# --------------------------------------------------------------------------------------
@dataclass
class Preprocessor:
    """Fitted preprocessing state shared by the models at inference time."""

    sensors: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    scaler: Optional[StandardScaler] = None

    def fit(self, df: pd.DataFrame, sensors: List[str]) -> "Preprocessor":
        self.sensors = list(sensors)
        self.features = feature_columns(self.sensors)
        self.scaler = StandardScaler().fit(df[self.features].astype(float).values)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Preprocessor must be fitted (or loaded) before transform().")
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns for inference: {missing}")
        return self.scaler.transform(df[self.features].astype(float).values)

    def to_dict(self) -> dict:
        return {"sensors": self.sensors, "features": self.features, "scaler": self.scaler}

    @classmethod
    def from_dict(cls, payload: dict) -> "Preprocessor":
        return cls(
            sensors=payload["sensors"],
            features=payload["features"],
            scaler=payload["scaler"],
        )
