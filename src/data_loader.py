"""Dataset access for the NASA C-MAPSS turbofan degradation dataset.

The platform prefers the *real* NASA C-MAPSS dataset (FD001 by default). When the
dataset cannot be downloaded (offline environment / restricted network) a physically
plausible synthetic surrogate with the exact same schema is generated so that the whole
application remains fully functional. The provenance of the data (``real`` vs
``synthetic``) is always reported to the UI so that nothing is silently faked.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src import config


@dataclass
class DatasetBundle:
    """Container holding a train/test split in the C-MAPSS layout."""

    train: pd.DataFrame
    test: pd.DataFrame
    rul: pd.DataFrame          # columns: unit_number, rul  (ground-truth for test units)
    subset: str
    source: str                # "real" or "synthetic"
    notes: str = ""


# --------------------------------------------------------------------------------------
# Real dataset handling
# --------------------------------------------------------------------------------------
def _local_dataset_paths(subset: str) -> Dict[str, str]:
    base = config.DATA_DIR
    return {
        "train": os.path.join(base, f"train_{subset}.txt"),
        "test": os.path.join(base, f"test_{subset}.txt"),
        "rul": os.path.join(base, f"RUL_{subset}.txt"),
    }


def _read_cmapss_file(path: str, columns: list[str]) -> pd.DataFrame:
    """C-MAPSS text files are whitespace separated with a trailing empty column."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")
    # Some releases carry an index column; keep the first len(columns) columns.
    df = df.iloc[:, : len(columns)]
    df.columns = columns
    return df


def _try_download(subset: str) -> bool:
    """Attempt to download the real dataset from the configured mirrors."""
    try:
        import requests  # local import so the app can run without network deps
    except Exception:
        return False

    paths = _local_dataset_paths(subset)
    filenames = {
        "train": f"train_{subset}.txt",
        "test": f"test_{subset}.txt",
        "rul": f"RUL_{subset}.txt",
    }
    for mirror in config.CMAPSS_MIRRORS:
        try:
            ok = True
            for key, fname in filenames.items():
                url = f"{mirror}/{fname}"
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200 or not resp.content:
                    ok = False
                    break
                with open(paths[key], "wb") as fh:
                    fh.write(resp.content)
            if ok:
                return True
        except Exception:
            continue
    return False


def load_real_dataset(subset: str = config.CMAPSS_SUBSET, allow_download: bool = True) -> Optional[DatasetBundle]:
    """Load the real C-MAPSS subset, downloading it on first use if necessary."""
    paths = _local_dataset_paths(subset)
    if not all(os.path.exists(p) for p in paths.values()):
        if not allow_download or not _try_download(subset):
            return None
    try:
        train = _read_cmapss_file(paths["train"], config.DATA_COLUMNS)
        test = _read_cmapss_file(paths["test"], config.DATA_COLUMNS)
        rul = pd.read_csv(paths["rul"], sep=r"\s+", header=None, engine="python")
        rul = rul.dropna(axis=1, how="all").iloc[:, :1]
        rul.columns = ["rul"]
        rul.insert(0, "unit_number", np.arange(1, len(rul) + 1))
        return DatasetBundle(
            train=train,
            test=test,
            rul=rul,
            subset=subset,
            source="real",
            notes="NASA C-MAPSS turbofan degradation dataset.",
        )
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Synthetic surrogate generator
# --------------------------------------------------------------------------------------
def _generate_unit(unit_id: int, rng: np.random.Generator, condition: int = 0) -> pd.DataFrame:
    """Generate a single engine run-to-failure trajectory with degradation drift.

    Values are calibrated to the statistical footprint of real C-MAPSS FD001 data:
    each engine runs for a random lifetime, sensors are roughly constant while healthy
    and drift monotonically as the unit approaches failure.
    """
    lifetime = int(np.clip(rng.normal(200, 45), 130, 360))

    # Healthy baselines per sensor (mean, std) approximated from real FD001 statistics.
    baselines = {
        "s1": (518.7, 0.5), "s2": (642.7, 0.5), "s3": (1590.0, 6.0), "s4": (1408.0, 9.0),
        "s5": (14.62, 0.02), "s6": (21.61, 0.02), "s7": (554.4, 0.5), "s8": (2388.1, 2.0),
        "s9": (9060.0, 15.0), "s10": (1.30, 0.005), "s11": (47.5, 0.3), "s12": (521.4, 2.0),
        "s13": (2388.0, 2.0), "s14": (8135.0, 12.0), "s15": (8.44, 0.02), "s16": (0.03, 0.001),
        "s17": (393.0, 1.5), "s18": (2388.0, 0.5), "s19": (100.0, 0.1), "s20": (38.9, 0.3),
        "s21": (23.3, 0.2),
    }
    # Channels that are effectively constant in C-MAPSS FD001 (line conditions).
    flat_sensors = {"s1", "s5", "s16", "s18", "s19"}
    # Sensors that drift upward (+) or downward (-) with wear and their drift magnitude
    # as a fraction of the baseline value at end-of-life.
    drift = {
        "s2": +0.010, "s3": +0.012, "s4": +0.013, "s7": +0.0025, "s8": -0.008,
        "s9": +0.010, "s11": +0.030, "s12": +0.022, "s13": -0.008, "s14": +0.012,
        "s15": +0.180, "s17": -0.035, "s20": -0.090, "s21": -0.130,
    }

    cycles = np.arange(1, lifetime + 1)
    progress = cycles / lifetime                       # 0 .. 1 wear progress
    accel = progress ** 2                              # non-linear, accelerates late

    data = {"unit_number": unit_id, "cycle": cycles}
    for c in config.SETTING_COLUMNS:
        data[c] = np.full(lifetime, 0.0 if c != "op_setting_3" else 100.0)
    for sensor, (mean, std) in baselines.items():
        if sensor in flat_sensors or sensor not in drift:
            # Line-condition channels: effectively constant (no signal, no drift).
            data[sensor] = np.full(lifetime, mean, dtype=float)
            continue
        # Degrading channels carry measurement noise proportional to the baseline.
        signal = np.full(lifetime, mean, dtype=float) * (1.0 + drift[sensor] * accel)
        signal = signal + rng.normal(0.0, abs(mean) * 0.0012, lifetime)
        data[sensor] = signal

    return pd.DataFrame(data)


def generate_synthetic_dataset(
    subset: str = config.CMAPSS_SUBSET,
    n_units: int = int(config.DEFAULT_FLEET_SIZE * 1.4),
    seed: int = 42,
) -> DatasetBundle:
    """Create a synthetic C-MAPSS-compatible bundle (train + test + RUL ground truth).

    Follows the real dataset protocol: the train split contains complete run-to-failure
    trajectories, while the test split contains *truncated* trajectories whose
    ground-truth RUL (cycles remaining after the last observed cycle) is supplied
    separately.
    """
    rng = np.random.default_rng(seed)
    n_units = max(20, n_units)
    n_test = max(10, n_units // 4)
    n_train = n_units - n_test

    # Complete run-to-failure trajectories.
    full_trajectories = [_generate_unit(uid, rng) for uid in range(1, n_units + 1)]

    train = pd.concat(full_trajectories[:n_train], ignore_index=True)

    test_rows = []
    rul_rows = []
    for new_id, full in enumerate(full_trajectories[n_train:], start=1):
        tail = int(rng.integers(20, 46))              # cycles held back (true RUL)
        cut = max(20, len(full) - tail)
        segment = full.iloc[:cut].copy()
        segment["unit_number"] = new_id               # renumber so it matches RUL truth
        test_rows.append(segment)
        rul_rows.append({"unit_number": new_id, "rul": len(full) - cut})

    test = pd.concat(test_rows, ignore_index=True)
    rul = pd.DataFrame(rul_rows)

    return DatasetBundle(
        train=train,
        test=test,
        rul=rul,
        subset=subset,
        source="synthetic",
        notes=(
            "Offline synthetic surrogate generated with the same schema and statistical "
            "footprint as NASA C-MAPSS FD001 (train = run-to-failure, test = truncated)."
        ),
    )


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def load_dataset(
    subset: str = config.CMAPSS_SUBSET,
    allow_download: bool = True,
    prefer_real: bool = True,
) -> DatasetBundle:
    """Return the best available dataset bundle.

    Tries the real NASA dataset first (download or cached), then falls back to the
    synthetic surrogate. Never raises for missing data.
    """
    if prefer_real:
        bundle = load_real_dataset(subset, allow_download=allow_download)
        if bundle is not None:
            return bundle
    return generate_synthetic_dataset(subset=subset)


def dataset_status() -> Dict[str, object]:
    """Describe what data is currently available without loading everything."""
    paths = _local_dataset_paths(config.CMAPSS_SUBSET)
    cached = all(os.path.exists(p) for p in paths.values())
    files = {k: os.path.basename(v) for k, v in paths.items()}
    return {"subset": config.CMAPSS_SUBSET, "cached_real_dataset": cached, "files": files}
