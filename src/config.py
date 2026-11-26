"""Central configuration for the Self-Healing Machines predictive maintenance platform.

This module holds every shared constant: filesystem paths, dataset column layout,
sensor metadata (display names, units, physical categories), health thresholds and
the look-and-feel palette used by the dashboard.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------------------
# Filesystem paths
# --------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DB_PATH = os.path.join(PROJECT_ROOT, "fleet.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --------------------------------------------------------------------------------------
# Dataset configuration (NASA C-MAPSS turbofan degradation dataset)
# --------------------------------------------------------------------------------------
# FD001 - 100 units, single operating condition, single fault mode (HPC degradation).
CMAPSS_SUBSET = os.environ.get("CMAPSS_SUBSET", "FD001")
CMAPSS_MIRRORS = [
    "https://raw.githubusercontent.com/biswajitsahoo1111/CMAPSS-Data-Analysis/master/CMAPSSData",
    "https://raw.githubusercontent.com/ojwalch/CMAPSS/master/CMAPSSData",
]

INDEX_COLUMNS = ["unit_number", "cycle"]
SETTING_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
DATA_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Sensors that are physically relevant in FD001 (some are near-constant and dropped).
# The preprocessing module re-computes this list dynamically from the data, this is a
# sensible default used when the dataset is not yet available.
DEFAULT_INFORMATIVE_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
    "s13", "s14", "s15", "s17", "s20", "s21",
]

# --------------------------------------------------------------------------------------
# Sensor metadata - maps raw C-MAPSS channels to descriptive industrial parameters
# --------------------------------------------------------------------------------------
# category is one of: Temperature, Pressure, Speed, Flow, Ratio
SENSOR_META = {
    "s1":  {"name": "Fan inlet temperature",        "unit": "°R",      "category": "Temperature"},
    "s2":  {"name": "LPC outlet temperature",       "unit": "°R",      "category": "Temperature"},
    "s3":  {"name": "HPC outlet temperature",       "unit": "°R",      "category": "Temperature"},
    "s4":  {"name": "LPT outlet temperature",       "unit": "°R",      "category": "Temperature"},
    "s5":  {"name": "Fan inlet pressure",           "unit": "psia",    "category": "Pressure"},
    "s6":  {"name": "Bypass duct pressure",         "unit": "psia",    "category": "Pressure"},
    "s7":  {"name": "HPC outlet pressure",          "unit": "psia",    "category": "Pressure"},
    "s8":  {"name": "Physical fan speed",           "unit": "rpm",     "category": "Speed"},
    "s9":  {"name": "Physical core speed",          "unit": "rpm",     "category": "Speed"},
    "s10": {"name": "Engine pressure ratio",        "unit": "-",       "category": "Ratio"},
    "s11": {"name": "HPC static pressure",          "unit": "psia",    "category": "Pressure"},
    "s12": {"name": "Fuel flow / static pressure",  "unit": "pps/psia","category": "Flow"},
    "s13": {"name": "Corrected fan speed",          "unit": "rpm",     "category": "Speed"},
    "s14": {"name": "Corrected core speed",         "unit": "rpm",     "category": "Speed"},
    "s15": {"name": "Bypass ratio",                 "unit": "-",       "category": "Ratio"},
    "s16": {"name": "Burner fuel-air ratio",        "unit": "-",       "category": "Ratio"},
    "s17": {"name": "Bleed enthalpy",               "unit": "-",       "category": "Flow"},
    "s18": {"name": "Demanded fan speed",           "unit": "rpm",     "category": "Speed"},
    "s19": {"name": "Demanded corrected fan speed", "unit": "rpm",     "category": "Speed"},
    "s20": {"name": "HPT coolant bleed",            "unit": "lbm/s",   "category": "Flow"},
    "s21": {"name": "LPT coolant bleed",            "unit": "lbm/s",   "category": "Flow"},
}

# The requirement asks for temperature / pressure / vibration / voltage style parameters.
# C-MAPSS provides temperature, pressure, speed and flow channels. Two auxiliary
# industrial channels (vibration, voltage) are derived deterministically from the
# physical sensor signals so the monitoring UI reflects a realistic instrument set.
AUX_SENSOR_META = {
    "vibration": {"name": "Vibration level", "unit": "g", "category": "Vibration"},
    "voltage":   {"name": "Supply voltage",  "unit": "V", "category": "Electrical"},
}

ALL_SENSOR_META = {**SENSOR_META, **AUX_SENSOR_META}

# Human readable ordering for the UI.
CATEGORY_ORDER = ["Temperature", "Pressure", "Vibration", "Speed", "Flow", "Ratio", "Electrical"]

# --------------------------------------------------------------------------------------
# Modelling configuration
# --------------------------------------------------------------------------------------
RUL_CAP = 125                 # classic piece-wise linear RUL target cap for C-MAPSS
HEALTHY_CYCLE_FRACTION = 0.30 # first 30% of each engine's life is treated as "healthy"

# Health thresholds - rule based decision system
THRESHOLDS = {
    "critical_rul": 30,       # cycles  -> immediate maintenance
    "warning_rul": 70,        # cycles  -> schedule maintenance
    "critical_score": 35.0,   # composite health score (0-100)
    "warning_score": 70.0,
    "anomaly_severity": 55.0, # severity (0-100) above which readings count as anomalous
}

# Public health status labels
STATUS_NORMAL = "Normal"
STATUS_WARNING = "Warning"
STATUS_CRITICAL = "Critical"
STATUS_ORDER = [STATUS_NORMAL, STATUS_WARNING, STATUS_CRITICAL]

# --------------------------------------------------------------------------------------
# Maintenance configuration
# --------------------------------------------------------------------------------------
WORK_ORDER_STATUS = ["Pending", "In Progress", "Completed"]
WORK_ORDER_PRIORITY = ["Low", "Medium", "High", "Critical"]

# Corrective action playbook keyed by physical sensor category.
ACTION_PLAYBOOK = {
    "Temperature": "Inspect cooling / thermal management path and verify sensor calibration.",
    "Pressure":    "Check for leaks, blockages or compressor fouling; verify pressure transducers.",
    "Vibration":   "Inspect bearings and rotor balance; schedule alignment and lubrication.",
    "Speed":       "Review rotor dynamics and speed control loop; inspect shaft and couplings.",
    "Flow":        "Validate fuel / coolant flow paths and clean filters and nozzles.",
    "Ratio":       "Audit control ratios and recalibrate the associated control loop.",
    "Electrical":  "Inspect power supply, wiring harness and voltage regulation.",
    "General":     "Perform a full diagnostic inspection and follow the maintenance checklist.",
}

# --------------------------------------------------------------------------------------
# Dashboard look and feel (dark industrial theme)
# --------------------------------------------------------------------------------------
PALETTE = {
    "bg": "#0e1117",
    "panel": "#161b27",
    "panel_alt": "#1c2333",
    "border": "#2a3243",
    "text": "#e6e9ef",
    "muted": "#8b93a7",
    "accent": "#3b82f6",
    "normal": "#22c55e",
    "warning": "#f59e0b",
    "critical": "#ef4444",
    "info": "#38bdf8",
}

STATUS_COLORS = {
    STATUS_NORMAL: PALETTE["normal"],
    STATUS_WARNING: PALETTE["warning"],
    STATUS_CRITICAL: PALETTE["critical"],
}

# Fleet-level simulation defaults
DEFAULT_FLEET_SIZE = 100
SIM_STEP_CYCLES = 5           # how many cycles each simulation tick advances
AUTO_REFRESH_SECONDS = 8

# --------------------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------------------
# The login gate can be disabled by setting the environment variable SHM_AUTH=0 (useful
# for automated demos and tests). The user store is kept in its own SQLite file so that
# rebuilding the fleet (which resets the operational database) never removes accounts.
AUTH_ENABLED = os.environ.get("SHM_AUTH", "1").lower() not in {"0", "false", "no"}

AUTH_DB_PATH = os.path.join(PROJECT_ROOT, "auth.db")

ROLE_ADMIN = "Administrator"

# Accounts seeded on first launch so the app is immediately usable. Passwords are stored
# only as salted PBKDF2 hashes - see src/auth.py.
DEFAULT_USERS = [
    {"username": "admin", "password": "admin123",
     "full_name": "Fleet administrator", "role": ROLE_ADMIN},
    {"username": "engineer", "password": "engineer123",
     "full_name": "Maintenance engineer", "role": "Maintenance engineer"},
    {"username": "viewer", "password": "viewer123",
     "full_name": "Shift operator", "role": "Viewer"},
]
