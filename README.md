Self-Healing Machines - Predictive Maintenance for a Connected Fleet

An AI-powered predictive maintenance platform that monitors industrial machine health,
detects abnormal operating conditions, predicts potential failures, estimates Remaining
Useful Life (RUL) and automatically generates preventive maintenance work orders before a
breakdown occurs.

The system is trained and evaluated on the **NASA C-MAPSS turbofan engine degradation
dataset** (FD001) and presented through a modern, dark-themed **Streamlit** industrial
monitoring dashboard.

---

## Features

| Module | What it does |
| --- | --- |
| **Dashboard overview** | Total / healthy / attention / critical machine counts and overall fleet health %. |
| **Machine monitoring** | Per-machine sensor readings (temperature, pressure, vibration, voltage, speed, flow), operating status and parameter trends. |
| **Anomaly detection** | Isolation Forest flags abnormal signatures, shows calibrated 0-100 severity scores and explains which channels deviated and by how much. |
| **RUL prediction** | Random Forest regression estimates remaining useful life with a 90% tree-based uncertainty band, degradation trends and near-failure alerts. |
| **Maintenance management** | Auto-generated work orders, manual dispatch, status tracking (pending / in progress / completed) and full maintenance history. |
| **Analytics & reports** | Sensor performance, fleet health trends, anomaly statistics, RUL analytics and downloadable CSV reports. |

---

## Machine learning

- **Isolation Forest** - anomaly detection. Trained on healthy early-life operating data;
  severity (0-100) is calibrated against the healthy-to-degraded decision-score
  distribution so it spreads realistically instead of saturating.
- **Random Forest Regression** - RUL prediction. Trained on a piece-wise linear, capped
  RUL target (cap = 125 cycles). Prediction intervals are derived from the spread of the
  forest's individual trees.
- **Rule-based decision system** - fuses RUL head-room and anomaly severity into a
  composite health score (0-100) and classifies each machine as **Normal**, **Warning** or
  **Critical**, always with a human-readable explanation and a recommended corrective
  action.

### Self-healing mechanism

After every prediction the decision engine:
1. evaluates machine health,
2. detects abnormal operating conditions,
3. creates a work order (Critical / High priority) when degradation is detected,
4. raises an alert with a recommended corrective action,
5. auto-resolves lingering preventive work orders once a machine returns to normal,
6. updates the machine's stored health status.

---

## Project structure

```
grand tata/
├── app.py                     # Streamlit entry point (sidebar navigation + fleet controls)
├── requirements.txt
├── README.md
├── src/
│   ├── config.py              # paths, sensor metadata, thresholds, theme palette
│   ├── data_loader.py         # C-MAPSS download + synthetic surrogate fallback
│   ├── preprocessing.py       # RUL target, feature selection, scaling, aux channels
│   ├── training.py            # training + evaluation pipeline
│   ├── inference.py           # end-to-end prediction facade
│   ├── database.py            # SQLite persistence layer
│   ├── simulation.py          # connected-fleet simulator
│   ├── self_healing.py        # self-healing decision engine
│   └── models/
│       ├── anomaly_detector.py
│       ├── rul_predictor.py
│       └── health_classifier.py
├── views/                     # dashboard modules
│   ├── common.py  overview.py  monitoring.py
│   ├── anomaly.py  rul.py  maintenance.py  analytics.py
└── artifacts/                 # trained model bundle + metadata (generated)
```

---

## Setup

```powershell
# 1. (optional) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install dependencies
python -m pip install -r requirements.txt

# 3. launch the dashboard
python -m streamlit run app.py
```

The application opens at <http://localhost:8501>.

On first launch the app automatically:

1. downloads the NASA C-MAPSS FD001 dataset (with an offline synthetic surrogate fallback
   if the network is unavailable),
2. trains the Isolation Forest and Random Forest models and caches them under
   `artifacts/`,
3. seeds the SQLite database (`fleet.db`) and warms up fleet history.

To train the models ahead of time (optional):

```powershell
python -m src.training
```

---

## Using the dashboard

- **Advance fleet** (sidebar) - streams the next N operating cycles into every machine and
  re-runs the prediction + self-healing pipeline.
- **Rebuild fleet** - resets the database and re-seeds a fresh fleet.
- **Retrain models** - retrains both models from scratch and reloads the bundle.

The sidebar also shows the signed-in user and their role. **Rebuild fleet** and
**Retrain models** are restricted to the *Administrator* role.

---

## Authentication

The dashboard is gated behind a login screen. Accounts are stored in their own SQLite
database (`auth.db`), separate from the operational database so that rebuilding the fleet
never removes them. Passwords are never stored in plain text - each account keeps a random
salt and a PBKDF2-HMAC-SHA256 hash (see `src/auth.py`).

Demo accounts seeded on first launch:

| Username | Password | Role |
| --- | --- | --- |
| `admin` | `admin123` | Administrator |
| `engineer` | `engineer123` | Maintenance engineer |
| `viewer` | `viewer123` | Viewer |

Change the seeds in `src/config.py`, or create accounts programmatically:

```python
from src.auth import get_auth_store
get_auth_store().create_user("ops_lead", "strong-password", "Operations lead", "Administrator")
```

### Sign up

The login screen also offers a **Create account** tab for self-service registration. New
users choose a *Maintenance engineer* or *Viewer* role; the **Administrator** role cannot
be self-assigned (it is provisioned via the API above). Registration validates the
username (3-20 characters, letters/digits/underscore, unique) and password (minimum 6
characters, confirmed), then signs the new user straight in.

To disable the login gate entirely (e.g. for an automated demo), set the environment
variable `SHM_AUTH=0` before launching.

---

All dashboard figures are produced from live model predictions on persisted sensor
readings - nothing is hardcoded. Vibration and supply voltage are deterministic,
documented transforms of the physical C-MAPSS signals (see
`src/preprocessing.py::add_auxiliary_sensors`) so the monitoring UI reflects a realistic
industrial instrument set.

---

## Model evaluation

The pipeline reports RUL regression metrics on held-out test units (RMSE, MAE, R2,
final-cycle RMSE and the asymmetric NASA C-MAPSS score) and anomaly detection rates on
healthy versus degraded operation. These are surfaced on the **Analytics & reports** page
and stored in `artifacts/model_metadata.json`.

Typical reference values for the bundled offline dataset:

| Metric | Value |
| --- | --- |
| RUL RMSE | ~11.6 cycles |
| RUL MAE | ~8.0 cycles |
| R2 | ~0.86 |
| Early-life anomaly rate | 0 % |
| Late-life anomaly rate | ~80 % |

---

## Database schema (SQLite)

`machines`, `sensor_readings`, `anomaly_results`, `rul_predictions`, `health_status`,
`work_orders`, `maintenance_history` and `alerts`. All records are written persistently and
read back dynamically by the dashboard.
