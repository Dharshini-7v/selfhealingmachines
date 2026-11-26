"""SQLite persistence layer.

Stores machines, sensor readings, anomaly results, RUL predictions, health status,
maintenance work orders, maintenance history and alerts. All access goes through the
:class:`Database` class which keeps a single connection (Streamlit-safe).
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterable, List, Optional

import pandas as pd

from src import config

SENSOR_COLS = list(config.SENSOR_COLUMNS) + list(config.AUX_SENSOR_META.keys())


def _now() -> str:
    """UTC timestamp in ISO-8601 (second precision)."""
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")


class Database:
    """Thin, thread-safe wrapper around a SQLite database file."""

    def __init__(self, path: str = config.DB_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

    # ------------------------------------------------------------------ helpers
    @contextmanager
    def cursor(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            finally:
                cur.close()

    def query_df(self, sql: str, params: Iterable | None = None) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query(sql, self._conn, params=tuple(params or ()))

    def execute(self, sql: str, params: Iterable | None = None) -> int:
        with self.cursor() as cur:
            cur.execute(sql, tuple(params or ()))
            return cur.lastrowid

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ schema
    def init_schema(self) -> None:
        sensor_defs = ",\n                ".join(f"{c} REAL" for c in SENSOR_COLS)
        ddl = f"""
        CREATE TABLE IF NOT EXISTS machines (
            machine_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_code    TEXT UNIQUE NOT NULL,
            name            TEXT NOT NULL,
            model           TEXT,
            location        TEXT,
            unit_number     INTEGER,
            commissioned_at TEXT,
            cycle_offset    INTEGER DEFAULT 0,
            status          TEXT DEFAULT 'Normal'
        );

        CREATE TABLE IF NOT EXISTS sensor_readings (
            reading_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id  INTEGER NOT NULL REFERENCES machines(machine_id) ON DELETE CASCADE,
            cycle       INTEGER NOT NULL,
            recorded_at TEXT NOT NULL,
            {sensor_defs}
        );
        CREATE INDEX IF NOT EXISTS idx_readings_machine ON sensor_readings(machine_id, cycle);

        CREATE TABLE IF NOT EXISTS anomaly_results (
            result_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id     INTEGER NOT NULL REFERENCES machines(machine_id) ON DELETE CASCADE,
            cycle          INTEGER NOT NULL,
            detected_at    TEXT NOT NULL,
            anomaly_score  REAL,
            severity       REAL,
            is_anomaly     INTEGER,
            explanation    TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_anomaly_machine ON anomaly_results(machine_id, cycle);

        CREATE TABLE IF NOT EXISTS rul_predictions (
            prediction_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id     INTEGER NOT NULL REFERENCES machines(machine_id) ON DELETE CASCADE,
            cycle          INTEGER NOT NULL,
            predicted_at   TEXT NOT NULL,
            predicted_rul  REAL,
            rul_lower      REAL,
            rul_upper      REAL
        );
        CREATE INDEX IF NOT EXISTS idx_rul_machine ON rul_predictions(machine_id, cycle);

        CREATE TABLE IF NOT EXISTS health_status (
            status_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id    INTEGER NOT NULL REFERENCES machines(machine_id) ON DELETE CASCADE,
            cycle         INTEGER NOT NULL,
            evaluated_at  TEXT NOT NULL,
            status        TEXT NOT NULL,
            health_score  REAL,
            explanation   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_health_machine ON health_status(machine_id, cycle);

        CREATE TABLE IF NOT EXISTS work_orders (
            work_order_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id       INTEGER NOT NULL REFERENCES machines(machine_id) ON DELETE CASCADE,
            created_at       TEXT NOT NULL,
            priority         TEXT NOT NULL,
            status           TEXT NOT NULL DEFAULT 'Pending',
            title            TEXT,
            description      TEXT,
            recommended_action TEXT,
            due_cycle        INTEGER,
            completed_at     TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_wo_machine ON work_orders(machine_id);

        CREATE TABLE IF NOT EXISTS maintenance_history (
            history_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            work_order_id  INTEGER REFERENCES work_orders(work_order_id) ON DELETE SET NULL,
            machine_id     INTEGER NOT NULL REFERENCES machines(machine_id) ON DELETE CASCADE,
            action_taken   TEXT,
            technician     TEXT,
            completed_at   TEXT NOT NULL,
            notes          TEXT
        );

        CREATE TABLE IF NOT EXISTS alerts (
            alert_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id  INTEGER NOT NULL REFERENCES machines(machine_id) ON DELETE CASCADE,
            created_at  TEXT NOT NULL,
            level       TEXT NOT NULL,
            message     TEXT NOT NULL,
            acknowledged INTEGER DEFAULT 0
        );
        """
        with self.cursor() as cur:
            cur.executescript(ddl)

    def is_empty(self) -> bool:
        row = self.query_df("SELECT COUNT(*) AS n FROM machines")
        return bool(row.empty or int(row["n"].iloc[0]) == 0)

    def reset(self) -> None:
        """Drop every table (used by the 'rebuild fleet' control)."""
        tables = [
            "alerts", "maintenance_history", "work_orders", "health_status",
            "rul_predictions", "anomaly_results", "sensor_readings", "machines",
        ]
        with self.cursor() as cur:
            for t in tables:
                cur.execute(f"DROP TABLE IF EXISTS {t}")
        self.init_schema()

    # ------------------------------------------------------------------ machines
    def seed_machines(self, rows: List[dict]) -> None:
        with self.cursor() as cur:
            cur.executemany(
                """INSERT OR IGNORE INTO machines
                   (machine_code, name, model, location, unit_number, commissioned_at, status)
                   VALUES (:machine_code, :name, :model, :location, :unit_number,
                           :commissioned_at, :status)""",
                rows,
            )

    def get_machines(self) -> pd.DataFrame:
        return self.query_df("SELECT * FROM machines ORDER BY unit_number")

    def machine_id_for_unit(self, unit_number: int) -> Optional[int]:
        df = self.query_df("SELECT machine_id FROM machines WHERE unit_number = ?", (unit_number,))
        return int(df["machine_id"].iloc[0]) if not df.empty else None

    def update_machine_status(self, machine_id: int, status: str) -> None:
        self.execute("UPDATE machines SET status = ? WHERE machine_id = ?", (status, machine_id))

    def set_cycle_offset(self, machine_id: int, offset: int) -> None:
        self.execute("UPDATE machines SET cycle_offset = ? WHERE machine_id = ?", (int(offset), machine_id))

    def max_reading_cycle(self, machine_id: int) -> int:
        df = self.query_df("SELECT COALESCE(MAX(cycle), 0) AS c FROM sensor_readings WHERE machine_id = ?",
                           (machine_id,))
        return int(df["c"].iloc[0]) if not df.empty else 0

    # ------------------------------------------------------------------ readings
    def insert_readings(self, machine_id: int, cycle: int, readings: dict) -> int:
        cols = ["machine_id", "cycle", "recorded_at"] + SENSOR_COLS
        values = [machine_id, cycle, _now()] + [float(readings.get(c, 0.0)) for c in SENSOR_COLS]
        placeholders = ", ".join(["?"] * len(cols))
        return self.execute(
            f"INSERT INTO sensor_readings ({', '.join(cols)}) VALUES ({placeholders})", values
        )

    def get_readings(self, machine_id: int, limit: int = 200) -> pd.DataFrame:
        return self.query_df(
            "SELECT * FROM sensor_readings WHERE machine_id = ? ORDER BY cycle DESC LIMIT ?",
            (machine_id, limit),
        ).sort_values("cycle")

    def latest_reading(self, machine_id: int) -> Optional[dict]:
        df = self.query_df(
            "SELECT * FROM sensor_readings WHERE machine_id = ? ORDER BY cycle DESC LIMIT 1",
            (machine_id,),
        )
        return df.iloc[0].to_dict() if not df.empty else None

    # ------------------------------------------------------------------ results
    def insert_anomaly(self, machine_id: int, cycle: int, score: float, severity: float,
                       is_anomaly: bool, explanation: str) -> int:
        return self.execute(
            """INSERT INTO anomaly_results
               (machine_id, cycle, detected_at, anomaly_score, severity, is_anomaly, explanation)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (machine_id, cycle, _now(), float(score), float(severity), int(is_anomaly), explanation),
        )

    def insert_rul(self, machine_id: int, cycle: int, rul: float, lower: float, upper: float) -> int:
        return self.execute(
            """INSERT INTO rul_predictions
               (machine_id, cycle, predicted_at, predicted_rul, rul_lower, rul_upper)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (machine_id, cycle, _now(), float(rul), float(lower), float(upper)),
        )

    def insert_health(self, machine_id: int, cycle: int, status: str, score: float, explanation: str) -> int:
        return self.execute(
            """INSERT INTO health_status
               (machine_id, cycle, evaluated_at, status, health_score, explanation)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (machine_id, cycle, _now(), status, float(score), explanation),
        )

    def latest_health(self) -> pd.DataFrame:
        """Latest health row per machine joined with machine metadata."""
        return self.query_df(
            """
            SELECT m.machine_id, m.machine_code, m.name, m.model, m.location, m.unit_number,
                   h.cycle, h.status, h.health_score, h.explanation, h.evaluated_at
            FROM machines m
            JOIN health_status h ON h.status_id = (
                SELECT status_id FROM health_status
                WHERE machine_id = m.machine_id
                ORDER BY cycle DESC, status_id DESC LIMIT 1
            )
            ORDER BY m.unit_number
            """
        )

    def health_history(self, machine_id: int) -> pd.DataFrame:
        return self.query_df(
            "SELECT * FROM health_status WHERE machine_id = ? ORDER BY cycle",
            (machine_id,),
        )

    def anomaly_history(self, machine_id: int, limit: int = 500) -> pd.DataFrame:
        return self.query_df(
            "SELECT * FROM anomaly_results WHERE machine_id = ? ORDER BY cycle DESC LIMIT ?",
            (machine_id, limit),
        ).sort_values("cycle")

    def rul_history(self, machine_id: int) -> pd.DataFrame:
        return self.query_df(
            "SELECT * FROM rul_predictions WHERE machine_id = ? ORDER BY cycle",
            (machine_id,),
        )

    # ------------------------------------------------------------------ work orders
    def open_work_order_exists(self, machine_id: int, title: str) -> bool:
        df = self.query_df(
            """SELECT COUNT(*) AS n FROM work_orders
               WHERE machine_id = ? AND title = ? AND status != 'Completed'""",
            (machine_id, title),
        )
        return bool(int(df["n"].iloc[0]) > 0)

    def create_work_order(self, machine_id: int, priority: str, title: str, description: str,
                          recommended_action: str, due_cycle: int | None = None) -> int:
        return self.execute(
            """INSERT INTO work_orders
               (machine_id, created_at, priority, status, title, description,
                recommended_action, due_cycle)
               VALUES (?, ?, ?, 'Pending', ?, ?, ?, ?)""",
            (machine_id, _now(), priority, title, description, recommended_action, due_cycle),
        )

    def get_work_orders(self, status: str | None = None) -> pd.DataFrame:
        base = """
            SELECT w.*, m.machine_code, m.name, m.location
            FROM work_orders w JOIN machines m ON m.machine_id = w.machine_id
        """
        if status:
            return self.query_df(base + " WHERE w.status = ? ORDER BY w.created_at DESC", (status,))
        return self.query_df(base + " ORDER BY w.created_at DESC")

    def update_work_order_status(self, work_order_id: int, status: str) -> None:
        completed = _now() if status == "Completed" else None
        self.execute(
            "UPDATE work_orders SET status = ?, completed_at = COALESCE(?, completed_at) WHERE work_order_id = ?",
            (status, completed, work_order_id),
        )

    def complete_work_order(self, work_order_id: int, technician: str, notes: str) -> None:
        wo = self.query_df("SELECT * FROM work_orders WHERE work_order_id = ?", (work_order_id,))
        if wo.empty:
            return
        machine_id = int(wo["machine_id"].iloc[0])
        action = wo["recommended_action"].iloc[0] if wo["recommended_action"].iloc[0] else wo["title"].iloc[0]
        self.update_work_order_status(work_order_id, "Completed")
        self.execute(
            """INSERT INTO maintenance_history
               (work_order_id, machine_id, action_taken, technician, completed_at, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (work_order_id, machine_id, action, technician, _now(), notes),
        )

    def get_maintenance_history(self, limit: int = 100) -> pd.DataFrame:
        return self.query_df(
            """SELECT h.*, m.machine_code, m.name
               FROM maintenance_history h JOIN machines m ON m.machine_id = h.machine_id
               ORDER BY h.completed_at DESC LIMIT ?""",
            (limit,),
        )

    # ------------------------------------------------------------------ alerts
    def add_alert(self, machine_id: int, level: str, message: str) -> int:
        return self.execute(
            "INSERT INTO alerts (machine_id, created_at, level, message) VALUES (?, ?, ?, ?)",
            (machine_id, _now(), level, message),
        )

    def get_alerts(self, limit: int = 100, unacknowledged_only: bool = False) -> pd.DataFrame:
        where = "WHERE a.acknowledged = 0" if unacknowledged_only else ""
        return self.query_df(
            f"""SELECT a.*, m.machine_code, m.name
                FROM alerts a JOIN machines m ON m.machine_id = a.machine_id
                {where} ORDER BY a.created_at DESC LIMIT ?""",
            (limit,),
        )

    def acknowledge_alert(self, alert_id: int) -> None:
        self.execute("UPDATE alerts SET acknowledged = 1 WHERE alert_id = ?", (alert_id,))

    # ------------------------------------------------------------------ summary
    def fleet_summary(self) -> dict:
        df = self.latest_health()
        total = len(self.get_machines())
        if df.empty:
            return {"total": total, "normal": 0, "warning": 0, "critical": 0,
                    "fleet_health": 0.0, "monitored": 0}
        counts = df["status"].value_counts().to_dict()
        monitored = len(df)
        fleet_health = round(float(df["health_score"].mean()), 1) if monitored else 0.0
        return {
            "total": total,
            "monitored": monitored,
            "normal": int(counts.get(config.STATUS_NORMAL, 0)),
            "warning": int(counts.get(config.STATUS_WARNING, 0)),
            "critical": int(counts.get(config.STATUS_CRITICAL, 0)),
            "fleet_health": fleet_health,
        }


# --------------------------------------------------------------------------------------
# Singleton accessor
# --------------------------------------------------------------------------------------
_DB_SINGLETON: Optional[Database] = None


def get_db(path: str | None = None) -> Database:
    global _DB_SINGLETON
    if _DB_SINGLETON is None:
        _DB_SINGLETON = Database(path or config.DB_PATH)
        _DB_SINGLETON.init_schema()
    return _DB_SINGLETON
