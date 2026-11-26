"""Connected-fleet simulator.

Each machine in the fleet is backed by one run-to-failure trajectory from the C-MAPSS
dataset. The simulator streams that trajectory cycle by cycle into the database, then
runs the full prediction + self-healing pipeline on the latest reading of every machine.
A "simulation tick" therefore advances the whole fleet and refreshes its health state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src import config
from src.data_loader import DatasetBundle, load_dataset
from src.database import Database
from src.inference import ModelBundle
from src.preprocessing import add_auxiliary_sensors
from src.self_healing import HealingOutcome, SelfHealingEngine

PLANT_AREAS = [
    "Plant A - Assembly line 1",
    "Plant A - Assembly line 2",
    "Plant B - Compressor hall",
    "Plant B - Turbine hall",
    "Plant C - Power block",
]
MACHINE_MODEL = "C-MAPSS F100-PW-100 turbofan"
EVAL_STRIDE = 10  # evaluate health every N cycles when warming up history


@dataclass
class StepSummary:
    evaluated: int = 0
    work_orders_created: int = 0
    alerts_raised: int = 0
    failed: int = 0
    statuses: Dict[str, int] = None
    current_cycle_max: int = 0

    def as_dict(self) -> dict:
        return {
            "evaluated": self.evaluated,
            "work_orders_created": self.work_orders_created,
            "alerts_raised": self.alerts_raised,
            "failed": self.failed,
            "statuses": self.statuses or {},
            "current_cycle_max": self.current_cycle_max,
        }


class FleetSimulator:
    def __init__(self, db: Database, bundle: ModelBundle,
                 dataset: Optional[DatasetBundle] = None,
                 fleet_size: int = config.DEFAULT_FLEET_SIZE):
        self.db = db
        self.bundle = bundle
        self.dataset = dataset or load_dataset()
        self.fleet_size = min(fleet_size, int(self.dataset.train["unit_number"].nunique()))
        self.healer = SelfHealingEngine()
        # Derive the auxiliary industrial channels once so they are persisted with the
        # readings (and shown in the monitoring UI), not only used transiently at inference.
        train = add_auxiliary_sensors(self.dataset.train)
        self._trajectories: Dict[int, pd.DataFrame] = {
            unit: grp.sort_values("cycle").reset_index(drop=True)
            for unit, grp in train.groupby("unit_number")
        }

    # ------------------------------------------------------------------ lifecycle
    def _start_index(self, unit: int, lifetime: int) -> int:
        """Deterministic mid-life start so the initial fleet shows a realistic mix."""
        fraction = ((unit * 37) % 89) / 100.0  # 0.00 - 0.88
        return int(max(5, min(lifetime - 3, round(lifetime * fraction))))

    def initialize(self, force: bool = False) -> None:
        """Seed the machines table and warm up history for the fleet."""
        if not force and not self.db.is_empty():
            return

        rows = []
        for unit in range(1, self.fleet_size + 1):
            rows.append({
                "machine_code": f"TF-{unit:03d}",
                "name": f"Turbofan engine TF-{unit:03d}",
                "model": MACHINE_MODEL,
                "location": PLANT_AREAS[(unit - 1) % len(PLANT_AREAS)],
                "unit_number": unit,
                "commissioned_at": "2023-01-01",
                "status": config.STATUS_NORMAL,
            })
        self.db.seed_machines(rows)
        if force:
            self._warm_up()
        elif self.db.query_df("SELECT COUNT(*) AS n FROM sensor_readings")["n"].iloc[0] == 0:
            self._warm_up()

    def _warm_up(self) -> None:
        """Insert full sensor history up to the start cycle and evaluate sampled points."""
        for unit in range(1, self.fleet_size + 1):
            traj = self._trajectories.get(unit)
            if traj is None:
                continue
            machine_id = self.db.machine_id_for_unit(unit)
            if machine_id is None:
                continue

            lifetime = len(traj)
            start = self._start_index(unit, lifetime)
            self.db.set_cycle_offset(machine_id, 0)

            history = traj.iloc[:start]
            for _, r in history.iterrows():
                self.db.insert_readings(machine_id, int(r["cycle"]), r.to_dict())

            # Evaluate health at sampled cycles to build trend history for analytics.
            sample_cycles = list(range(EVAL_STRIDE, start + 1, EVAL_STRIDE))
            if not sample_cycles or sample_cycles[-1] != start:
                sample_cycles.append(start)

            latest_result = None
            for cyc in sample_cycles:
                row = traj.iloc[cyc - 1]
                result = self.bundle.predict_machine(row.to_dict())
                self.db.insert_anomaly(machine_id, cyc, result.anomaly_score,
                                       result.anomaly_severity, result.is_anomaly,
                                       result.anomaly_explanation)
                self.db.insert_rul(machine_id, cyc, result.predicted_rul,
                                   result.rul_lower, result.rul_upper)
                self.db.insert_health(machine_id, cyc, result.status,
                                      result.health_score, result.explanation)
                latest_result = result

            # Run self-healing only for the current (latest) state.
            machine = self.db.get_machines().set_index("unit_number").loc[unit].to_dict()
            machine["machine_id"] = machine_id
            if latest_result is not None:
                self.healer.process(self.db, machine, latest_result, start)

    # ------------------------------------------------------------------ stepping
    def step(self, cycles: int = config.SIM_STEP_CYCLES) -> StepSummary:
        """Advance every machine by ``cycles`` and refresh predictions + self-healing."""
        summary = StepSummary(statuses={s: 0 for s in config.STATUS_ORDER})
        machines = self.db.get_machines()
        predictions: Dict[int, object] = {}
        cycle_lookup: Dict[int, int] = {}

        for _, machine in machines.iterrows():
            unit = int(machine["unit_number"])
            machine_id = int(machine["machine_id"])
            traj = self._trajectories.get(unit)
            if traj is None:
                continue

            offset = int(machine["cycle_offset"] or 0)
            lifetime = len(traj)
            age = self.db.max_reading_cycle(machine_id)
            idx = age - offset

            advanced = False
            last_row = None
            # Insert new sensor readings for the requested horizon.
            for _ in range(cycles):
                next_idx = idx + 1
                if next_idx > lifetime:
                    break  # end of life reached
                next_age = offset + next_idx
                last_row = traj.iloc[next_idx - 1]
                self.db.insert_readings(machine_id, next_age, last_row.to_dict())
                idx = next_idx
                age = next_age
                advanced = True

            if not advanced:
                summary.failed += 1
                continue

            # Predict on the latest reading.
            result = self.bundle.predict_machine(last_row.to_dict())
            self.db.insert_anomaly(machine_id, age, result.anomaly_score,
                                   result.anomaly_severity, result.is_anomaly,
                                   result.anomaly_explanation)
            self.db.insert_rul(machine_id, age, result.predicted_rul,
                               result.rul_lower, result.rul_upper)
            self.db.insert_health(machine_id, age, result.status,
                                  result.health_score, result.explanation)

            predictions[machine_id] = result
            cycle_lookup[machine_id] = age
            summary.evaluated += 1
            summary.statuses[result.status] = summary.statuses.get(result.status, 0) + 1
            summary.current_cycle_max = max(summary.current_cycle_max, age)

        # Self-healing automation across the whole fleet.
        outcomes: List[HealingOutcome] = self.healer.evaluate_fleet(self.db, predictions, cycle_lookup)
        for o in outcomes:
            summary.work_orders_created += sum(1 for a in o.actions if "Created" in a)
            summary.alerts_raised += 1 if o.alert_level else 0

        return summary

    # ------------------------------------------------------------------ maintenance
    def replace_component(self, machine_id: int) -> None:
        """Reset a machine's wear clock after a component replacement (self-healing)."""
        current_age = self.db.max_reading_cycle(machine_id)
        self.db.set_cycle_offset(machine_id, current_age)
        self.db.update_machine_status(machine_id, config.STATUS_NORMAL)
        self.db.add_alert(machine_id, "Info",
                          "Component replacement performed - degradation clock reset.")

    def simultaneous_step(self, cycles: int = config.SIM_STEP_CYCLES) -> dict:
        return self.step(cycles).as_dict()
