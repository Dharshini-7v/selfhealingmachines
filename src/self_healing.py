"""Self-healing decision engine.

Continuously evaluates machine health, generates maintenance alerts, creates work orders
and recommends corrective actions. It also auto-resolves pending preventive work orders
when a machine recovers, closing the preventive-maintenance loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src import config
from src.database import Database
from src.inference import PredictionResult


@dataclass
class HealingOutcome:
    machine_id: int
    machine_code: str
    status: str
    actions: List[str] = field(default_factory=list)
    work_order_id: Optional[int] = None
    alert_level: Optional[str] = None


class SelfHealingEngine:
    """Rule-driven automation layer that turns predictions into maintenance workflows."""

    def __init__(self, thresholds: Dict[str, float] | None = None):
        self.thresholds = thresholds or config.THRESHOLDS

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _primary_category(result: PredictionResult) -> str:
        if result.contributors:
            return str(result.contributors[0].get("category", "General"))
        return "General"

    def process(self, db: Database, machine: dict, result: PredictionResult, cycle: int) -> HealingOutcome:
        """Evaluate a single machine's prediction and act on it."""
        machine_id = int(machine["machine_id"])
        machine_code = machine["machine_code"]
        outcome = HealingOutcome(machine_id=machine_id, machine_code=machine_code, status=result.status)

        # Persist the updated machine status (self-healing status sync).
        db.update_machine_status(machine_id, result.status)

        if result.status == config.STATUS_CRITICAL:
            category = self._primary_category(result)
            title = f"Critical: {category} degradation"
            due_cycle = cycle + max(int(result.predicted_rul), 0)
            if not db.open_work_order_exists(machine_id, title):
                wo_id = db.create_work_order(
                    machine_id=machine_id,
                    priority="Critical",
                    title=title,
                    description=result.explanation,
                    recommended_action=result.recommended_action,
                    due_cycle=due_cycle,
                )
                outcome.work_order_id = wo_id
                outcome.actions.append(f"Created critical work order #{wo_id}")
            db.add_alert(
                machine_id, "Critical",
                f"{machine_code}: {result.explanation} Action: {result.recommended_action}",
            )
            outcome.alert_level = "Critical"
            outcome.actions.append("Raised critical alert")

        elif result.status == config.STATUS_WARNING:
            category = self._primary_category(result)
            title = f"Preventive: {category} maintenance"
            due_cycle = cycle + max(int(result.predicted_rul), 0)
            if not db.open_work_order_exists(machine_id, title):
                wo_id = db.create_work_order(
                    machine_id=machine_id,
                    priority="High",
                    title=title,
                    description=result.explanation,
                    recommended_action=result.recommended_action,
                    due_cycle=due_cycle,
                )
                outcome.work_order_id = wo_id
                outcome.actions.append(f"Created preventive work order #{wo_id}")
                db.add_alert(machine_id, "Warning",
                             f"{machine_code}: {result.explanation}")
                outcome.alert_level = "Warning"

        else:  # Normal
            # Auto-resolve lingering preventive orders once the machine is clearly healthy.
            if result.health_score >= 85:
                open_wos = db.query_df(
                    """SELECT work_order_id, title FROM work_orders
                       WHERE machine_id = ? AND status != 'Completed' AND priority IN ('Low', 'Medium')""",
                    (machine_id,),
                )
                for _, wo in open_wos.iterrows():
                    db.complete_work_order(int(wo["work_order_id"]), technician="auto-heal",
                                           notes="Auto-resolved: machine returned to normal operating envelope.")
                    outcome.actions.append(f"Auto-resolved work order #{int(wo['work_order_id'])}")

        return outcome

    # ------------------------------------------------------------------ bulk
    def evaluate_fleet(self, db: Database, predictions: Dict[int, PredictionResult],
                       cycle_lookup: Dict[int, int]) -> List[HealingOutcome]:
        """Run the engine across many machines at once."""
        machines = db.get_machines()
        outcomes: List[HealingOutcome] = []
        for _, machine in machines.iterrows():
            mid = int(machine["machine_id"])
            if mid in predictions:
                outcomes.append(
                    self.process(db, machine.to_dict(), predictions[mid], cycle_lookup.get(mid, 0))
                )
        return outcomes
