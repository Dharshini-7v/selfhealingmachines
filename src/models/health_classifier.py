"""Rule-based decision system that fuses anomaly and RUL signals into a machine health
classification (Normal / Warning / Critical) with a human-readable explanation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src import config


@dataclass
class HealthAssessment:
    status: str
    health_score: float                      # 0-100 (higher = healthier)
    rul_component: float
    anomaly_component: float
    explanation: str
    reasons: List[str] = field(default_factory=list)
    recommended_action: str = ""


class HealthClassifier:
    """Composite, fully explainable rule engine.

    The classifier combines two independent signals:
      * RUL head-room  -> how many cycles remain before predicted failure.
      * Anomaly severity -> how far current readings stray from the healthy envelope.
    """

    def __init__(self, thresholds: Dict[str, float] | None = None, rul_cap: int = config.RUL_CAP):
        self.thresholds = thresholds or config.THRESHOLDS
        self.rul_cap = rul_cap

    # ------------------------------------------------------------------ scoring
    def _rul_component(self, predicted_rul: float) -> float:
        return float(np.clip(predicted_rul / self.rul_cap, 0.0, 1.0) * 100.0)

    def _anomaly_component(self, severity: float) -> float:
        return float(np.clip(100.0 - severity, 0.0, 100.0))

    def classify(
        self,
        predicted_rul: float,
        anomaly_score: float,
        is_anomaly: bool,
        anomaly_severity: float,
        contributors: List[Dict[str, object]] | None = None,
    ) -> HealthAssessment:
        rul_component = self._rul_component(predicted_rul)
        anomaly_component = self._anomaly_component(anomaly_severity)

        # Composite health score: RUL head-room dominates, anomaly severity refines it.
        health_score = round(0.6 * rul_component + 0.4 * anomaly_component, 1)

        t = self.thresholds
        reasons: List[str] = []

        if predicted_rul <= t["critical_rul"]:
            reasons.append(
                f"Predicted remaining life is critical ({predicted_rul:.0f} cycles <= {t['critical_rul']:.0f})."
            )
        elif predicted_rul <= t["warning_rul"]:
            reasons.append(
                f"Predicted remaining life is limited ({predicted_rul:.0f} cycles <= {t['warning_rul']:.0f})."
            )

        if is_anomaly:
            reasons.append(
                f"Anomaly detector flagged an abnormal operating signature "
                f"(severity {anomaly_severity:.0f}/100)."
            )
        elif anomaly_severity >= 60:
            reasons.append(
                f"Elevated deviation from the healthy envelope (severity {anomaly_severity:.0f}/100)."
            )

        # ---- decision rules (ordered, first match wins) ----
        if (
            predicted_rul <= t["critical_rul"]
            or health_score < t["critical_score"]
            or (is_anomaly and predicted_rul <= 60)
        ):
            status = config.STATUS_CRITICAL
        elif (
            predicted_rul <= t["warning_rul"]
            or is_anomaly
            or health_score < t["warning_score"]
        ):
            status = config.STATUS_WARNING
        else:
            status = config.STATUS_NORMAL

        if status == config.STATUS_NORMAL:
            reasons = reasons or ["All sensor channels within the healthy operating envelope."]
            explanation = "Machine is operating normally - no maintenance action required."
        else:
            lead = {
                config.STATUS_WARNING: "Machine requires attention: ",
                config.STATUS_CRITICAL: "Immediate maintenance required: ",
            }[status]
            explanation = lead + " ".join(reasons)

        action = self._recommend_action(status, contributors or [])
        return HealthAssessment(
            status=status,
            health_score=health_score,
            rul_component=round(rul_component, 1),
            anomaly_component=round(anomaly_component, 1),
            explanation=explanation,
            reasons=reasons,
            recommended_action=action,
        )

    # ------------------------------------------------------------------ actions
    @staticmethod
    def _recommend_action(status: str, contributors: List[Dict[str, object]]) -> str:
        if status == config.STATUS_NORMAL:
            return "Continue routine monitoring."

        # Prioritise the corrective action for the most deviating sensor category.
        categories: List[str] = []
        for c in contributors:
            cat = str(c.get("category", "General"))
            if cat not in categories:
                categories.append(cat)

        actions = [config.ACTION_PLAYBOOK.get(cat, config.ACTION_PLAYBOOK["General"]) for cat in categories[:2]]
        if not actions:
            actions = [config.ACTION_PLAYBOOK["General"]]

        if status == config.STATUS_CRITICAL:
            prefix = "Schedule immediate corrective maintenance within the next maintenance window. "
        else:
            prefix = "Plan preventive maintenance during the next planned downtime. "
        return prefix + " ".join(actions)
