"""Individual machine monitoring: live sensor readings, trends and operating status."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from views import common


def render(db, bundle, sim, meta) -> None:
    common.page_header(
        "Machine monitoring",
        "Inspect a single unit's sensor readings and operating status.",
    )

    machines = db.get_machines()
    if machines.empty:
        st.warning("No machines registered. Run a simulation step from the sidebar.")
        return

    labels = {row["machine_code"]: row for _, row in machines.iterrows()}
    col_a, col_b = st.columns([1.2, 2.8])
    with col_a:
        code = st.selectbox("Select machine", list(labels.keys()))
    machine = labels[code]
    machine_id = int(machine["machine_id"])

    latest_reading = db.latest_reading(machine_id)
    health = db.latest_health()
    hrow = health[health["machine_id"] == machine_id]
    rul_hist = db.rul_history(machine_id)
    anom_hist = db.anomaly_history(machine_id)

    if latest_reading is None:
        st.info("No readings available for this machine yet.")
        return

    status = hrow["status"].iloc[0] if not hrow.empty else machine["status"]
    health_score = float(hrow["health_score"].iloc[0]) if not hrow.empty else 0.0
    explanation = hrow["explanation"].iloc[0] if not hrow.empty else ""
    pred_rul = float(rul_hist["predicted_rul"].iloc[-1]) if not rul_hist.empty else float("nan")
    severity = float(anom_hist["severity"].iloc[-1]) if not anom_hist.empty else 0.0

    with col_b:
        st.markdown(
            f"""<div class="card">
                <h4>{machine['name']} &nbsp; {common.status_pill(status)}</h4>
                <div class="muted">{machine['model']} &nbsp;|&nbsp; {machine['location']}
                &nbsp;|&nbsp; operating cycle {int(latest_reading['cycle'])}</div>
                <div style="margin-top:8px">{explanation}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        common.kpi_card("Health score", f"{health_score:.0f}%", "Composite (RUL + anomaly)",
                        _accent(status))
    with k2:
        common.kpi_card("Predicted RUL", "n/a" if pd.isna(pred_rul) else f"{pred_rul:.0f}",
                        "cycles remaining", config.PALETTE["info"])
    with k3:
        common.kpi_card("Anomaly severity", f"{severity:.0f}/100", "Deviation from envelope",
                        config.PALETTE["warning"])
    with k4:
        common.kpi_card("Channels online", f"{len(config.ALL_SENSOR_META)}", "Sensor parameters",
                        config.PALETTE["accent"])

    st.write("")
    tab_readings, tab_trends = st.tabs(["Sensor readings", "Parameter trends"])

    with tab_readings:
        st.markdown("##### Current sensor snapshot")
        snap = common.sensor_table(latest_reading)
        st.dataframe(snap[["parameter", "value", "unit", "category"]],
                     width="stretch", hide_index=True, height=430)

    with tab_trends:
        st.markdown("##### Sensor history")
        readings = db.get_readings(machine_id, limit=400)
        if readings.empty:
            st.info("No history recorded yet.")
            return
        options = [f"{v['name']} ({k})" for k, v in config.ALL_SENSOR_META.items() if k in readings.columns]
        default = [o for o in options if "(s3)" in o or "(s9)" in o or "(vibration)" in o][:3] or options[:3]
        chosen = st.multiselect("Parameters to plot", options, default=default)
        keys = [o.split("(")[-1].rstrip(")") for o in chosen]
        if keys:
            fig = go.Figure()
            for k in keys:
                meta_k = config.ALL_SENSOR_META[k]
                fig.add_trace(go.Scatter(
                    x=readings["cycle"], y=readings[k], mode="lines",
                    name=f"{meta_k['name']} [{meta_k['unit']}]",
                ))
            fig.update_layout(xaxis_title="operating cycle", yaxis_title="value",
                              hovermode="x unified")
            st.plotly_chart(common.style_fig(fig, 420), width="stretch")


def _accent(status: str) -> str:
    return config.STATUS_COLORS.get(status, config.PALETTE["accent"])
