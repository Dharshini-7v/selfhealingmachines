"""Dashboard overview: fleet-level KPIs, status distribution and priority machines."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from views import common


def render(db, bundle, sim, meta) -> None:
    common.page_header(
        "Fleet monitoring overview",
        "Real-time health of the connected fleet, powered by Isolation Forest anomaly "
        "detection and Random Forest RUL regression.",
    )

    summary = db.fleet_summary()
    latest = db.latest_health()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        common.kpi_card("Total machines monitored", f"{summary['total']}", "Connected units",
                        config.PALETTE["accent"])
    with c2:
        common.kpi_card("Healthy", f"{summary['normal']}", f"{_pct(summary['normal'], summary['total'])} of fleet",
                        config.PALETTE["normal"])
    with c3:
        common.kpi_card("Requiring attention", f"{summary['warning']}",
                        f"{_pct(summary['warning'], summary['total'])} of fleet",
                        config.PALETTE["warning"])
    with c4:
        common.kpi_card("Critical", f"{summary['critical']}", "Immediate maintenance",
                        config.PALETTE["critical"])
    with c5:
        common.kpi_card("Overall fleet health", f"{summary['fleet_health']}%",
                        "Mean composite health score", config.PALETTE["info"])

    st.write("")
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("##### Fleet health distribution")
        fig = go.Figure(go.Pie(
            labels=["Normal", "Warning", "Critical"],
            values=[summary["normal"], summary["warning"], summary["critical"]],
            hole=0.62,
            marker=dict(colors=[config.PALETTE["normal"], config.PALETTE["warning"],
                                config.PALETTE["critical"]]),
            textinfo="label+percent",
        ))
        fig.add_annotation(text=f"{summary['monitored']}<br><span style='font-size:11px'>assessed</span>",
                           showarrow=False, font=dict(size=20, color=config.PALETTE["text"]))
        st.plotly_chart(common.style_fig(fig, 320), width="stretch")

    with right:
        st.markdown("##### Machines requiring attention")
        if latest.empty:
            st.info("No health assessments recorded yet. Run a simulation step.")
        else:
            attention = latest[latest["status"] != config.STATUS_NORMAL].copy()
            attention = attention.sort_values("health_score")
            if attention.empty:
                st.success("All machines are operating within normal parameters.")
            else:
                show = attention.head(10)[["machine_code", "location", "cycle",
                                           "status", "health_score"]]
                show = show.rename(columns={
                    "machine_code": "machine", "location": "location", "cycle": "cycle",
                    "status": "status", "health_score": "health",
                })
                st.dataframe(show, width="stretch", hide_index=True, height=300)

    st.write("")
    st.markdown("##### Recent maintenance alerts")
    alerts = db.get_alerts(limit=8, unacknowledged_only=True)
    if alerts.empty:
        st.info("No active alerts.")
    else:
        view = alerts[["created_at", "machine_code", "level", "message"]].rename(columns={
            "created_at": "raised at", "machine_code": "machine", "level": "level",
            "message": "message"})
        st.dataframe(view, width="stretch", hide_index=True)


def _pct(n: int, total: int) -> str:
    return f"{(100.0 * n / total):.0f}%" if total else "0%"
