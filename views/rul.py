"""RUL prediction module: remaining life per machine, degradation trends and near-failure alerts."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src import config
from views import common


def _latest_rul(db):
    """Latest RUL prediction per machine, closest to failure first."""
    return db.query_df(
        """
        SELECT m.machine_id, m.machine_code, m.name, m.location, m.unit_number,
               r.cycle, r.predicted_rul, r.rul_lower, r.rul_upper, r.predicted_at
        FROM machines m
        JOIN rul_predictions r ON r.prediction_id = (
            SELECT prediction_id FROM rul_predictions
            WHERE machine_id = m.machine_id
            ORDER BY cycle DESC, prediction_id DESC LIMIT 1
        )
        ORDER BY r.predicted_rul ASC
        """
    )


def render(db, bundle, sim, meta) -> None:
    common.page_header(
        "RUL prediction",
        "Random Forest regression estimates remaining useful life, with tree-based "
        "uncertainty bands derived from the ensemble spread.",
    )

    df = _latest_rul(db)
    if df.empty:
        st.info("No RUL predictions recorded yet. Run a simulation step from the sidebar.")
        return

    t = config.THRESHOLDS
    near = df[df["predicted_rul"] <= t["critical_rul"]]
    warn = df[(df["predicted_rul"] > t["critical_rul"]) & (df["predicted_rul"] <= t["warning_rul"])]
    healthy = len(df) - len(near) - len(warn)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        common.kpi_card("Mean predicted RUL", f"{df['predicted_rul'].mean():.0f}",
                        "cycles remaining", config.PALETTE["accent"])
    with c2:
        common.kpi_card("Near failure", f"{len(near)}", f"<= {t['critical_rul']} cycles",
                        config.PALETTE["critical"])
    with c3:
        common.kpi_card("Limited life", f"{len(warn)}", f"<= {t['warning_rul']} cycles",
                        config.PALETTE["warning"])
    with c4:
        common.kpi_card("Healthy headroom", f"{healthy}", "comfortable margin",
                        config.PALETTE["normal"])

    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("##### Fleet RUL distribution")
        fig = go.Figure(go.Histogram(x=df["predicted_rul"], nbinsx=20,
                                     marker=dict(color=config.PALETTE["info"])))
        fig.add_vline(x=t["critical_rul"], line_dash="dash", line_color=config.PALETTE["critical"])
        fig.add_vline(x=t["warning_rul"], line_dash="dash", line_color=config.PALETTE["warning"])
        fig.update_layout(xaxis_title="predicted RUL (cycles)", yaxis_title="machines",
                          showlegend=False)
        st.plotly_chart(common.style_fig(fig, 300), width="stretch")

    with right:
        st.markdown("##### Machines closest to failure")
        head = df.head(10).copy()
        fig = go.Figure(go.Bar(
            x=head["predicted_rul"], y=head["machine_code"], orientation="h",
            error_x=dict(type="data", symmetric=False,
                         array=(head["rul_upper"] - head["predicted_rul"]).clip(lower=0),
                         arrayminus=(head["predicted_rul"] - head["rul_lower"]).clip(lower=0)),
            marker=dict(color=[_bar_color(v) for v in head["predicted_rul"]]),
        ))
        fig.update_layout(xaxis_title="predicted RUL (cycles)", yaxis=dict(autorange="reversed"),
                          showlegend=False)
        st.plotly_chart(common.style_fig(fig, 300), width="stretch")

    st.write("")
    st.markdown("##### Machine degradation trajectory")
    labels = {r["machine_code"]: r for _, r in df.iterrows()}
    code = st.selectbox("Select machine", list(labels.keys()))
    machine_id = int(labels[code]["machine_id"])
    hist = db.rul_history(machine_id)

    if hist.empty:
        st.info("No RUL history for this machine yet.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["rul_upper"], mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["rul_lower"], mode="lines",
                                 fill="tonexty", fillcolor="rgba(56,189,248,0.15)",
                                 line=dict(width=0), name="90% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["predicted_rul"], mode="lines+markers",
                                 name="predicted RUL",
                                 line=dict(color=config.PALETTE["info"], width=2)))
        fig.add_hline(y=t["critical_rul"], line_dash="dash", line_color=config.PALETTE["critical"])
        fig.add_hline(y=t["warning_rul"], line_dash="dash", line_color=config.PALETTE["warning"])
        fig.update_layout(xaxis_title="operating cycle",
                          yaxis_title="remaining useful life (cycles)", hovermode="x unified")
        st.plotly_chart(common.style_fig(fig, 380), width="stretch")
        st.caption(
            "The shaded band is the 90% prediction interval estimated from the Random "
            "Forest's individual tree predictions. Dashed lines mark the warning and "
            "critical RUL thresholds."
        )

    st.write("")
    st.markdown("##### Fleet RUL table")
    show = df.copy()
    show["predicted RUL"] = show["predicted_rul"].round(0)
    show["90% interval (cycles)"] = show.apply(
        lambda r: f"{r['rul_lower']:.0f} - {r['rul_upper']:.0f}", axis=1)
    st.dataframe(
        show[["machine_code", "location", "cycle", "predicted RUL", "90% interval (cycles)"]]
        .rename(columns={"machine_code": "machine"}),
        width="stretch", hide_index=True, height=340,
    )


def _bar_color(value: float) -> str:
    if value <= config.THRESHOLDS["critical_rul"]:
        return config.PALETTE["critical"]
    if value <= config.THRESHOLDS["warning_rul"]:
        return config.PALETTE["warning"]
    return config.PALETTE["normal"]
