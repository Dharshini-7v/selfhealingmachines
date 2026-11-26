"""Anomaly detection module: flagged machines, abnormal readings, scores and explanations."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.preprocessing import add_auxiliary_sensors
from views import common


def _latest_anomalies(db) -> pd.DataFrame:
    """Latest Isolation Forest assessment per machine, most abnormal first."""
    return db.query_df(
        """
        SELECT m.machine_id, m.machine_code, m.name, m.location, m.unit_number,
               a.cycle, a.anomaly_score, a.severity, a.is_anomaly, a.explanation, a.detected_at
        FROM machines m
        JOIN anomaly_results a ON a.result_id = (
            SELECT result_id FROM anomaly_results
            WHERE machine_id = m.machine_id
            ORDER BY cycle DESC, result_id DESC LIMIT 1
        )
        ORDER BY a.severity DESC
        """
    )


def render(db, bundle, sim, meta) -> None:
    common.page_header(
        "Anomaly detection",
        "Isolation Forest surveillance of every machine's operating signature, with a "
        "calibrated 0-100 severity score.",
    )

    df = _latest_anomalies(db)
    if df.empty:
        st.info("No anomaly assessments recorded yet. Run a simulation step from the sidebar.")
        return

    flagged = df[df["is_anomaly"] == 1]
    threshold = config.THRESHOLDS["anomaly_severity"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        common.kpi_card("Machines monitored", f"{len(df)}", "Latest assessment each",
                        config.PALETTE["accent"])
    with c2:
        common.kpi_card("Anomalies flagged", f"{len(flagged)}",
                        f"{(100.0 * len(flagged) / len(df)):.0f}% of fleet",
                        config.PALETTE["critical"])
    with c3:
        common.kpi_card("Mean severity", f"{df['severity'].mean():.0f}/100", "Fleet-wide",
                        config.PALETTE["warning"])
    with c4:
        common.kpi_card("Detection threshold", f"{threshold:.0f}/100", "Severity cut-off",
                        config.PALETTE["info"])

    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("##### Severity distribution across the fleet")
        fig = go.Figure(go.Histogram(x=df["severity"], nbinsx=20,
                                     marker=dict(color=config.PALETTE["accent"])))
        fig.add_vline(x=threshold, line_dash="dash", line_color=config.PALETTE["critical"])
        fig.update_layout(xaxis_title="anomaly severity (0-100)", yaxis_title="machines",
                          showlegend=False)
        st.plotly_chart(common.style_fig(fig, 300), width="stretch")

    with right:
        st.markdown("##### Most deviating fault categories")
        cat_counts = _category_counts(bundle, df, db)
        if cat_counts.empty:
            st.success("No anomalous channels detected in the current fleet state.")
        else:
            fig = go.Figure(go.Bar(
                x=cat_counts["count"], y=cat_counts["category"], orientation="h",
                marker=dict(color=config.PALETTE["warning"]),
            ))
            fig.update_layout(xaxis_title="occurrences", showlegend=False)
            st.plotly_chart(common.style_fig(fig, 300), width="stretch")

    st.write("")
    st.markdown("##### Detected anomalies")
    show = df.copy()
    show["severity"] = show["severity"].round(0)
    show["anomaly_score"] = show["anomaly_score"].round(3)
    show["flag"] = show["is_anomaly"].map({1: "Anomaly", 0: "Normal"})
    st.dataframe(
        show[["machine_code", "location", "cycle", "flag", "severity",
              "anomaly_score", "explanation"]]
        .rename(columns={"machine_code": "machine", "flag": "status",
                         "severity": "severity", "anomaly_score": "score"}),
        width="stretch", hide_index=True, height=320,
    )

    st.write("")
    st.markdown("##### Reading-level diagnostics")
    labels = {r["machine_code"]: r for _, r in df.iterrows()}
    code = st.selectbox("Inspect machine", list(labels.keys()))
    row = labels[code]
    machine_id = int(row["machine_id"])
    reading = db.latest_reading(machine_id)
    if reading is None:
        st.info("No readings available for this machine.")
        return

    raw = add_auxiliary_sensors(pd.DataFrame([reading])).iloc[0]
    result = bundle.anomaly_detector.predict_row(raw)

    st.markdown(
        f"""<div class="card">
            <h4>{code} &nbsp; {_flag_pill(result.is_anomaly)}</h4>
            <div class="muted">severity {result.severity:.0f}/100 &nbsp;|&nbsp;
            decision score {result.score:.3f} &nbsp;|&nbsp; cycle {int(row['cycle'])}</div>
            <div style="margin-top:8px">{result.explanation}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    if result.contributors:
        cdf = pd.DataFrame(result.contributors)
        cdf = cdf.rename(columns={
            "name": "parameter", "value": "value", "unit": "unit", "category": "category",
            "z_score": "deviation (sigma)", "direction": "direction",
        })[["parameter", "value", "unit", "category", "deviation (sigma)", "direction"]]
        cdf["value"] = cdf["value"].round(3)
        cdf["deviation (sigma)"] = cdf["deviation (sigma)"].round(2)
        st.dataframe(cdf, width="stretch", hide_index=True)
    else:
        st.success("This machine's readings sit comfortably inside the learned healthy envelope.")


def _category_counts(bundle, df: pd.DataFrame, db) -> pd.DataFrame:
    """Tally the fault categories behind the currently flagged anomalies."""
    counts: dict = {}
    for _, row in df[df["is_anomaly"] == 1].iterrows():
        reading = db.latest_reading(int(row["machine_id"]))
        if reading is None:
            continue
        raw = add_auxiliary_sensors(pd.DataFrame([reading])).iloc[0]
        res = bundle.anomaly_detector.predict_row(raw)
        for c in res.contributors:
            counts[c["category"]] = counts.get(c["category"], 0) + 1
    if not counts:
        return pd.DataFrame(columns=["category", "count"])
    return (pd.DataFrame({"category": list(counts.keys()), "count": list(counts.values())})
            .sort_values("count", ascending=True))


def _flag_pill(is_anomaly: bool) -> str:
    if is_anomaly:
        return '<span class="pill pill-critical">Anomaly detected</span>'
    return '<span class="pill pill-normal">Within envelope</span>'
