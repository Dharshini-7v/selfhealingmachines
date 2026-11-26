"""Analytics and reports module: fleet trends, model diagnostics and downloadable reports."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src import config
from views import common


def render(db, bundle, sim, meta) -> None:
    common.page_header(
        "Analytics & reports",
        "Model diagnostics, sensor performance, fleet health trends and downloadable "
        "maintenance reports.",
    )

    _model_diagnostics(bundle, meta)

    st.write("")
    tab_sensor, tab_health, tab_anomaly, tab_rul = st.tabs(
        ["Sensor performance", "Health trends", "Anomaly statistics", "RUL analytics"]
    )

    with tab_sensor:
        _sensor_performance(db)
    with tab_health:
        _health_trends(db)
    with tab_anomaly:
        _anomaly_statistics(db, meta)
    with tab_rul:
        _rul_analytics(db, bundle)

    st.write("")
    _reports(db)


# --------------------------------------------------------------------------------------
# Model diagnostics
# --------------------------------------------------------------------------------------
def _model_diagnostics(bundle, meta) -> None:
    metrics = meta.get("metrics", {})
    astats = meta.get("anomaly_stats", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        common.kpi_card("RUL RMSE", f"{metrics.get('rmse', 0):.1f}", "cycles (lower is better)",
                        config.PALETTE["accent"])
    with c2:
        common.kpi_card("RUL MAE", f"{metrics.get('mae', 0):.1f}", "mean absolute error",
                        config.PALETTE["info"])
    with c3:
        common.kpi_card("R2 score", f"{metrics.get('r2', 0):.3f}", "variance explained",
                        config.PALETTE["normal"])
    with c4:
        common.kpi_card("Early-life anomaly rate",
                        f"{100 * astats.get('healthy_anomaly_rate', 0):.0f}%",
                        "false positives", config.PALETTE["warning"])
    with c5:
        common.kpi_card("Late-life anomaly rate",
                        f"{100 * astats.get('late_life_anomaly_rate', 0):.0f}%",
                        "true detection", config.PALETTE["critical"])

    st.caption(
        f"Dataset: {meta.get('data_source', 'n/a')} ({meta.get('subset', '')}) | "
        f"trained {meta.get('trained_at', 'n/a')} | "
        f"{meta.get('n_train_units', 0)} train / {meta.get('n_test_units', 0)} test units | "
        f"RUL target capped at {meta.get('rul_cap', config.RUL_CAP)} cycles."
    )


# --------------------------------------------------------------------------------------
# Sensor performance
# --------------------------------------------------------------------------------------
def _sensor_performance(db) -> None:
    st.markdown("##### Fleet sensor performance")
    options = {f"{v['name']} ({k})": k for k, v in config.ALL_SENSOR_META.items()}
    choice = st.selectbox("Parameter", list(options.keys()))
    sensor = options[choice]
    meta_k = config.ALL_SENSOR_META[sensor]

    agg = db.query_df(
        f"""SELECT cycle, AVG({sensor}) AS avg, MIN({sensor}) AS lo, MAX({sensor}) AS hi,
                   COUNT(*) AS n
            FROM sensor_readings GROUP BY cycle ORDER BY cycle"""
    )
    if agg.empty:
        st.info("No sensor readings recorded yet.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg["cycle"], y=agg["hi"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=agg["cycle"], y=agg["lo"], mode="lines", fill="tonexty",
                             fillcolor="rgba(59,130,246,0.15)", line=dict(width=0),
                             name="min-max band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=agg["cycle"], y=agg["avg"], mode="lines",
                             name="fleet mean", line=dict(color=config.PALETTE["accent"], width=2)))
    fig.update_layout(xaxis_title="operating cycle",
                      yaxis_title=f"{meta_k['name']} [{meta_k['unit']}]", hovermode="x unified")
    st.plotly_chart(common.style_fig(fig, 380), width="stretch")
    st.caption("Fleet-aggregated signal across all monitored machines, aligned by operating cycle.")


# --------------------------------------------------------------------------------------
# Health trends
# --------------------------------------------------------------------------------------
def _health_trends(db) -> None:
    st.markdown("##### Fleet health trend")
    trend = db.query_df(
        """SELECT cycle, AVG(health_score) AS avg_health, MIN(health_score) AS min_health,
                  COUNT(*) AS samples
           FROM health_status GROUP BY cycle ORDER BY cycle"""
    )
    if trend.empty:
        st.info("No health history recorded yet.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["cycle"], y=trend["avg_health"], mode="lines",
                             name="mean health score",
                             line=dict(color=config.PALETTE["normal"], width=2)))
    fig.add_trace(go.Scatter(x=trend["cycle"], y=trend["min_health"], mode="lines",
                             name="worst machine", line=dict(color=config.PALETTE["critical"],
                                                             width=1, dash="dot")))
    fig.add_hline(y=config.THRESHOLDS["warning_score"], line_dash="dash",
                  line_color=config.PALETTE["warning"])
    fig.add_hline(y=config.THRESHOLDS["critical_score"], line_dash="dash",
                  line_color=config.PALETTE["critical"])
    fig.update_layout(xaxis_title="operating cycle", yaxis_title="health score (0-100)",
                      hovermode="x unified")
    st.plotly_chart(common.style_fig(fig, 380), width="stretch")

    status_counts = db.query_df(
        "SELECT status, COUNT(*) AS n FROM health_status GROUP BY status"
    )
    if not status_counts.empty:
        fig2 = go.Figure(go.Pie(
            labels=status_counts["status"], values=status_counts["n"], hole=0.6,
            marker=dict(colors=[config.STATUS_COLORS.get(s, config.PALETTE["accent"])
                                for s in status_counts["status"]]),
            textinfo="label+percent",
        ))
        fig2.update_layout(title="Share of all assessments by status")
        st.plotly_chart(common.style_fig(fig2, 320), width="stretch")


# --------------------------------------------------------------------------------------
# Anomaly statistics
# --------------------------------------------------------------------------------------
def _anomaly_statistics(db, meta) -> None:
    st.markdown("##### Anomaly detection statistics")
    arate = db.query_df(
        """SELECT cycle, AVG(is_anomaly) AS rate FROM anomaly_results
           GROUP BY cycle ORDER BY cycle"""
    )
    if arate.empty:
        st.info("No anomaly history recorded yet.")
        return

    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        fig = go.Figure(go.Scatter(x=arate["cycle"], y=arate["rate"], mode="lines",
                                   fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
                                   line=dict(color=config.PALETTE["critical"])))
        fig.update_layout(xaxis_title="operating cycle",
                          yaxis_title="fraction of machines flagged", hovermode="x unified")
        st.plotly_chart(common.style_fig(fig, 340), width="stretch")

    with col_b:
        summary = db.query_df(
            """SELECT is_anomaly, COUNT(*) AS n FROM anomaly_results GROUP BY is_anomaly"""
        )
        mapping = {0: "Normal", 1: "Anomaly"}
        fig = go.Figure(go.Pie(
            labels=[mapping.get(int(v), str(v)) for v in summary["is_anomaly"]],
            values=summary["n"], hole=0.6,
            marker=dict(colors=[config.PALETTE["normal"], config.PALETTE["critical"]]),
            textinfo="label+percent",
        ))
        fig.update_layout(title="All assessments")
        st.plotly_chart(common.style_fig(fig, 340), width="stretch")

    astats = meta.get("anomaly_stats", {})
    st.caption(
        f"Hold-out evaluation: {100 * astats.get('healthy_anomaly_rate', 0):.1f}% of early-life "
        f"(healthy) cycles flagged as anomalous, versus "
        f"{100 * astats.get('late_life_anomaly_rate', 0):.1f}% of degraded cycles - a clear "
        f"separation between normal and degrading behaviour."
    )


# --------------------------------------------------------------------------------------
# RUL analytics
# --------------------------------------------------------------------------------------
def _rul_analytics(db, bundle) -> None:
    st.markdown("##### RUL predictor feature importance")
    imp = bundle.rul_predictor.feature_importances().head(12)
    if not imp.empty:
        labels = [f"{config.SENSOR_META.get(f, config.AUX_SENSOR_META.get(f, {})).get('name', f)} ({f})"
                  for f in imp["feature"]]
        fig = go.Figure(go.Bar(
            x=imp["importance"], y=labels, orientation="h",
            marker=dict(color=config.PALETTE["info"]),
        ))
        fig.update_layout(xaxis_title="relative importance", yaxis=dict(autorange="reversed"),
                          showlegend=False)
        st.plotly_chart(common.style_fig(fig, 360), width="stretch")

    st.markdown("##### Predicted RUL versus operating cycle")
    hist = db.query_df(
        "SELECT cycle, predicted_rul, rul_lower, rul_upper FROM rul_predictions ORDER BY cycle"
    )
    if hist.empty:
        st.info("No RUL predictions recorded yet.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["rul_upper"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["rul_lower"], mode="lines",
                             fill="tonexty", fillcolor="rgba(56,189,248,0.12)",
                             line=dict(width=0), name="90% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["predicted_rul"], mode="markers",
                             name="predictions", marker=dict(size=4,
                             color=config.PALETTE["accent"], opacity=0.5)))
    fig.update_layout(xaxis_title="operating cycle", yaxis_title="predicted RUL (cycles)",
                      hovermode="closest")
    st.plotly_chart(common.style_fig(fig, 380), width="stretch")
    st.caption("Each point is a machine/cycle prediction; RUL declines as machines age.")


# --------------------------------------------------------------------------------------
# Report downloads
# --------------------------------------------------------------------------------------
def _reports(db) -> None:
    st.markdown("##### Downloadable reports")
    st.caption("Export the live operational data and maintenance records as CSV.")

    exports = {
        "Machine registry": ("SELECT * FROM machines", "machines.csv"),
        "Sensor readings": ("SELECT * FROM sensor_readings ORDER BY machine_id, cycle",
                            "sensor_readings.csv"),
        "Health status": ("SELECT * FROM health_status ORDER BY machine_id, cycle",
                          "health_status.csv"),
        "Anomaly results": ("SELECT * FROM anomaly_results ORDER BY machine_id, cycle",
                            "anomaly_results.csv"),
        "RUL predictions": ("SELECT * FROM rul_predictions ORDER BY machine_id, cycle",
                            "rul_predictions.csv"),
        "Work orders": ("SELECT * FROM work_orders ORDER BY created_at", "work_orders.csv"),
        "Maintenance history": ("SELECT * FROM maintenance_history ORDER BY completed_at",
                                "maintenance_history.csv"),
        "Active alerts": ("SELECT * FROM alerts ORDER BY created_at", "alerts.csv"),
    }

    columns = st.columns(4)
    for i, (label, (sql, filename)) in enumerate(exports.items()):
        df = db.query_df(sql)
        with columns[i % 4]:
            st.download_button(
                label=label,
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=filename,
                mime="text/csv",
                width="stretch",
                key=f"dl_{filename}",
            )
