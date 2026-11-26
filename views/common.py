"""Shared UI helpers and cached application resources for the Streamlit dashboard."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.data_loader import load_dataset
from src.database import Database, get_db
from src.inference import ModelBundle
from src.simulation import FleetSimulator
from src.training import bundle_exists, train_and_save


# --------------------------------------------------------------------------------------
# Theme / CSS
# --------------------------------------------------------------------------------------
def inject_theme() -> None:
    """Inject small CSS refinements that complement the dark theme in .streamlit/config.toml."""
    p = config.PALETTE
    st.markdown(
        f"""
        <style>
        /* Base surfaces (the theme config sets native widget colours) */
        .stApp {{ background: {p['bg']}; color: {p['text']}; }}
        .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; }}

        /* Typography */
        h1, h2, h3, h4, h5 {{ color: {p['text']}; letter-spacing: .2px; }}
        a {{ color: {p['accent']}; }}
        hr {{ border-color: {p['border']}; }}
        .app-title {{ font-size: 1.6rem; font-weight: 700; margin-bottom: .1rem;
            color: {p['text']}; }}
        .app-sub {{ color: {p['muted']}; font-size: .85rem; margin-bottom: 1.1rem; }}
        .muted {{ color: {p['muted']}; font-size: .8rem; }}

        /* Cards and KPI tiles */
        .kpi {{ background: {p['panel']}; border: 1px solid {p['border']};
            border-radius: 12px; padding: 16px 18px; height: 100%; }}
        .kpi .label {{ color: {p['muted']}; font-size: .78rem; letter-spacing: .3px; }}
        .kpi .value {{ font-size: 1.9rem; font-weight: 700; margin-top: 4px; line-height: 1.1;
            color: {p['text']}; }}
        .kpi .sub {{ color: {p['muted']}; font-size: .74rem; margin-top: 4px; }}
        .card {{ background: {p['panel']}; border: 1px solid {p['border']};
            border-radius: 12px; padding: 16px 18px; margin-bottom: 12px; }}
        .card h4 {{ margin: 0 0 6px 0; font-size: .95rem; color: {p['text']}; }}

        /* Status pills */
        .pill {{ display: inline-block; padding: 3px 12px; border-radius: 999px;
            font-size: .72rem; font-weight: 600; }}
        .pill-normal {{ background: rgba(34,197,94,.16); color: {p['normal']};
            border: 1px solid rgba(34,197,94,.4); }}
        .pill-warning {{ background: rgba(245,158,11,.16); color: {p['warning']};
            border: 1px solid rgba(245,158,11,.4); }}
        .pill-critical {{ background: rgba(239,68,68,.16); color: {p['critical']};
            border: 1px solid rgba(239,68,68,.4); }}
        .pill-info {{ background: rgba(56,189,248,.16); color: {p['info']};
            border: 1px solid rgba(56,189,248,.4); }}

        /* Native Streamlit components - borders, radii and weights only,
           colours come from the theme config so they stay in sync */
        section[data-testid="stSidebar"] {{ border-right: 1px solid {p['border']}; }}
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
            border: 1px solid {p['border']}; border-radius: 10px; overflow: hidden; }}
        div[data-testid="stExpander"] details {{ border: 1px solid {p['border']};
            border-radius: 10px; }}
        button[data-baseweb="tab"] {{ font-weight: 600; }}
        div[data-testid="stMetricValue"] {{ color: {p['text']}; }}
        .stButton > button {{ border-radius: 8px; font-weight: 600; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="app-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="app-sub">{subtitle}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "", accent: Optional[str] = None) -> None:
    accent_style = f"border-left: 4px solid {accent};" if accent else ""
    st.markdown(
        f"""<div class="kpi" style="{accent_style}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def status_pill(status: str) -> str:
    css = {config.STATUS_NORMAL: "pill-normal",
           config.STATUS_WARNING: "pill-warning",
           config.STATUS_CRITICAL: "pill-critical"}.get(status, "pill-info")
    return f'<span class="pill {css}">{status}</span>'


def info_card(title: str, body_html: str) -> None:
    st.markdown(
        f'<div class="card"><h4>{title}</h4><div>{body_html}</div></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# Plotly styling
# --------------------------------------------------------------------------------------
def style_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    p = config.PALETTE
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=p["text"], size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(gridcolor=p["border"], zerolinecolor=p["border"]),
        yaxis=dict(gridcolor=p["border"], zerolinecolor=p["border"]),
    )
    return fig


# --------------------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_resources(fleet_size: int = config.DEFAULT_FLEET_SIZE) -> Tuple[Database, ModelBundle, FleetSimulator, dict]:
    """Load (or train/seed) the shared application resources exactly once per process."""
    if not bundle_exists():
        train_and_save(verbose=False)

    db = get_db()
    bundle = ModelBundle.load()
    dataset = load_dataset()
    simulator = FleetSimulator(db, bundle, dataset=dataset, fleet_size=fleet_size)
    simulator.initialize()
    meta = dict(bundle.metadata)
    meta["dataset_source"] = dataset.source
    meta["dataset_notes"] = dataset.notes
    return db, bundle, simulator, meta


def sensor_table(readings: Dict[str, float]) -> pd.DataFrame:
    """Render a sensor snapshot as a tidy table with names, units and categories."""
    rows = []
    for key, meta in config.ALL_SENSOR_META.items():
        if key in readings:
            rows.append({
                "parameter": meta["name"],
                "key": key,
                "value": round(float(readings[key]), 3),
                "unit": meta["unit"],
                "category": meta["category"],
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        order = {c: i for i, c in enumerate(config.CATEGORY_ORDER)}
        df["_o"] = df["category"].map(order).fillna(99)
        df = df.sort_values(["_o", "parameter"]).drop(columns="_o").reset_index(drop=True)
    return df
