"""Self-Healing Machines - Predictive Maintenance for a Connected Fleet.

Streamlit entry point. Provides an authentication gate, sidebar navigation, fleet
simulation controls and routes to the individual dashboard modules.

Run with:  python -m streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src import config
from src.auth import get_auth_store
from src.training import train_and_save
from views import analytics, anomaly, common, login, maintenance, monitoring, overview, rul

st.set_page_config(
    page_title="Self-Healing Machines",
    layout="wide",
    initial_sidebar_state="expanded",
)
common.inject_theme()

PAGES = {
    "Dashboard overview": overview.render,
    "Machine monitoring": monitoring.render,
    "Anomaly detection": anomaly.render,
    "RUL prediction": rul.render,
    "Maintenance management": maintenance.render,
    "Analytics & reports": analytics.render,
}


def main() -> None:
    auth = get_auth_store()

    # ---- authentication gate ---------------------------------------------------------
    if config.AUTH_ENABLED and not st.session_state.get("user"):
        login.render_login(auth)
        st.stop()  # nothing below runs until a user signs in

    user = st.session_state.get("user") or {"full_name": "Guest", "role": "Viewer",
                                            "username": "guest"}
    is_admin = user["role"] == config.ROLE_ADMIN

    with st.spinner("Loading models and warming up the fleet..."):
        db, bundle, sim, meta = common.get_resources(config.DEFAULT_FLEET_SIZE)

    with st.sidebar:
        st.markdown('<div class="app-title">Self-healing machines</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-sub">Predictive maintenance for a connected fleet</div>',
                    unsafe_allow_html=True)

        st.markdown(
            f"""<div class="card" style="padding:10px 12px; margin-bottom:8px">
                <div><b>{user['full_name']}</b></div>
                <div class="muted">@{user['username']} &nbsp;|&nbsp; {user['role']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if config.AUTH_ENABLED:
            if st.button("Sign out", width="stretch"):
                login.sign_out()
        st.divider()

        page = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()

        st.markdown("##### Fleet simulation")
        cycles = st.slider("Cycles per step", min_value=1, max_value=30,
                           value=config.SIM_STEP_CYCLES)
        if st.button("Advance fleet", width="stretch", type="primary"):
            summary = sim.step(cycles)
            st.success(
                f"Advanced {summary.evaluated} machines. "
                f"{summary.work_orders_created} work order(s) created, "
                f"{summary.alerts_raised} alert(s) raised."
            )
            st.rerun()

        if not config.AUTH_ENABLED or is_admin:
            with st.expander("Fleet controls"):
                st.caption("Rebuild a fresh fleet from the C-MAPSS trajectories.")
                if st.button("Rebuild fleet", width="stretch"):
                    db.reset()
                    sim.initialize(force=True)
                    st.rerun()
                st.caption("Retrain the Isolation Forest and Random Forest models.")
                if st.button("Retrain models", width="stretch"):
                    with st.spinner("Retraining models..."):
                        train_and_save(verbose=False)
                    st.cache_resource.clear()
                    st.rerun()
        else:
            st.caption("Fleet controls are restricted to administrators.")

        st.divider()
        summary = db.fleet_summary()
        st.metric("Fleet health", f"{summary['fleet_health']}%")
        st.caption(
            f"{summary['total']} machines monitored\n\n"
            f"{summary['normal']} normal / {summary['warning']} attention / "
            f"{summary['critical']} critical"
        )
        st.caption(f"Data source: {meta.get('data_source', 'n/a')} "
                   f"({meta.get('subset', '')})")

    try:
        PAGES[page](db, bundle, sim, meta)
    except Exception as exc:  # surface a friendly error instead of a stack trace
        st.error(f"This module could not be rendered: {exc}")
        st.exception(exc)


if __name__ == "__main__":
    main()
