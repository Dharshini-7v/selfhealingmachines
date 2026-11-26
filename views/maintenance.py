"""Maintenance management module: work orders, dispatch and maintenance history."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src import config
from views import common


def render(db, bundle, sim, meta) -> None:
    common.page_header(
        "Maintenance management",
        "Work orders generated automatically by the self-healing engine, plus manual "
        "dispatch and the full maintenance history.",
    )

    orders = db.get_work_orders()
    if orders.empty:
        st.info("No work orders yet. The self-healing engine raises them as machines degrade. "
                "Advance the fleet from the sidebar to generate activity.")
        _manual_create(db)
        return

    open_wo = orders[orders["status"] != "Completed"]
    pending = orders[orders["status"] == "Pending"]
    in_progress = orders[orders["status"] == "In Progress"]
    completed = orders[orders["status"] == "Completed"]
    critical = orders[orders["priority"] == "Critical"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        common.kpi_card("Open work orders", f"{len(open_wo)}", "Pending + in progress",
                        config.PALETTE["accent"])
    with c2:
        common.kpi_card("Critical priority", f"{len(critical)}", "Immediate action",
                        config.PALETTE["critical"])
    with c3:
        common.kpi_card("Pending", f"{len(pending)}", "Awaiting dispatch",
                        config.PALETTE["warning"])
    with c4:
        common.kpi_card("Completed", f"{len(completed)}", "Closed work orders",
                        config.PALETTE["normal"])

    st.write("")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("##### Work orders by priority")
        order = config.WORK_ORDER_PRIORITY
        counts = orders["priority"].value_counts().reindex(order).fillna(0)
        fig = go.Figure(go.Bar(
            x=order, y=counts.values,
            marker=dict(color=[config.PALETTE["normal"], config.PALETTE["info"],
                               config.PALETTE["warning"], config.PALETTE["critical"]]),
        ))
        fig.update_layout(xaxis_title="priority", yaxis_title="work orders", showlegend=False)
        st.plotly_chart(common.style_fig(fig, 280), width="stretch")

    with right:
        st.markdown("##### Work orders by status")
        fig = go.Figure(go.Pie(
            labels=list(config.WORK_ORDER_STATUS),
            values=[orders["status"].value_counts().get(s, 0) for s in config.WORK_ORDER_STATUS],
            hole=0.6,
            marker=dict(colors=[config.PALETTE["warning"], config.PALETTE["info"],
                                config.PALETTE["normal"]]),
            textinfo="label+value",
        ))
        st.plotly_chart(common.style_fig(fig, 280), width="stretch")

    st.write("")
    st.markdown("##### Work order register")
    tabs = st.tabs(["All", "Pending", "In progress", "Completed"])
    for tab, subset in zip(tabs, [orders, pending, in_progress, completed]):
        with tab:
            if subset.empty:
                st.info("No work orders in this category.")
                continue
            view = subset.copy()
            view["due"] = view["due_cycle"].apply(
                lambda v: "-" if v is None or (isinstance(v, float) and v != v) else int(v))
            st.dataframe(
                view[["work_order_id", "machine_code", "priority", "status", "title",
                      "recommended_action", "due", "created_at"]]
                .rename(columns={"work_order_id": "id", "machine_code": "machine",
                                 "recommended_action": "recommended action",
                                 "due": "due cycle"}),
                width="stretch", hide_index=True, height=300,
            )

    st.write("")
    _manual_create(db)

    st.write("")
    st.markdown("##### Action a work order")
    if open_wo.empty:
        st.success("No open work orders to action.")
    else:
        labels = {
            f"#{int(r['work_order_id'])} | {r['machine_code']} | {r['title']}": r
            for _, r in open_wo.iterrows()
        }
        sel = st.selectbox("Select work order", list(labels.keys()))
        row = labels[sel]
        col_a, col_b = st.columns(2)
        with col_a:
            new_status = st.selectbox("Set status", ["In Progress", "Completed"])
        with col_b:
            technician = st.text_input("Technician", value="Maintenance team")
        notes = st.text_area("Completion notes", value="", height=80)
        if st.button("Update work order", width="stretch"):
            wo_id = int(row["work_order_id"])
            if new_status == "Completed":
                db.complete_work_order(wo_id, technician, notes)
            else:
                db.update_work_order_status(wo_id, new_status)
            st.success(f"Work order #{wo_id} updated to '{new_status}'.")
            st.rerun()

    st.write("")
    st.markdown("##### Maintenance history")
    history = db.get_maintenance_history(limit=200)
    if history.empty:
        st.info("No maintenance actions recorded yet.")
    else:
        st.dataframe(
            history[["completed_at", "machine_code", "action_taken", "technician", "notes"]]
            .rename(columns={"completed_at": "completed at", "machine_code": "machine",
                             "action_taken": "action taken"}),
            width="stretch", hide_index=True, height=300,
        )


def _manual_create(db) -> None:
    """Allow an operator to raise a work order manually (in addition to automation)."""
    machines = db.get_machines()
    if machines.empty:
        return
    with st.expander("Raise a manual work order"):
        options = {f"{r['machine_code']} - {r['name']}": int(r["machine_id"])
                   for _, r in machines.iterrows()}
        chosen = st.selectbox("Machine", list(options.keys()), key="manual_wo_machine")
        col_a, col_b = st.columns(2)
        with col_a:
            priority = st.selectbox("Priority", config.WORK_ORDER_PRIORITY, index=1,
                                    key="manual_wo_priority")
        with col_b:
            title = st.text_input("Title", value="Manual inspection", key="manual_wo_title")
        description = st.text_area("Description",
                                   value="Manually raised inspection work order.", height=70,
                                   key="manual_wo_desc")
        action = st.text_input("Recommended action", value=config.ACTION_PLAYBOOK["General"],
                               key="manual_wo_action")
        if st.button("Create work order", key="manual_wo_create"):
            wo_id = db.create_work_order(options[chosen], priority, title, description, action)
            st.success(f"Created work order #{wo_id} for {chosen.split(' - ')[0]}.")
            st.rerun()
