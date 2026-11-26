"""Login and self-service signup screen shown before the dashboard is accessible."""

from __future__ import annotations

import re

import streamlit as st

from src import config

# Roles a visitor may assign themselves. Administrator is intentionally excluded so that
# self-registration can never unlock the destructive fleet controls.
SELF_SIGNUP_ROLES = ["Maintenance engineer", "Viewer"]

_USERNAME_RE = re.compile(r"[A-Za-z0-9_]+")
_MIN_USERNAME = 3
_MAX_USERNAME = 20
_MIN_PASSWORD = 6


def render_login(auth) -> None:
    """Render the sign-in / create-account screen."""
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.write("")
        st.write("")
        st.markdown(
            '<div class="app-title" style="text-align:center">Self-healing machines</div>'
            '<div class="app-sub" style="text-align:center">'
            'Predictive maintenance for a connected fleet</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            tab_login, tab_signup = st.tabs(["Sign in", "Create account"])
            with tab_login:
                _login_form(auth)
            with tab_signup:
                _signup_form(auth)

        with st.expander("Demo credentials"):
            rows = "".join(
                f"<div style='margin:2px 0'><b>{u['username']}</b> / {u['password']}"
                f" &nbsp; <span class='muted'>({u['role']})</span></div>"
                for u in config.DEFAULT_USERS
            )
            st.markdown(rows, unsafe_allow_html=True)
            st.caption("Accounts are seeded on first launch. Update them in src/config.py "
                       "or create new ones via the AuthStore.")


# --------------------------------------------------------------------------------------
# Sign in
# --------------------------------------------------------------------------------------
def _login_form(auth) -> None:
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", key="login_user", placeholder="e.g. admin")
        password = st.text_input("Password", type="password", key="login_pw",
                                 placeholder="Password")
        submitted = st.form_submit_button("Sign in", type="primary", width="stretch")

    if submitted:
        user = auth.authenticate(username, password)
        if user:
            _start_session(user)
        else:
            st.error("Invalid username or password. Please try again.")


# --------------------------------------------------------------------------------------
# Sign up
# --------------------------------------------------------------------------------------
def _signup_form(auth) -> None:
    with st.form("signup_form", clear_on_submit=False):
        full_name = st.text_input("Full name", key="su_name", placeholder="e.g. Jane Doe")
        username = st.text_input("Username", key="su_user",
                                 placeholder="3-20 letters, digits or underscore")
        col_a, col_b = st.columns(2)
        with col_a:
            password = st.text_input("Password", type="password", key="su_pw")
        with col_b:
            confirm = st.text_input("Confirm password", type="password", key="su_pw2")
        role = st.selectbox("Role", SELF_SIGNUP_ROLES, key="su_role")
        submitted = st.form_submit_button("Create account", type="primary", width="stretch")

    if not submitted:
        return

    error = _validate_signup(auth, username, password, confirm)
    if error:
        st.error(error)
        return

    clean_username = username.strip()
    clean_name = full_name.strip() or clean_username
    auth.create_user(clean_username, password, clean_name, role)
    _start_session({"username": clean_username, "full_name": clean_name, "role": role})


def _validate_signup(auth, username, password, confirm) -> str | None:
    """Return a human-readable validation error, or ``None`` when the input is valid."""
    username = (username or "").strip()
    if not (_MIN_USERNAME <= len(username) <= _MAX_USERNAME):
        return f"Username must be between {_MIN_USERNAME} and {_MAX_USERNAME} characters."
    if not _USERNAME_RE.fullmatch(username):
        return "Username may only contain letters, digits and underscores."
    if auth.user_exists(username):
        return "That username is already taken. Please choose another."
    if len(password or "") < _MIN_PASSWORD:
        return f"Password must be at least {_MIN_PASSWORD} characters long."
    if password != confirm:
        return "Passwords do not match."
    return None


# --------------------------------------------------------------------------------------
# Session helpers
# --------------------------------------------------------------------------------------
def _start_session(user: dict) -> None:
    st.session_state["user"] = user
    st.rerun()


def sign_out() -> None:
    """Clear the authenticated session and return to the login screen."""
    st.session_state.pop("user", None)
    st.rerun()
