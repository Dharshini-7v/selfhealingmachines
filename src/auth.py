"""Authentication backend.

A compact, self-contained user store backed by its own SQLite file. Passwords are never
stored in plain text - each account keeps a random salt and a PBKDF2-HMAC-SHA256 hash.
The store lives in a separate database (``auth.db``) so that rebuilding the fleet (which
resets the operational database) can never wipe user accounts.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
from typing import Dict, Optional

import pandas as pd

from src import config

_ITERATIONS = 200_000


def _now() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _hash_password(password: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256 hash of ``password`` using the given hex-encoded salt."""
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             bytes.fromhex(salt), _ITERATIONS)
    return dk.hex()


class AuthStore:
    """Thread-safe SQLite user store with salted password hashing."""

    def __init__(self, path: str = config.AUTH_DB_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._seed_defaults()

    # ------------------------------------------------------------------ internals
    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    username      TEXT UNIQUE NOT NULL,
                    full_name     TEXT,
                    role          TEXT NOT NULL DEFAULT 'Viewer',
                    password_hash TEXT NOT NULL,
                    salt          TEXT NOT NULL,
                    created_at    TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def _seed_defaults(self) -> None:
        if self.count() > 0:
            return
        for user in config.DEFAULT_USERS:
            self.create_user(user["username"], user["password"],
                             user["full_name"], user["role"])

    def _query(self, sql: str, params=()) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query(sql, self._conn, params=tuple(params))

    def _execute(self, sql: str, params=()) -> None:
        with self._lock:
            self._conn.execute(sql, tuple(params))
            self._conn.commit()

    # ------------------------------------------------------------------ public API
    def create_user(self, username: str, password: str, full_name: str, role: str) -> None:
        """Create or replace a user, storing only a salted password hash."""
        salt = secrets.token_hex(16)
        self._execute(
            """INSERT OR REPLACE INTO users
               (username, full_name, role, password_hash, salt, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (username.strip(), full_name, role, _hash_password(password, salt), salt, _now()),
        )

    def user_exists(self, username: str) -> bool:
        """Whether an account already uses the given username."""
        if not username or not username.strip():
            return False
        return not self._query("SELECT 1 FROM users WHERE username = ?",
                               (username.strip(),)).empty

    def authenticate(self, username: Optional[str],
                     password: Optional[str]) -> Optional[Dict[str, str]]:
        """Return the user record when the credentials are valid, otherwise ``None``."""
        if not username or not password:
            return None
        df = self._query("SELECT * FROM users WHERE username = ?", (username.strip(),))
        if df.empty:
            return None
        row = df.iloc[0]
        candidate = _hash_password(password, row["salt"])
        if secrets.compare_digest(candidate, row["password_hash"]):
            return {
                "username": row["username"],
                "full_name": row["full_name"],
                "role": row["role"],
            }
        return None

    def list_users(self) -> pd.DataFrame:
        return self._query(
            "SELECT username, full_name, role, created_at FROM users ORDER BY user_id"
        )

    def count(self) -> int:
        return int(self._query("SELECT COUNT(*) AS n FROM users")["n"].iloc[0])


# --------------------------------------------------------------------------------------
# Singleton accessor
# --------------------------------------------------------------------------------------
_AUTH_SINGLETON: Optional[AuthStore] = None


def get_auth_store(path: Optional[str] = None) -> AuthStore:
    """Process-wide cached accessor for the user store."""
    global _AUTH_SINGLETON
    if _AUTH_SINGLETON is None:
        _AUTH_SINGLETON = AuthStore(path or config.AUTH_DB_PATH)
    return _AUTH_SINGLETON
