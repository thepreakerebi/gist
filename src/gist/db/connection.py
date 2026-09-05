"""Neon Postgres connectivity for the web demo.

Deliberately thin: a lazily-created connection pool, the DDL bootstrap, and a
context manager. Everything that knows about Gist's tables lives in
:mod:`gist.db.repository` instead, so the rest of the codebase never imports
psycopg directly.

The database is *optional*. The CLI, the evaluation harnesses, and every
measured capstone result run without it — persistence exists to make the
demo's library and chat history work, not to sit in the path of a benchmark.
Callers should treat :func:`database_url` returning ``None`` as "no database
configured" rather than as an error.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_pool: Any | None = None
_pool_lock = Lock()


class DatabaseNotConfiguredError(RuntimeError):
    """Raised when a database operation is attempted with no DATABASE_URL."""


def database_url() -> str | None:
    """Return the configured Neon connection string, if any."""

    url = os.environ.get("DATABASE_URL", "").strip()
    return url or None


def is_configured() -> bool:
    return database_url() is not None


def _require_url() -> str:
    url = database_url()
    if url is None:
        raise DatabaseNotConfiguredError(
            "DATABASE_URL is not set; copy .env.example to .env and add your "
            "Neon connection string to enable the video library."
        )
    return url


def get_pool() -> Any:
    """Return the process-wide connection pool, creating it on first use.

    Neon closes idle connections aggressively (compute suspends on the free
    tier), so the pool is configured to check a connection before handing it
    out rather than failing the first query after an idle period.
    """

    global _pool
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:
            return _pool

        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        _pool = ConnectionPool(
            conninfo=_require_url(),
            min_size=0,
            max_size=8,
            kwargs={"row_factory": dict_row, "autocommit": True},
            check=ConnectionPool.check_connection,
            open=True,
        )
        return _pool


@contextmanager
def connection() -> Iterator[Any]:
    """Borrow a pooled connection."""

    with get_pool().connection() as conn:
        yield conn


@contextmanager
def cursor() -> Iterator[Any]:
    """Borrow a pooled connection and yield a cursor on it."""

    with connection() as conn, conn.cursor() as cur:
        yield cur


def apply_schema() -> None:
    """Create the schema if it does not exist.

    The DDL is idempotent (``create ... if not exists`` throughout), so this is
    safe to call on every API startup and doubles as the migration entry point
    for a project small enough not to need a migration tool.
    """

    with connection() as conn:
        conn.execute(SCHEMA_PATH.read_text())


def close_pool() -> None:
    """Close the pool. Used by tests and on API shutdown."""

    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None
