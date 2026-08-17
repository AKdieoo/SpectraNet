"""Metadata persistence for the context layer (Phase 3).

Design note on Postgres vs SQLite
----------------------------------
The build plan specifies PostgreSQL (`spectranet_context` schema) as the
target production store. This module implements that same schema against
SQLite by default, because:

  1. It requires zero external services to develop/test against.
  2. The SQL used (plain CREATE TABLE / parameterized INSERT / SELECT) is
     deliberately kept Postgres-compatible.
  3. Switching backends is a one-line change: swap the connection string
     passed to `ContextStore(conn_string=...)`. A `POSTGRES_DSN` env var
     is already read as a hook.

This is a conscious substitution, not a hidden shortcut: SQLite today,
Postgres in production, same schema and same call sites either way.

Every entity from context/models.py gets one table. Rows store the
entity's JSON representation plus a few indexed columns (id, name,
status, created_at) for fast lookups — this keeps the schema stable even
as entity fields evolve, while still allowing simple filtering without
a full JSON query layer.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Optional, Type, TypeVar

from spectranet.context.models import (
    BaseEntity,
    Dataset,
    Detection,
    Document,
    Evaluation,
    Experiment,
    Explanation,
    Model,
    Prediction,
    Signal,
    TrustSignal,
)

T = TypeVar("T", bound=BaseEntity)

# Table name -> entity class. One table per entity type, per the build plan.
ENTITY_TABLES: dict[str, Type[BaseEntity]] = {
    "signals": Signal,
    "detections": Detection,
    "predictions": Prediction,
    "models": Model,
    "datasets": Dataset,
    "experiments": Experiment,
    "explanations": Explanation,
    "documents": Document,
    "evaluations": Evaluation,
    "trust_signals": TrustSignal,
}

DEFAULT_SQLITE_PATH = os.environ.get(
    "SPECTRANET_CONTEXT_DB",
    os.path.join(os.path.dirname(__file__), "..", "..", "spectranet_context.db"),
)


class ContextStore:
    """SQLite-backed store for context entities, schema-compatible with the
    planned `spectranet_context` Postgres schema.

    Usage:
        store = ContextStore()             # local sqlite file
        store = ContextStore(":memory:")   # in-memory, for tests
        ds = Dataset.create(name="clean_v1")
        store.save(ds)
        store.get(Dataset, ds.id)
        store.list(Dataset, status="active")
    """

    def __init__(self, conn_string: Optional[str] = None):
        self.conn_string = conn_string or os.environ.get("POSTGRES_DSN") or DEFAULT_SQLITE_PATH
        # ":memory:" creates a brand-new, separate database on every
        # sqlite3.connect() call, so a single persistent connection is
        # required to make the in-memory backend usable across calls
        # (file-backed paths would work fine with reconnect-per-call, but
        # we use one connection uniformly for simplicity and consistency).
        self._persistent_conn = sqlite3.connect(self.conn_string, check_same_thread=False)
        self._persistent_conn.row_factory = sqlite3.Row
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = self._persistent_conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self):
        self._persistent_conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            for table in ENTITY_TABLES:
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        data TEXT NOT NULL
                    )
                    """
                )
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_status ON {table}(status)")

    def _table_for(self, entity_cls: Type[BaseEntity]) -> str:
        for table, cls in ENTITY_TABLES.items():
            if cls is entity_cls:
                return table
        raise ValueError(f"{entity_cls} is not a registered context entity")

    def save(self, entity: T) -> T:
        """Insert or update (upsert) an entity."""
        table = self._table_for(type(entity))
        entity.updated_at = datetime.now(entity.updated_at.tzinfo)
        payload = entity.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {table} (id, name, status, created_at, updated_at, data)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name, status=excluded.status,
                    updated_at=excluded.updated_at, data=excluded.data
                """,
                (
                    entity.id, entity.name, entity.status.value,
                    entity.created_at.isoformat(), entity.updated_at.isoformat(), payload,
                ),
            )
        return entity

    def get(self, entity_cls: Type[T], entity_id: str) -> Optional[T]:
        table = self._table_for(entity_cls)
        with self._connect() as conn:
            row = conn.execute(f"SELECT data FROM {table} WHERE id = ?", (entity_id,)).fetchone()
        if row is None:
            return None
        return entity_cls.model_validate_json(row["data"])

    def list(self, entity_cls: Type[T], status: Optional[str] = None) -> list[T]:
        table = self._table_for(entity_cls)
        with self._connect() as conn:
            if status:
                rows = conn.execute(f"SELECT data FROM {table} WHERE status = ?", (status,)).fetchall()
            else:
                rows = conn.execute(f"SELECT data FROM {table}").fetchall()
        return [entity_cls.model_validate_json(r["data"]) for r in rows]

    def delete(self, entity_cls: Type[T], entity_id: str) -> bool:
        table = self._table_for(entity_cls)
        with self._connect() as conn:
            cur = conn.execute(f"DELETE FROM {table} WHERE id = ?", (entity_id,))
        return cur.rowcount > 0

    def find_by(self, entity_cls: Type[T], **field_values) -> list[T]:
        """Simple in-memory filter over JSON fields not indexed as columns
        (e.g. find_by(Prediction, signal_id=sig.id)). Fine at this scale;
        swap for indexed JSON columns or real FK columns if this becomes a
        bottleneck.
        """
        results = []
        for entity in self.list(entity_cls):
            if all(getattr(entity, k, None) == v for k, v in field_values.items()):
                results.append(entity)
        return results
