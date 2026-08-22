"""Append-only audit log for tool requests, confirmations and results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from kiki.storage.database import Database


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class AuditEntry:
    id: int
    ts: str
    tool: str
    params_json: str
    decision: str
    result: str | None
    error: str | None


class AuditLog:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._lock = Lock()

    def record(
        self,
        tool: str,
        params: dict[str, Any],
        decision: str,
        *,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._db.conn.execute(
                "INSERT INTO audit_log(ts, tool, params_json, decision, result, error) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (_now(), tool, json.dumps(params, ensure_ascii=False, default=str), decision, result, error),
            )
            self._db.conn.commit()

    def recent(self, limit: int = 50) -> list[AuditEntry]:
        with self._lock:
            rows = self._db.conn.execute(
                "SELECT id, ts, tool, params_json, decision, result, error "
                "FROM audit_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [AuditEntry(**dict(row)) for row in rows]
