"""One-time approvals bound to tool id + canonical parameter hash."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from kiki.agents.models import arguments_hash
from kiki.storage.database import Database


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ApprovalRequest:
    id: str
    session_id: str | None
    tool: str
    params: dict[str, Any]
    params_hash: str
    risk_class: str
    created_at: str


class ApprovalRepository:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._lock = Lock()

    def create(
        self,
        tool: str,
        params: dict[str, Any],
        *,
        risk_class: str,
        session_id: str | None = None,
        stored_params: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        aid = str(uuid.uuid4())
        digest = arguments_hash(params)
        ts = _now()
        persisted = params if stored_params is None else stored_params
        blob = json.dumps(persisted, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock:
            self._db.conn.execute(
                "INSERT INTO approval_requests("
                "id, session_id, tool, params_json, params_hash, risk_class, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                (aid, session_id, tool, blob, digest, risk_class, ts),
            )
            self._db.conn.commit()
        return ApprovalRequest(
            id=aid,
            session_id=session_id,
            tool=tool,
            params=params,
            params_hash=digest,
            risk_class=risk_class,
            created_at=ts,
        )

    def get(self, approval_id: str) -> ApprovalRequest | None:
        with self._lock:
            row = self._db.conn.execute(
                "SELECT * FROM approval_requests WHERE id = ?",
                (approval_id,),
            ).fetchone()
        if row is None:
            return None
        params = json.loads(row["params_json"])
        return ApprovalRequest(
            id=row["id"],
            session_id=row["session_id"],
            tool=row["tool"],
            params=params if isinstance(params, dict) else {},
            params_hash=row["params_hash"],
            risk_class=row["risk_class"],
            created_at=row["created_at"],
        )

    def decide(self, approval_id: str, *, approved: bool, actor: str = "user") -> None:
        with self._lock:
            self._db.conn.execute(
                "INSERT INTO approval_decisions(id, approval_id, decided_at, approved, actor, consumed) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (str(uuid.uuid4()), approval_id, _now(), 1 if approved else 0, actor),
            )
            self._db.conn.commit()

    def consume_if_valid(self, approval_id: str, tool: str, params: dict[str, Any]) -> bool:
        """Return True once if this approval matches tool+params and was granted."""
        digest = arguments_hash(params)
        with self._lock:
            row = self._db.conn.execute(
                "SELECT r.tool AS tool, r.params_hash AS params_hash, "
                "d.id AS decision_id, d.approved AS approved, d.consumed AS consumed "
                "FROM approval_requests r "
                "JOIN approval_decisions d ON d.approval_id = r.id "
                "WHERE r.id = ?",
                (approval_id,),
            ).fetchone()
            if row is None:
                return False
            if int(row["approved"]) != 1 or int(row["consumed"]) != 0:
                return False
            if row["tool"] != tool or row["params_hash"] != digest:
                return False
            self._db.conn.execute(
                "UPDATE approval_decisions SET consumed = 1 WHERE id = ?",
                (row["decision_id"],),
            )
            self._db.conn.commit()
            return True
