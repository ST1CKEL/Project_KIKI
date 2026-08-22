"""Agent/tool audit. No secrets, no full prompts — hashes and short summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock

from kiki.storage.database import Database


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class AgentAuditEntry:
    id: int
    ts: str
    session_id: str | None
    actor: str
    event_type: str
    risk_class: str
    requested_action: str
    resolved_arguments_hash: str
    policy_decision: str
    approval_id: str | None
    result_status: str | None
    result_summary: str | None


class AgentAuditRepository:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._lock = Lock()

    def record(
        self,
        *,
        actor: str,
        event_type: str,
        risk_class: str,
        requested_action: str,
        arguments_hash: str,
        policy_decision: str,
        session_id: str | None = None,
        approval_id: str | None = None,
        result_status: str | None = None,
        result_summary: str | None = None,
    ) -> None:
        summary = (result_summary or "")[:400] or None
        with self._lock:
            self._db.conn.execute(
                "INSERT INTO agent_audit("
                "ts, session_id, actor, event_type, risk_class, requested_action, "
                "resolved_arguments_hash, policy_decision, approval_id, result_status, result_summary"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _now(),
                    session_id,
                    actor,
                    event_type,
                    risk_class,
                    requested_action,
                    arguments_hash,
                    policy_decision,
                    approval_id,
                    result_status,
                    summary,
                ),
            )
            self._db.conn.commit()

    def recent(self, limit: int = 50) -> list[AgentAuditEntry]:
        with self._lock:
            rows = self._db.conn.execute(
                "SELECT * FROM agent_audit ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [AgentAuditEntry(**dict(row)) for row in rows]
