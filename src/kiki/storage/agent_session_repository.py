"""Persist agent sessions and their events."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from kiki.agents.models import AgentEvent, AgentEventType, AgentSession, AgentSessionStatus, SessionKind
from kiki.storage.database import Database

_MAX_EVENT_JSON_CHARS = 8000


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AgentSessionRepository:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._lock = Lock()

    def insert(self, session: AgentSession) -> AgentSession:
        with self._lock:
            self._db.conn.execute(
                "INSERT INTO agent_sessions("
                "id, workspace_id, agent_name, agent_version, model_name, task_text, "
                "status, permission_profile, kind, git_branch_before, git_head_before, "
                "started_at, finished_at, exit_code, summary, plan_session_id"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session.id,
                    session.workspace_id,
                    session.agent_name,
                    session.agent_version,
                    session.model_name,
                    session.task_text,
                    session.status.value,
                    session.permission_profile,
                    session.kind.value,
                    session.git_branch_before,
                    session.git_head_before,
                    session.started_at,
                    session.finished_at,
                    session.exit_code,
                    session.summary,
                    session.plan_session_id,
                ),
            )
            self._db.conn.commit()
        return session

    def get(self, session_id: str) -> AgentSession | None:
        with self._lock:
            row = self._db.conn.execute(
                "SELECT * FROM agent_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return _session_from_row(row) if row else None

    def list_for_workspace(self, workspace_id: str) -> list[AgentSession]:
        with self._lock:
            rows = self._db.conn.execute(
                "SELECT * FROM agent_sessions WHERE workspace_id = ? ORDER BY started_at DESC",
                (workspace_id,),
            ).fetchall()
        return [_session_from_row(row) for row in rows]

    def update_status(
        self,
        session_id: str,
        status: AgentSessionStatus,
        *,
        exit_code: int | None = None,
        summary: str | None = None,
        finished: bool = False,
    ) -> None:
        finished_at = _now() if finished else None
        with self._lock:
            self._db.conn.execute(
                "UPDATE agent_sessions SET status = ?, exit_code = COALESCE(?, exit_code), "
                "summary = COALESCE(?, summary), finished_at = COALESCE(?, finished_at) "
                "WHERE id = ?",
                (status.value, exit_code, summary, finished_at, session_id),
            )
            self._db.conn.commit()

    def add_event(self, session_id: str, event: AgentEvent) -> None:
        payload_json = _event_payload_json(event)
        with self._lock:
            self._db.conn.execute(
                "INSERT INTO agent_events(id, session_id, ts, event_type, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    session_id,
                    event.ts or _now(),
                    event.type.value,
                    payload_json,
                ),
            )
            self._db.conn.commit()

    def list_events(self, session_id: str) -> list[AgentEvent]:
        with self._lock:
            rows = self._db.conn.execute(
                "SELECT ts, event_type, payload_json FROM agent_events "
                "WHERE session_id = ? ORDER BY ts ASC",
                (session_id,),
            ).fetchall()
        events: list[AgentEvent] = []
        for row in rows:
            data: dict[str, Any]
            try:
                loaded = json.loads(row["payload_json"])
                data = loaded if isinstance(loaded, dict) else {"raw": loaded}
            except json.JSONDecodeError:
                data = {}
            text = str(data.pop("text", ""))
            events.append(
                AgentEvent(
                    type=AgentEventType(row["event_type"]),
                    text=text,
                    data=data,
                    ts=row["ts"],
                )
            )
        return events


    def insert_test_run(
        self,
        *,
        test_id: str,
        workspace_id: str,
        profile: str,
        argv: list[str],
        session_id: str | None = None,
    ) -> None:
        with self._lock:
            self._db.conn.execute(
                "INSERT INTO test_runs("
                "id, session_id, workspace_id, profile, argv_json, status, started_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    test_id,
                    session_id,
                    workspace_id,
                    profile,
                    json.dumps(argv, ensure_ascii=False),
                    "running",
                    _now(),
                ),
            )
            self._db.conn.commit()

    def finish_test_run(
        self,
        test_id: str,
        *,
        status: str,
        exit_code: int | None,
        summary: str,
        output: str = "",
        output_truncated: bool = False,
    ) -> None:
        with self._lock:
            self._db.conn.execute(
                "UPDATE test_runs SET status = ?, exit_code = ?, finished_at = ?, summary = ?, "
                "output_text = ?, output_truncated = ? "
                "WHERE id = ?",
                (
                    status,
                    exit_code,
                    _now(),
                    summary[:400],
                    output[:65536],
                    int(output_truncated),
                    test_id,
                ),
            )
            self._db.conn.commit()


def _session_from_row(row: object) -> AgentSession:
    data = dict(row)  # type: ignore[arg-type]
    return AgentSession(
        id=str(data["id"]),
        workspace_id=str(data["workspace_id"]),
        agent_name=str(data["agent_name"]),
        agent_version=data["agent_version"],
        model_name=data["model_name"],
        task_text=str(data["task_text"]),
        status=AgentSessionStatus(str(data["status"])),
        permission_profile=str(data["permission_profile"]),
        kind=SessionKind(str(data["kind"])),
        git_branch_before=data["git_branch_before"],
        git_head_before=data["git_head_before"],
        started_at=str(data["started_at"]),
        finished_at=data["finished_at"],
        exit_code=data["exit_code"],
        summary=data["summary"],
        plan_session_id=data.get("plan_session_id"),
    )


def _event_payload_json(event: AgentEvent) -> str:
    """Keep event JSON valid while bounding a potentially huge tool transcript."""
    payload = {
        str(key): value
        for key, value in event.data.items()
        if str(key) not in {"text", "_payload_truncated"}
    }
    payload["text"] = event.text
    try:
        encoded = _strict_json(payload)
    except (TypeError, ValueError, RecursionError):
        encoded = ""
    if encoded and len(encoded) <= _MAX_EVENT_JSON_CHARS:
        return encoded

    compact: dict[str, Any] = {
        "text": event.text,
        "_payload_truncated": True,
    }
    for raw_key, value in event.data.items():
        key = str(raw_key)
        if (
            key in {"text", "_payload_truncated"}
            or len(key) > 80
            or not isinstance(value, (str, int, float, bool, type(None)))
        ):
            continue
        try:
            rendered = _strict_json(value)
        except (TypeError, ValueError, RecursionError):
            continue
        if len(rendered) > 256:
            continue
        candidate = {**compact, key: value, "text": ""}
        try:
            if len(_strict_json(candidate)) <= 1400:
                compact[key] = value
        except (TypeError, ValueError, RecursionError):
            continue

    text = str(compact["text"])
    low = 0
    high = len(text)
    while low < high:
        middle = (low + high + 1) // 2
        compact["text"] = text[:middle]
        candidate = _strict_json(compact)
        if len(candidate) <= _MAX_EVENT_JSON_CHARS:
            low = middle
        else:
            high = middle - 1
    compact["text"] = text[:low]
    encoded = _strict_json(compact)
    if len(encoded) <= _MAX_EVENT_JSON_CHARS:
        return encoded
    return _strict_json({"text": "", "_payload_truncated": True})


def _strict_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)
