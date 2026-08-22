"""Structured types for coding-agent sessions. No process control here."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AgentEventType(StrEnum):
    SESSION_STARTED = "session_started"
    STATUS_CHANGED = "status_changed"
    MESSAGE = "message"
    PLAN = "plan"
    TOOL_REQUEST = "tool_request"
    TOOL_RESULT = "tool_result"
    FILE_CHANGE = "file_change"
    DIFF_AVAILABLE = "diff_available"
    TEST_STARTED = "test_started"
    TEST_OUTPUT = "test_output"
    TEST_FINISHED = "test_finished"
    APPROVAL_REQUIRED = "approval_required"
    ERROR = "error"
    SESSION_FINISHED = "session_finished"


class AgentSessionStatus(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FINISHED = "finished"
    FAILED = "failed"
    DENIED = "denied"


class SessionKind(StrEnum):
    PLAN = "plan"
    IMPLEMENT = "implement"


class PermissionProfile(StrEnum):
    OBSERVE = "observe"
    DEVELOP = "develop"
    OPERATOR = "operator"


class AgentError(Exception):
    """Fail-closed agent/session error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def arguments_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AgentAvailability:
    ok: bool
    name: str
    binary: str
    version: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class AgentStartRequest:
    workspace_id: str
    workspace_path: str
    task: str
    kind: SessionKind
    permission_profile: PermissionProfile
    model: str = ""
    session_id: str = ""


@dataclass
class AgentSession:
    id: str
    workspace_id: str
    agent_name: str
    agent_version: str | None
    model_name: str | None
    task_text: str
    status: AgentSessionStatus
    permission_profile: str
    kind: SessionKind
    git_branch_before: str | None
    git_head_before: str | None
    started_at: str
    finished_at: str | None = None
    exit_code: int | None = None
    summary: str | None = None
    plan_session_id: str | None = None


@dataclass(frozen=True)
class AgentEvent:
    type: AgentEventType
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    ts: str = ""


@dataclass(frozen=True)
class AgentPlan:
    steps: tuple[str, ...]
    files: tuple[str, ...]
    tests: tuple[str, ...]
    risks: tuple[str, ...]
    questions: tuple[str, ...]
    raw: str = ""


@dataclass(frozen=True)
class AgentToolRequest:
    tool: str
    params: dict[str, Any]


@dataclass(frozen=True)
class AgentFileChange:
    path: str
    kind: str


@dataclass(frozen=True)
class AgentTestResult:
    profile: str
    ok: bool
    exit_code: int | None
    summary: str


def parse_plan_text(raw: str) -> AgentPlan:
    """Best-effort split of a free-text plan. Never executes anything."""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    steps = tuple(lines[:20])
    return AgentPlan(steps=steps, files=(), tests=(), risks=(), questions=(), raw=raw[:8000])
