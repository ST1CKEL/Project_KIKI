"""Types the harness is built from. No I/O, no dependencies, no side effects.

Every failure the harness can report is one of `ERROR_CODES` — a fixed category,
never an exception message. That is the same rule the voice layer follows: a
message can carry a path, a prompt or a token, and a category cannot.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

# The complete vocabulary of things that can go wrong. Anything outside this set
# is a bug in the harness, not a condition it is allowed to report.
ERROR_CODES: frozenset[str] = frozenset(
    {
        "unknown_tool",
        "invalid_arguments",
        "tool_unavailable",
        "tool_failed",
        "model_protocol_error",
        "step_limit",
        "tool_call_limit",
        "trace_write_failed",
        "run_busy",
        # Slice 2: the write tool and the binding in front of it.
        "note_exists",
        "confirmation_rejected",
        "confirmation_mismatch",
        "confirmation_already_used",
        "no_pending_confirmation",
        "confirmation_abandoned",
        "provider_error",
    }
)


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    # Waiting for a human. Explicitly *not* terminal: the run continues or ends,
    # but it never sits here after the process is gone.
    NEEDS_CONFIRMATION = "needs_confirmation"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    LIMIT_REACHED = "limit_reached"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL


_TERMINAL: frozenset[RunStatus] = frozenset(
    {RunStatus.COMPLETED, RunStatus.CANCELLED, RunStatus.FAILED, RunStatus.LIMIT_REACHED}
)

HARNESS_MESSAGE_CODES: frozenset[str] = frozenset(
    {
        "working",
        "tool_running",
        "needs_confirmation",
        "completed",
        "cancelled",
        "failed",
        "limit_reached",
    }
)


@dataclass(frozen=True)
class HarnessStatusEvent:
    """Structured, run-bound status transition. No internal or private details."""

    run_id: str
    status: RunStatus
    message_code: str
    terminal: bool

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id darf nicht leer sein")
        if not isinstance(self.status, RunStatus):
            raise ValueError(f"ungültiger RunStatus: {self.status}")
        if self.message_code not in HARNESS_MESSAGE_CODES:
            raise ValueError(f"unbekannter message_code: {self.message_code}")
        if self.terminal != self.status.is_terminal:
            raise ValueError(f"terminal={self.terminal} stimmt nicht mit Status {self.status} überein")


class ActionKind(StrEnum):
    TOOL_CALL = "tool_call"
    FINAL = "final"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class CancelToken:
    """Request-bound cooperative cancellation. One token, one run.

    A flag, never an exception: the loop checks it at its own boundaries and
    ends through the normal path, so a cancel can never be mistaken for a
    failure. Idempotent by construction.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _new_id("call"))


class RunBusyError(Exception):
    """A run is already active. The active one is untouched.

    Lives here rather than beside a runner: both the unified runner and the
    legacy one raise it, and a shared error should not make one package depend
    on the other's implementation.
    """

    error_code = "run_busy"


@dataclass(frozen=True)
class ToolResult:
    """Always tied to a call that really happened.

    `data` is what the model gets to see. It is small and it is checked: a tool
    that wants to return a home path or an environment has to be changed, not
    configured.
    """

    call_id: str
    name: str
    ok: bool
    data: dict[str, Any] | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.ok and self.error_code is not None:
            raise ValueError("ein erfolgreiches Ergebnis hat keinen Fehlercode")
        if not self.ok and self.error_code not in ERROR_CODES:
            raise ValueError(f"unbekannter Fehlercode: {self.error_code}")


@dataclass(frozen=True)
class ModelAction:
    """What the adapter decided: ask for a tool, or answer.

    Never validated in the constructor — a misbehaving adapter must be able to
    produce a malformed action so the runner can reject it as
    `model_protocol_error` rather than crash on construction.
    """

    kind: ActionKind
    tool_call: ToolCall | None = None
    final_text: str | None = None

    @classmethod
    def call(cls, name: str, arguments: dict[str, Any] | None = None) -> ModelAction:
        return cls(ActionKind.TOOL_CALL, tool_call=ToolCall(name, dict(arguments or {})))

    @classmethod
    def answer(cls, text: str) -> ModelAction:
        return cls(ActionKind.FINAL, final_text=text)


def validate_action(action: Any) -> str:
    """Empty when the action is usable; otherwise the error code to report.

    Checked by the runner on everything an adapter returns, because an adapter
    is the one part of this harness that will one day be a real model.
    """
    if not isinstance(action, ModelAction):
        return "model_protocol_error"
    if action.kind is ActionKind.TOOL_CALL:
        call = action.tool_call
        if call is None or action.final_text is not None:
            return "model_protocol_error"
        if not isinstance(call, ToolCall) or not isinstance(call.name, str) or not call.name:
            return "model_protocol_error"
        if not isinstance(call.arguments, dict):
            return "model_protocol_error"
        return ""
    if action.kind is ActionKind.FINAL:
        if action.tool_call is not None:
            return "model_protocol_error"
        if not isinstance(action.final_text, str) or not action.final_text.strip():
            return "model_protocol_error"
        return ""
    return "model_protocol_error"


@dataclass
class AgentRun:
    """One run and its single terminal state."""

    user_text: str
    id: str = field(default_factory=lambda: _new_id("run"))
    status: RunStatus = RunStatus.PENDING
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    final_text: str | None = None
    error_code: str | None = None
    tool_calls: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    def start(self) -> None:
        if self.status is not RunStatus.PENDING:
            raise RuntimeError("ein Run startet genau einmal")
        self.status = RunStatus.RUNNING

    def await_confirmation(self) -> None:
        """Pause for a human. Only from RUNNING, and never out of a terminal."""
        if self.status is not RunStatus.RUNNING:
            raise RuntimeError("nur ein laufender Run kann auf Bestätigung warten")
        self.status = RunStatus.NEEDS_CONFIRMATION

    def resume(self) -> None:
        if self.status is not RunStatus.NEEDS_CONFIRMATION:
            raise RuntimeError("nur ein wartender Run wird fortgesetzt")
        self.status = RunStatus.RUNNING

    def finish(
        self,
        status: RunStatus,
        *,
        final_text: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """Settle the run once, and refuse to pretend.

        A terminal run never goes back to RUNNING, `COMPLETED` needs an answer,
        and every other terminal state needs a category — so no failure can be
        dressed up as a success.
        """
        if self.is_terminal:
            raise RuntimeError("ein Run hat genau einen Terminalzustand")
        if not status.is_terminal:
            raise ValueError(f"{status} ist kein Terminalzustand")
        if status is RunStatus.COMPLETED:
            if not final_text or not final_text.strip():
                raise ValueError("COMPLETED braucht eine Antwort")
            if error_code is not None:
                raise ValueError("COMPLETED hat keinen Fehlercode")
        else:
            if final_text is not None:
                raise ValueError(f"{status} liefert keine Antwort")
            if status is not RunStatus.CANCELLED and error_code not in ERROR_CODES:
                raise ValueError(f"{status} braucht eine bekannte Fehlerkategorie")
        self.status = status
        self.final_text = final_text
        self.error_code = error_code
        self.finished_at = _now()
