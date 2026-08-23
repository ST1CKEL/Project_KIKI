"""What the application talks to. Knows about runs; knows nothing about GTK.

The application supplies four callbacks — status, answer, confirmation and
speech — and this layer decides *when* each fires and *what text* it gets. Every
callback is invoked with short, already-sanitised German: never a path, never a
tool argument, never an exception, never anything from the trace.

Threading is the application's business. This module only promises that it does
its own work on the asyncio side and hands finished strings over; the wiring
marshals them to the GTK thread.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kiki.harness.confirmation import ConfirmationError, ConfirmationRequest
from kiki.harness.models import AgentRun, HarnessStatusEvent, RunStatus
from kiki.harness.runner import AgentRunner, RunBusyError

log = logging.getLogger(__name__)

# What the user reads. Short, German, and free of anything internal.
STATUS_WORKING = "KIKI arbeitet …"
STATUS_NEEDS_CONFIRMATION = "KIKI benötigt Bestätigung"
STATUS_CANCELLED = "KIKI abgebrochen"
STATUS_DONE = "KIKI fertig"
STATUS_FAILED = "KIKI konnte das nicht ausführen"

SPEECH_NEEDS_CONFIRMATION = (
    "Ich habe eine Notiz vorbereitet. Bitte bestätige sie in der Oberfläche."
)
SPEECH_NOTE_CREATED = "Die Notiz ist angelegt."
SPEECH_REJECTED = "Gut, ich lege nichts an."
SPEECH_FAILED = "Das hat nicht geklappt."

# One line per failure category. A category the user never has to decode, and
# never the underlying message.
_FAILURE_TEXT: dict[str, str] = {
    "unknown_tool": "Dieses Werkzeug kenne ich nicht.",
    "invalid_arguments": "Die Angaben passen nicht.",
    "tool_unavailable": "Das Werkzeug steht gerade nicht bereit.",
    "tool_failed": "Das Werkzeug hat nicht funktioniert.",
    "note_exists": "Eine Notiz mit diesem Namen gibt es schon.",
    "model_protocol_error": "Ich habe mich vertan.",
    "provider_error": "Ich konnte das Modell nicht erreichen.",
    "step_limit": "Das wurde mir zu verschachtelt.",
    "tool_call_limit": "Das wurde mir zu verschachtelt.",
    "trace_write_failed": "Ich konnte den Ablauf nicht mitschreiben.",
    "confirmation_rejected": "Gut, ich lege nichts an.",
}


@dataclass(frozen=True)
class SessionCallbacks:
    """Everything that leaves the harness. All optional, all sanitised."""

    on_status: Callable[[HarnessStatusEvent], None] | None = None
    on_answer: Callable[[str], None] | None = None
    on_confirmation: Callable[[ConfirmationRequest], None] | None = None
    on_speak: Callable[[str], None] | None = None


class HarnessSession:
    """One runner, one active run, and the text the user sees and hears."""

    def __init__(
        self,
        runner: AgentRunner,
        callbacks: SessionCallbacks | None = None,
    ) -> None:
        self._runner = runner
        self._callbacks = callbacks or SessionCallbacks()
        self._task: asyncio.Task[AgentRun] | None = None
        self._closed = False

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def pending_confirmation(self) -> ConfirmationRequest | None:
        return self._runner.pending_confirmation

    async def ask(self, user_text: str) -> AgentRun:
        """Run one task end to end and deliver its result.

        Raises `RunBusyError` if one is already going; the active run is
        untouched by the refusal.
        """
        if self._closed:
            raise RunBusyError("die Sitzung ist beendet")
        run = self._runner.begin(user_text)
        self._emit_status(run.id, RunStatus.RUNNING, "working")
        try:
            finished = await self._runner.drive(run)
        except RunBusyError:
            raise
        except Exception:
            log.warning("harness run crashed", exc_info=False)
            self._emit_status(run.id, RunStatus.FAILED, "failed")
            self._speak(SPEECH_FAILED)
            raise
        self._deliver(finished)
        return finished

    def start(self, user_text: str) -> asyncio.Task[AgentRun]:
        """Fire and forget, for a caller that must not block."""
        task = asyncio.create_task(self.ask(user_text), name="harness-run")
        self._task = task
        return task

    def confirm(self, run_id: str, call_id: str, fingerprint: str) -> bool:
        """Approve a pending write. False when the approval no longer matches."""
        try:
            self._runner.confirm(run_id, call_id, fingerprint)
        except ConfirmationError as exc:
            log.info("confirmation refused: %s", exc.code)
            return False
        self._emit_status(run_id, RunStatus.RUNNING, "working")
        return True

    def reject(self, run_id: str, call_id: str) -> bool:
        try:
            self._runner.reject(run_id, call_id)
        except ConfirmationError as exc:
            log.info("rejection refused: %s", exc.code)
            return False
        return True

    @property
    def active_run_id(self) -> str | None:
        """The id of the currently executing run, or None if idle."""
        return self._runner.active_run_id

    def cancel(self, run_id: str | None = None) -> bool:
        """Cancel the active run. If `run_id` is given, it must match."""
        active_id = self._runner.active_run_id
        if active_id is None:
            return False
        if run_id is not None and run_id != active_id:
            return False
        return self._runner.cancel(active_id)

    def shutdown(self) -> None:
        """Window closing or app quitting: nothing pending may still be written."""
        self._closed = True
        self._runner.abandon_confirmation()
        self.cancel()

    # --- what the user sees and hears --------------------------------------

    def _deliver(self, run: AgentRun) -> None:
        if run.status is RunStatus.COMPLETED:
            text = (run.final_text or "").strip()
            self._emit_status(run.id, RunStatus.COMPLETED, "completed")
            self._emit_answer(text)
            self._speak(text)
            return
        if run.status is RunStatus.CANCELLED:
            # Nothing is spoken: a cancel is what the user asked for, and a
            # spoken "done" after an interruption is the worst possible answer.
            self._emit_status(run.id, RunStatus.CANCELLED, "cancelled")
            return
        if run.status is RunStatus.LIMIT_REACHED:
            self._emit_status(run.id, RunStatus.LIMIT_REACHED, "limit_reached")
            self._emit_answer("Das wurde mir zu verschachtelt.")
            self._speak("Das wurde mir zu verschachtelt.")
            return
        code = run.error_code or "tool_failed"
        line = _FAILURE_TEXT.get(code, SPEECH_FAILED)
        self._emit_status(
            run.id,
            RunStatus.CANCELLED if code == "confirmation_rejected" else RunStatus.FAILED,
            "cancelled" if code == "confirmation_rejected" else "failed",
        )
        self._emit_answer(line)
        self._speak(line)

    def announce_confirmation(self, request: ConfirmationRequest) -> None:
        """Called by the runner when a write is waiting for a person."""
        self._emit_status(request.run_id, RunStatus.NEEDS_CONFIRMATION, "needs_confirmation")
        callback = self._callbacks.on_confirmation
        if callback is not None:
            try:
                callback(request)
            except Exception:
                log.debug("confirmation presentation failed", exc_info=True)
        self._speak(SPEECH_NEEDS_CONFIRMATION)

    def _emit_status(
        self,
        run_id: str,
        status: RunStatus,
        message_code: str,
        *,
        terminal: bool | None = None,
    ) -> None:
        event = HarnessStatusEvent(
            run_id=run_id,
            status=status,
            message_code=message_code,
            terminal=status.is_terminal if terminal is None else terminal,
        )
        _safely(self._callbacks.on_status, event)

    def _emit_answer(self, text: str) -> None:
        if text:
            _safely(self._callbacks.on_answer, text)

    def _speak(self, text: str) -> None:
        if text:
            _safely(self._callbacks.on_speak, text)


def _safely(callback: Callable[[Any], None] | None, payload: Any) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        # A listener that throws must not take the run down with it.
        log.debug("session listener failed", exc_info=True)


def note_answer(data: dict[str, Any] | None) -> str:
    """The sentence KIKI says after a note was written. Name only, no path."""
    note = (data or {}).get("note")
    if isinstance(note, str) and note:
        return f"Die Notiz {note} ist angelegt."
    return SPEECH_NOTE_CREATED
