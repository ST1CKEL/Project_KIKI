"""The application's face of a run: categories in, short German out.

Sits between `AssistantRunner` and whatever shows a run to a person. The
runner speaks in events and error categories -- the vocabulary of
`ERROR_CODES`, never a message -- because a message can carry a path, a prompt
or a token and a category cannot. This module is where those categories become
sentences: one fixed line per failure, the model's answer when the run
completed, and silence when the user cancelled, because a spoken "done" after
an interruption is the worst possible answer.

The service owns no GTK, no bus, no storage: four optional callbacks, all
invoked with short, already-sanitised German. Threading stays the
application's business -- this module works on the asyncio side and hands
finished strings over.

Two promises carried over from the harness session, adjusted for a runner
that streams events instead of calling back:

* a confirmation nobody can present is *rejected*, never waited for. With the
  old runner a throwing listener hung the run forever; with the new one the
  question arrives as an event, and an event nobody answers is an event
  nobody was asked -- so the service refuses it, the gateway never mints a
  grant, and the run carries on with the refusal as its observation;
* a listener that throws never takes the run down with it.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from kiki.assistant.runner import AssistantRunner, RunnerEvent
from kiki.harness.confirmation import ConfirmationError, ConfirmationRequest
from kiki.harness.models import AgentRun, HarnessStatusEvent, RunBusyError, RunStatus
from kiki.runtime.pause import AssistantPause

log = logging.getLogger(__name__)

class RunPausedError(RunBusyError):
    """The assistant is paused. No new work until it is resumed.

    A subclass of `RunBusyError` on purpose: existing callers that treat
    "no run now" as one case keep working; those that want to say why can
    tell them apart. The active run, if any, is untouched.
    """

    error_code = "run_paused"


# How many recent spoken-event ids stay remembered. A duplicate delivery of
# an utterance arrives moments after the original, not hours later; the bound
# keeps a long session from growing the set without end.
CORRELATION_MEMORY = 256

SPEECH_NEEDS_CONFIRMATION = (
    "Ich habe eine Aktion vorbereitet. Bitte bestätige sie in der Oberfläche."
)
SPEECH_FAILED = "Das hat nicht geklappt."
LIMIT_TEXT = "Das wurde mir zu verschachtelt."

# One line per failure category. A category the user never has to decode, and
# never the underlying message. Codes the runner cannot produce any more (a
# tool refusal no longer fails a run, it becomes an observation) are absent
# on purpose: dead branches here would suggest guarantees that do not exist.
_FAILURE_TEXT: dict[str, str] = {
    "model_protocol_error": "Ich habe mich vertan.",
    "provider_error": "Ich konnte das Modell nicht erreichen.",
    "step_limit": LIMIT_TEXT,
    "tool_call_limit": LIMIT_TEXT,
    "trace_write_failed": "Ich konnte den Ablauf nicht mitschreiben.",
    "tool_failed": "Das hat nicht funktioniert.",
}


def failure_text(error_code: str | None) -> str:
    """The one sentence a person gets for a failure category. Nothing else."""
    return _FAILURE_TEXT.get(error_code or "", SPEECH_FAILED)


def with_tool_note(answer: str, used_tools: list[str]) -> str:
    """The transcript line that keeps an answer auditable.

    The same convention the chat path has always had: what the person reads
    says which tools produced the number. It belongs to the transcript only
    -- never to speech, where a bracketed tool list is noise.
    """
    if not used_tools:
        return answer
    seen: list[str] = []
    for name in used_tools:
        if name not in seen:
            seen.append(name)
    return f"{answer}\n\n_[KIKI hat benutzt: {', '.join(seen)}]_"


class DuplicateCorrelationError(Exception):
    """The same spoken event asked twice. Its first run *is* the answer.

    Raised before anything runs, so a duplicate delivery can never create a
    second run for one utterance -- and never disturbs the run it belongs to.
    """

    error_code = "duplicate_correlation"


@dataclass(frozen=True)
class RunCallbacks:
    """Everything that leaves the service. All optional, all sanitised."""

    on_status: Callable[[HarnessStatusEvent], None] | None = None
    on_answer: Callable[[str], None] | None = None
    on_confirmation: Callable[[ConfirmationRequest], None] | None = None
    on_speak: Callable[[str], None] | None = None


class RunService:
    """One runner, one active run, and the text the user sees and hears."""

    def __init__(
        self,
        runner: AssistantRunner,
        callbacks: RunCallbacks | None = None,
        *,
        paused: AssistantPause | None = None,
    ) -> None:
        self._runner = runner
        self._callbacks = callbacks or RunCallbacks()
        self._closed = False
        # Optional session gate: paused refuses new work at the structural
        # door, so no caller can forget the check.
        self._paused = paused
        # Spoken-event ids that already have (or had) their run, oldest first.
        self._correlations: OrderedDict[str, None] = OrderedDict()

    @property
    def busy(self) -> bool:
        """The runner is the authority: it releases the shell when the run
        settles, so this is true for an awaited `ask` and a started task
        alike."""
        return self._runner.busy

    @property
    def pending_confirmation(self) -> ConfirmationRequest | None:
        return self._runner.pending_confirmation

    @property
    def active_run_id(self) -> str | None:
        return self._runner.active_run_id

    async def ask(
        self,
        user_text: str,
        *,
        correlation_id: str = "",
    ) -> AgentRun:
        """Run one task end to end, translating its events on the way.

        `correlation_id` names the event that asked for this run -- one spoken
        utterance, one id. The same id can start at most one run, ever: a
        duplicate delivery raises `DuplicateCorrelationError` before anything
        begins. Raises `RunPausedError` when the assistant is paused,
        `RunBusyError` if a run is already going or the service is shut down;
        the active run is untouched by any refusal, and a refusal does not
        burn the id -- an utterance refused because KIKI was busy or paused
        may be said again.
        """
        if self._closed:
            raise RunBusyError("die Sitzung ist beendet")
        if self._paused is not None and self._paused.paused:
            raise RunPausedError("KIKI macht gerade Pause")
        if correlation_id and correlation_id in self._correlations:
            raise DuplicateCorrelationError(correlation_id)
        run = self._runner.begin(user_text)
        if correlation_id:
            self._remember(correlation_id)
        try:
            async for event in self._runner.drive(run):
                self._handle(event)
        except RunBusyError:
            raise
        except Exception:
            # The runner settles its own crashes; this net is for the
            # impossible case. It must not stay silent about it.
            log.warning("assistant run crashed", exc_info=False)
            self._emit_status(run.id, RunStatus.FAILED, "failed")
            self._speak(SPEECH_FAILED)
            raise
        return run

    def _remember(self, correlation_id: str) -> None:
        """One id, one run: remembered when its run begins, oldest evicted."""
        self._correlations[correlation_id] = None
        while len(self._correlations) > CORRELATION_MEMORY:
            self._correlations.popitem(last=False)

    def confirm(self, run_id: str, call_id: str, request_id: str) -> bool:
        """Approve a pending write. False when the approval no longer matches."""
        try:
            self._runner.confirm(run_id, call_id, request_id)
        except ConfirmationError as exc:
            log.info("confirmation refused: %s", exc.code)
            return False
        return True

    def reject(self, run_id: str, call_id: str) -> bool:
        try:
            self._runner.reject(run_id, call_id)
        except ConfirmationError as exc:
            log.info("rejection refused: %s", exc.code)
            return False
        return True

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

    # --- event translation --------------------------------------------------

    def _handle(self, event: RunnerEvent) -> None:
        if event.kind == "status":
            _safely(self._callbacks.on_status, event.status)
        elif event.kind == "confirmation_requested":
            self._present(event.request)
        elif event.kind == "finished":
            self._deliver(event.run)
        # delta, tool_start, tool_end: the run bar does not show them. The
        # settled answer is what the person reads; the model's preamble is
        # not an answer and never becomes one.

    def _present(self, request: ConfirmationRequest | None) -> None:
        if request is None:
            return
        callback = self._callbacks.on_confirmation
        if callback is not None:
            try:
                callback(request)
            except Exception:
                log.debug("confirmation presentation failed", exc_info=True)
            else:
                self._speak(SPEECH_NEEDS_CONFIRMATION)
                return
        # Nobody was asked -- no callback, or the one there broke. Nobody
        # will answer either, and waiting would hang the run behind a card
        # that never came up. Refusing means the gateway never mints a
        # grant, so nothing runs unasked.
        self._refuse_pending(request)

    def _refuse_pending(self, request: ConfirmationRequest) -> None:
        try:
            self._runner.reject(request.run_id, request.call_id)
        except ConfirmationError:
            # The question vanished while it was queued -- the run was
            # cancelled or settled. Nothing to refuse, nothing to do.
            log.debug("pending confirmation vanished before refusal")

    def _deliver(self, run: AgentRun | None) -> None:
        if run is None:
            return
        if run.status is RunStatus.COMPLETED:
            text = (run.final_text or "").strip()
            self._answer(text)
            self._speak(text)
            return
        if run.status is RunStatus.CANCELLED:
            # Nothing is spoken and nothing answered: a cancel is what the
            # user asked for.
            return
        if run.status is RunStatus.LIMIT_REACHED:
            self._answer(LIMIT_TEXT)
            self._speak(LIMIT_TEXT)
            return
        line = failure_text(run.error_code)
        self._answer(line)
        self._speak(line)

    # --- the four exits -----------------------------------------------------

    def _emit_status(self, run_id: str, status: RunStatus, message_code: str) -> None:
        _safely(
            self._callbacks.on_status,
            HarnessStatusEvent(
                run_id=run_id,
                status=status,
                message_code=message_code,
                terminal=status.is_terminal,
            ),
        )

    def _answer(self, text: str) -> None:
        if text:
            _safely(self._callbacks.on_answer, text)

    def _speak(self, text: str) -> None:
        if text:
            _safely(self._callbacks.on_speak, text)


def _safely(callback: Callable[[object], None] | None, payload: object) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        # A listener that throws must not take the run down with it.
        log.debug("run service listener failed", exc_info=True)
