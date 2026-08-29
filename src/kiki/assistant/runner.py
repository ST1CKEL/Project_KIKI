"""The one runner for assistant turns: run shell around model, tools, answer.

This is `AgentLoop` and `AgentRunner` unified. From the loop it takes the
streaming step machine -- text deltas reach the person while the model still
thinks, tool calls come back, the answer only counts when it is complete. From
the harness it takes the run shell -- one run with exactly one terminal state,
cooperative cancellation, a structured trace that never holds content, and a
confirmation pause that sits *inside* the tool call, where the gateway holds
the armed request.

What it deliberately does not have: GTK, a provider import, a tool handler
call, an opinion about which tools exist. The tool list is rebuilt from the
live policy before every model step and again before every execution, and
every execution goes through the `ToolGateway` -- the same door the chat path
uses, with `Origin.MODEL`.

Boundaries are the loop's, not the harness's: a run that needs more steps,
more calls or repeats itself than allowed ends visibly as `LIMIT_REACHED`,
never as a half answer. A model that misbehaves -- an invalid action, more
than one tool call in a stream, an adapter that crashes -- fails the run as
`model_protocol_error` instead of executing something plausible.

Tool refusals are different: `unknown_tool`, `invalid_arguments`, a rejected
confirmation or a policy denial never kill the run. They become the
observation the model sees next, so it can correct course. The category
reaches the model, never the executor's human-written reason.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from kiki.assistant.adapter import as_step_adapter
from kiki.harness.adapter import ProviderError
from kiki.harness.confirmation import ConfirmationError, ConfirmationRequest
from kiki.harness.models import (
    ERROR_CODES,
    ActionKind,
    AgentRun,
    CancelToken,
    HarnessStatusEvent,
    ModelAction,
    RunBusyError,
    RunStatus,
    ToolResult,
    validate_action,
)
from kiki.harness.trace import TraceRecorder, TraceWriteError
from kiki.tools.exposure import exposed_specs
from kiki.tools.gateway import ToolGateway, ToolInvocation
from kiki.tools.policy import Origin

log = logging.getLogger(__name__)

MAX_STEPS = 6
MAX_TOOL_CALLS = 12
RESULT_LIMIT = 8192

# The message codes a status event may carry, one per terminal state and one
# for each phase on the way there. Kept in step with HARNESS_MESSAGE_CODES.
_MESSAGE_FOR_STATUS: dict[RunStatus, str] = {
    RunStatus.COMPLETED: "completed",
    RunStatus.CANCELLED: "cancelled",
    RunStatus.FAILED: "failed",
    RunStatus.LIMIT_REACHED: "limit_reached",
}


class ModelProtocolFault(Exception):
    """The adapter produced something no protocol can accept."""


@dataclass(frozen=True)
class RunnerEvent:
    """One observable moment of a run. Text is either a model delta or empty.

    The runner never writes user-facing prose: consumers translate categories
    into sentences. `finished` carries the settled run and is always the last
    event a consumer sees.
    """

    kind: str  # delta | tool_start | tool_end | confirmation_requested | status | finished
    text: str = ""
    tool: str = ""
    ok: bool = True
    request: ConfirmationRequest | None = None
    status: HarnessStatusEvent | None = None
    run: AgentRun | None = None


class AssistantRunner:
    """One active run at a time, cancellable, observable, and gateway-gated."""

    def __init__(
        self,
        adapter: Any,
        gateway: ToolGateway,
        *,
        profile: str = "observe",
        trace_dir: Any,
        max_steps: int = MAX_STEPS,
        max_tool_calls: int = MAX_TOOL_CALLS,
        result_limit: int = RESULT_LIMIT,
    ) -> None:
        if int(max_tool_calls) < 0 or int(max_steps) < 1:
            raise ValueError("Grenzen müssen sinnvoll sein")
        self._adapter = as_step_adapter(adapter)
        self._gateway = gateway
        self._profile = profile
        self._trace_dir = trace_dir
        self._max_steps = int(max_steps)
        self._max_tool_calls = int(max_tool_calls)
        self._result_limit = int(result_limit)
        self._active: AgentRun | None = None
        self._token: CancelToken | None = None
        self._decision: asyncio.Event | None = None
        # The question currently on screen, and the answer to it. Neither is an
        # authorisation: the broker behind the gateway holds that.
        self._pending: ConfirmationRequest | None = None
        self._verdict: bool | None = None
        # Set only when a card was on screen and went away unanswered.
        self._abandoned = False
        # Whether the call currently executing was approved by a person. Trace
        # only; the gateway's grant is the authorisation, this is the record.
        self._approved_this_call = False

    # -- run shell -----------------------------------------------------------

    @property
    def active_run_id(self) -> str | None:
        return self._active.id if self._active is not None else None

    @property
    def busy(self) -> bool:
        return self._active is not None

    @property
    def pending_confirmation(self) -> ConfirmationRequest | None:
        return self._pending

    def begin(self, user_text: str) -> AgentRun:
        """Create and arm one run. Raises `RunBusyError` if one is active."""
        if self._active is not None:
            raise RunBusyError("ein Run läuft bereits")
        run = AgentRun(user_text=user_text)
        self._active = run
        self._token = CancelToken()
        self._pending = None
        self._verdict = None
        self._abandoned = False
        return run

    def drive(self, run: AgentRun) -> AsyncIterator[RunnerEvent]:
        """Consume one begun run; yields events, the last one is `finished`.

        The machine runs in its own task and feeds a queue, so a confirmation
        pause inside a tool call cannot stop the question from reaching the
        consumer. A consumer that walks away before `finished` cancels the
        run rather than leaving it half-alive.
        """
        if self._active is not run or self._token is None:
            raise RuntimeError("nur ein begonnener Run kann ausgeführt werden")
        token = self._token
        queue: asyncio.Queue[RunnerEvent] = asyncio.Queue()
        task = asyncio.create_task(self._drive_guarded(run, token, queue.put_nowait))

        async def _events() -> AsyncIterator[RunnerEvent]:
            try:
                while True:
                    event = await queue.get()
                    yield event
                    if event.kind == "finished":
                        return
            finally:
                if not task.done():
                    # The consumer left before the run settled. A zombie run
                    # would block every later one; a cancelled one is honest.
                    self._abort_pending()
                    token.cancel()
                    try:
                        await task
                    except Exception as exc:
                        log.warning("assistant run task failed: %s", type(exc).__name__)

        return _events()

    async def _drive_guarded(
        self,
        run: AgentRun,
        token: CancelToken,
        emit: Callable[[RunnerEvent], None],
    ) -> None:
        """Drive one run, and settle it even if the machine itself crashes.

        Every expected failure is caught on the way; this is the net for the
        one that is not. A crash is not a behaviour the run can honestly name,
        but a run that never settles is worse: the consumer would wait
        forever, and "exactly one terminal state" would be broken from below.
        The generic category says what is true -- something failed, nothing
        ran -- and the log keeps the class name.
        """
        try:
            await self._drive(run, token, emit)
        except Exception as exc:
            log.warning("assistant runner crashed: %s", type(exc).__name__)
            if not run.is_terminal:
                run.finish(RunStatus.FAILED, error_code="tool_failed")
            emit(_status_event(run.id, run.status, _MESSAGE_FOR_STATUS[run.status]))
            emit(RunnerEvent(kind="finished", run=run))
        finally:
            # _drive releases the shell itself; this covers a crash before
            # its cleanup was ever reached.
            self._release_run()

    async def run(self, user_text: str) -> AgentRun:
        """One turn, start to settle. For callers that only want the result."""
        run = self.begin(user_text)
        async for event in self.drive(run):
            if event.kind == "finished":
                break
        return run

    def cancel(self, run_id: str) -> bool:
        """Cancel one run by id. Unknown or finished ids do nothing.

        Idempotent, and deliberately narrow: it can only reach the run whose
        id was named, never whatever happens to be active.
        """
        if self._active is None or self._active.id != run_id or self._token is None:
            return False
        # A pending proposal dies with the run it belonged to.
        self._abort_pending()
        self._token.cancel()
        return True

    def confirm(self, run_id: str, call_id: str, request_id: str) -> None:
        """Answer the question on screen. Raises `ConfirmationError` otherwise.

        The answer is an id the UI was handed, not a value it worked out. What
        it buys is one pass through the gateway, where the broker mints a
        one-shot grant over the run, the call, the validated arguments and the
        card -- and refuses it if any of those moved in the meantime.
        """
        request = self._match(run_id, call_id)
        if request.request_id != request_id:
            # No exception for an empty id: "blank means skip" is how a check
            # like this turns into a way around it.
            raise ConfirmationError("confirmation_mismatch")
        self._pending = None
        self._verdict = True
        self._release()

    def reject(self, run_id: str, call_id: str) -> None:
        self._match(run_id, call_id)
        self._pending = None
        self._verdict = False
        self._release()

    def abandon_confirmation(self) -> None:
        """Shutdown or window close: whatever was waiting can never be written."""
        self._abort_pending()
        if self._token is not None:
            self._token.cancel()

    def _abort_pending(self) -> None:
        """Take a pending question away: the run is going, nobody answers it."""
        self._pending = None
        self._verdict = None
        if self._decision is not None:
            self._decision.set()

    def _release(self) -> None:
        if self._decision is not None:
            self._decision.set()

    def _match(self, run_id: str, call_id: str) -> ConfirmationRequest:
        request = self._pending
        if request is None:
            raise ConfirmationError("no_pending_confirmation")
        if request.run_id != run_id or request.call_id != call_id:
            raise ConfirmationError("confirmation_mismatch")
        return request

    # -- the machine ---------------------------------------------------------

    async def _drive(
        self,
        run: AgentRun,
        token: CancelToken,
        emit: Callable[[RunnerEvent], None],
    ) -> AgentRun:
        trace = TraceRecorder(self._trace_dir, run.id)
        began = time.monotonic()
        observations: list[ToolResult] = []
        seen_signatures: set[str] = set()
        try:
            trace.write(
                "run_started",
                user_text_length=len(run.user_text),
                tools=[spec.name for spec in self._exposed()],
                max_steps=self._max_steps,
                max_tool_calls=self._max_tool_calls,
            )
            run.start()
            emit(_status_event(run.id, RunStatus.RUNNING, "working"))
            for _step in range(self._max_steps):
                if token.cancelled:
                    return self._cancel(run, trace, began, emit)
                try:
                    action = await self._model_step(run, token, observations, emit)
                except ProviderError as exc:
                    return self._stop(
                        run, trace, began, emit, RunStatus.FAILED, _provider_code(exc)
                    )
                except ModelProtocolFault:
                    return self._stop(
                        run,
                        trace,
                        began,
                        emit,
                        RunStatus.FAILED,
                        "model_protocol_error",
                    )
                if token.cancelled:
                    return self._cancel(run, trace, began, emit)

                problem = validate_action(action)
                trace.write(
                    "model_action_received",
                    kind=getattr(getattr(action, "kind", None), "value", "invalid"),
                    valid=not problem,
                )
                if problem:
                    return self._stop(run, trace, began, emit, RunStatus.FAILED, problem)

                if action.kind is ActionKind.FINAL:
                    return self._complete(
                        run, trace, began, emit, action.final_text or ""
                    )

                settled = await self._tool_step(
                    run,
                    token,
                    trace,
                    began,
                    emit,
                    action.tool_call,
                    observations,
                    seen_signatures,
                )
                if settled is not None:
                    return settled
            return self._stop(
                run, trace, began, emit, RunStatus.LIMIT_REACHED, "step_limit"
            )
        except TraceWriteError:
            # The one failure that cannot be traced. Settle the run honestly,
            # try once to write that down, and tell the consumer anyway.
            if not run.is_terminal:
                run.finish(RunStatus.FAILED, error_code="trace_write_failed")
            _try_finished(trace, run, began)
            emit(_status_event(run.id, run.status, _MESSAGE_FOR_STATUS[run.status]))
            emit(RunnerEvent(kind="finished", run=run))
            return run
        finally:
            self._release_run()

    def _release_run(self) -> None:
        self._active = None
        self._token = None
        self._pending = None
        self._verdict = None
        self._abandoned = False
        self._decision = None

    async def _model_step(
        self,
        run: AgentRun,
        token: CancelToken,
        observations: list[ToolResult],
        emit: Callable[[RunnerEvent], None],
    ) -> ModelAction:
        """One model round: deltas stream out, one decision comes back.

        The tool list is rebuilt here, per step, from the live policy. A
        panic switch flipped mid-run removes the tools from the very next
        question instead of leaving them callable.
        """
        schemas = [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": dict(spec.parameters),
            }
            for spec in self._exposed()
        ]
        action: ModelAction | None = None
        try:
            async for event in self._adapter.next_action_stream(
                user_text=run.user_text,
                tool_schemas=schemas,
                observations=list(observations),
                cancel_token=token,
            ):
                if event.kind == "delta" and event.text:
                    emit(RunnerEvent(kind="delta", text=event.text))
                elif event.kind == "action":
                    if action is not None:
                        # The contract is exactly one decision per step. Two
                        # of them is the same protocol break as two tool
                        # calls in one stream -- refused, not picked over.
                        raise ModelProtocolFault
                    action = event.action
        except ProviderError as exc:
            raise ProviderError(_provider_code(exc)) from exc
        except Exception:
            # An adapter that crashes broke its protocol, not just a network.
            log.warning("assistant adapter step failed")
            raise ModelProtocolFault from None
        if action is None:
            # The step adapter's contract is exactly one action event; a
            # stream without one told the runner nothing it may act on.
            raise ModelProtocolFault
        return action

    async def _tool_step(
        self,
        run: AgentRun,
        token: CancelToken,
        trace: TraceRecorder,
        began: float,
        emit: Callable[[RunnerEvent], None],
        call: Any,
        observations: list[ToolResult],
        seen_signatures: set[str],
    ) -> AgentRun | None:
        """Validate, then run one call through the gateway.

        Returns a run only when this step ends the whole run (the call
        budget). Refusals and failures become observations; the model sees
        the category and the run goes on.
        """
        if run.tool_calls >= self._max_tool_calls:
            return self._stop(
                run, trace, began, emit, RunStatus.LIMIT_REACHED, "tool_call_limit"
            )

        # Read again, not remembered: what the policy hides between the
        # model's answer and this line does not exist, like any unknown tool.
        specs = {spec.name: spec for spec in self._exposed()}
        spec = specs.get(call.name)
        if spec is None:
            return self._refuse(
                run, trace, emit, call, observations, "unknown_tool"
            )
        problem = _arguments_problem(call.arguments, spec.parameters)
        if problem:
            return self._refuse(
                run, trace, emit, call, observations, problem
            )

        signature = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
        if signature in seen_signatures:
            # The first result still stands in the observations. Saying so
            # breaks the retry cycle without pretending the tool ran twice.
            return self._refuse(
                run,
                trace,
                emit,
                call,
                observations,
                "duplicate_call",
                duplicate=True,
            )
        seen_signatures.add(signature)

        if token.cancelled:
            # Checked again right here: a cancel that arrived while the model
            # was thinking must not still cost a tool run.
            return self._cancel(run, trace, began, emit)

        emit(RunnerEvent(kind="tool_start", tool=call.name, text=spec.title or call.name))
        emit(_status_event(run.id, RunStatus.RUNNING, "tool_running"))
        trace.write(
            "tool_requested",
            tool=call.name,
            call_id=call.id,
            # Shape, not content: for a write tool the arguments *are* the
            # payload, and a note's text has no business in a trace.
            arguments=_argument_shape(call.arguments),
            accepted=True,
        )

        self._approved_this_call = False
        outcome = await self._gateway.invoke(
            ToolInvocation(
                tool=call.name,
                arguments=dict(call.arguments),
                actor=Origin.MODEL,
                run_id=run.id,
                call_id=call.id,
                profile=self._profile,
            ),
            confirm=self._confirm_fn(run, trace, call, token, emit),
        )
        if self._abandoned:
            # The window closed while the card was up. A cancel that arrives
            # after the tool ran is not this: that one is caught at the next
            # model step, so the result still counts.
            self._abandoned = False
            return self._cancel(run, trace, began, emit)

        result = _as_run_result(call, outcome)
        run.tool_calls += 1
        trace.write(
            "tool_completed",
            tool=result.name,
            call_id=result.call_id,
            ok=result.ok,
            error_code=result.error_code,
        )
        emit(
            RunnerEvent(
                kind="tool_end",
                tool=result.name,
                ok=result.ok,
                text=result.error_code or "",
            )
        )
        if result.ok and self._approved_this_call:
            trace.write("write_executed", tool=result.name, call_id=result.call_id)
        observations.append(result)
        return None

    def _refuse(
        self,
        run: AgentRun,
        trace: TraceRecorder,
        emit: Callable[[RunnerEvent], None],
        call: Any,
        observations: list[ToolResult],
        error_code: str,
        *,
        duplicate: bool = False,
    ) -> AgentRun | None:
        """Record a refusal, hand the category to the model, keep the run."""
        trace.write(
            "tool_requested",
            tool=call.name,
            call_id=call.id,
            arguments=_argument_shape(getattr(call, "arguments", {}) or {}),
            accepted=False,
            reason=error_code,
        )
        if duplicate:
            observations.append(
                ToolResult(call_id=call.id, name=call.name, ok=True, data={"already_ran": True})
            )
        else:
            observations.append(
                ToolResult(
                    call_id=call.id, name=call.name, ok=False, error_code=error_code
                )
            )
        emit(RunnerEvent(kind="tool_end", tool=call.name, ok=False, text=error_code))
        return None

    def _confirm_fn(
        self,
        run: AgentRun,
        trace: TraceRecorder,
        call: Any,
        token: CancelToken,
        emit: Callable[[RunnerEvent], None],
    ) -> Any:
        """The callback the gateway invokes while it holds an armed request.

        The run pauses here, inside the tool call rather than in front of it,
        so there is one place where a person is asked and one place where that
        answer is spent. What the dialog shows is the registry's card, so the
        user sees what the tool would do, not a paraphrase the model supplied.
        """

        async def _ask(preview: Any) -> bool:
            request = ConfirmationRequest.from_preview(run.id, call.id, preview)
            self._pending = request
            self._verdict = None
            self._decision = asyncio.Event()
            run.await_confirmation()
            # Target and request id only: the content is for the dialog, never
            # the trace.
            trace.write(
                "confirmation_requested",
                tool=call.name,
                call_id=call.id,
                target=request.target,
                content_chars=len(request.content),
                request_id=request.request_id,
            )
            emit(_status_event(run.id, RunStatus.NEEDS_CONFIRMATION, "needs_confirmation"))
            emit(RunnerEvent(kind="confirmation_requested", request=request))
            await self._decision.wait()
            self._decision = None
            self._pending = None
            verdict = self._verdict
            self._verdict = None
            run.resume()
            emit(_status_event(run.id, RunStatus.RUNNING, "working"))
            if token.cancelled or verdict is None:
                # Abandoned. Refusing here means the gateway never mints a
                # grant, so nothing is left armed behind the closed window.
                self._abandoned = True
                return False
            if verdict is False:
                trace.write("confirmation_rejected", tool=call.name, call_id=call.id)
                return False
            self._approved_this_call = True
            trace.write(
                "confirmation_approved",
                tool=call.name,
                call_id=call.id,
                request_id=request.request_id,
            )
            return True

        return _ask

    # -- settling ------------------------------------------------------------

    def _complete(
        self,
        run: AgentRun,
        trace: TraceRecorder,
        began: float,
        emit: Callable[[RunnerEvent], None],
        final_text: str,
    ) -> AgentRun:
        run.finish(RunStatus.COMPLETED, final_text=final_text)
        trace.write(
            "run_finished",
            status=run.status.value,
            error_code=None,
            tool_calls=run.tool_calls,
            duration_ms=int((time.monotonic() - began) * 1000),
        )
        emit(_status_event(run.id, RunStatus.COMPLETED, "completed"))
        emit(RunnerEvent(kind="finished", run=run))
        return run

    def _stop(
        self,
        run: AgentRun,
        trace: TraceRecorder,
        began: float,
        emit: Callable[[RunnerEvent], None],
        status: RunStatus,
        error_code: str,
    ) -> AgentRun:
        trace.write("run_failed", status=status.value, error_code=error_code)
        return self._finish(run, trace, began, emit, status, error_code=error_code)

    def _cancel(
        self,
        run: AgentRun,
        trace: TraceRecorder,
        began: float,
        emit: Callable[[RunnerEvent], None],
    ) -> AgentRun:
        trace.write("run_cancelled")
        return self._finish(run, trace, began, emit, RunStatus.CANCELLED)

    def _finish(
        self,
        run: AgentRun,
        trace: TraceRecorder,
        began: float,
        emit: Callable[[RunnerEvent], None],
        status: RunStatus,
        *,
        final_text: str | None = None,
        error_code: str | None = None,
    ) -> AgentRun:
        run.finish(status, final_text=final_text, error_code=error_code)
        trace.write(
            "run_finished",
            status=run.status.value,
            error_code=run.error_code,
            tool_calls=run.tool_calls,
            duration_ms=int((time.monotonic() - began) * 1000),
        )
        emit(_status_event(run.id, status, _MESSAGE_FOR_STATUS[status]))
        emit(RunnerEvent(kind="finished", run=run))
        return run

    # -- what the model may see, right now -----------------------------------

    def _exposed(self) -> list[Any]:
        return exposed_specs(
            self._gateway.registry,
            self._gateway.policy,
            panic=self._gateway.panic_now(),
            integrations_enabled=self._gateway.integrations_now(),
            profile=self._profile,
        )


def _provider_code(exc: ProviderError) -> str:
    """The adapter's category, if it is one the run may report."""
    code = getattr(exc, "code", "")
    return code if code in ERROR_CODES else "provider_error"


def _as_run_result(call: Any, outcome: Any) -> ToolResult:
    """The executor's outcome as a run observation: data, or one category."""
    if outcome.ok:
        return ToolResult(call_id=call.id, name=call.name, ok=True, data=outcome.data)
    kind = getattr(getattr(outcome, "decision", None), "kind", None)
    if kind is not None and kind.name == "DENY":
        # The policy refused it. From where the model sits that is the same as
        # a tool that is not there, which is all it needs to change course.
        return ToolResult(
            call_id=call.id, name=call.name, ok=False, error_code="tool_unavailable"
        )
    if kind is not None and kind.name == "CONFIRM":
        # Rejected, expired or spent: nothing ran, and the model gets the word.
        return ToolResult(
            call_id=call.id,
            name=call.name,
            ok=False,
            error_code="confirmation_rejected",
        )
    return ToolResult(call_id=call.id, name=call.name, ok=False, error_code="tool_failed")


def _status_event(run_id: str, status: RunStatus, message_code: str) -> RunnerEvent:
    return RunnerEvent(
        kind="status",
        status=HarnessStatusEvent(
            run_id=run_id,
            status=status,
            message_code=message_code,
            terminal=status.is_terminal,
        ),
    )


def _argument_shape(arguments: dict[str, Any]) -> dict[str, int]:
    """Which arguments were passed and how big each was. Never their values."""
    return {str(key): len(str(value)) for key, value in list(arguments.items())[:20]}


def _arguments_problem(arguments: Any, schema: dict[str, Any]) -> str:
    """A deliberately small subset of JSON Schema: objects, required, no extras."""
    if not isinstance(arguments, dict):
        return "invalid_arguments"
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    if schema.get("additionalProperties", False) is False:
        if set(arguments) - set(properties):
            return "invalid_arguments"
    required = schema.get("required")
    required = required if isinstance(required, list | tuple) else ()
    if set(required) - set(arguments):
        return "invalid_arguments"
    return ""


def _try_finished(trace: TraceRecorder, run: AgentRun, began: float) -> None:
    try:
        trace.write(
            "run_finished",
            status=run.status.value,
            error_code=run.error_code,
            tool_calls=run.tool_calls,
            duration_ms=int((time.monotonic() - began) * 1000),
        )
    except TraceWriteError:
        pass
