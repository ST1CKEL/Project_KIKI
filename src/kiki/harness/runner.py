"""The loop: ask the model, run one tool, ask again, stop.

Bounded on two axes and cancellable at every boundary. One run at a time — a
second is refused rather than queued, and refusing does not touch the first.

The runner knows nothing about any particular model. It receives a
`ModelAdapter`, validates whatever comes back, and treats a malformed action as
`model_protocol_error` instead of letting it become a traceback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from kiki.harness.confirmation import (
    ConfirmationRequest,
    PendingConfirmation,
)
from kiki.harness.models import (
    ActionKind,
    AgentRun,
    CancelToken,
    ModelAction,
    RunStatus,
    ToolResult,
    validate_action,
)
from kiki.harness.tools import ToolRegistry
from kiki.harness.trace import TraceRecorder, TraceWriteError

# Small on purpose: this harness answers one question with at most one look at
# the system. Anything longer is a different product.
log = logging.getLogger(__name__)

MAX_TOOL_CALLS = 3
MAX_STEPS = 4


@runtime_checkable
class ModelAdapter(Protocol):
    """Turns the conversation so far into exactly one next action."""

    async def next_action(
        self,
        *,
        user_text: str,
        tool_schemas: list[dict[str, Any]],
        observations: list[ToolResult],
        cancel_token: CancelToken,
    ) -> ModelAction: ...


class RunBusyError(Exception):
    """A run is already active. The active one is untouched."""

    error_code = "run_busy"


class AgentRunner:
    def __init__(
        self,
        adapter: ModelAdapter,
        registry: ToolRegistry,
        *,
        trace_dir: Any,
        max_tool_calls: int = MAX_TOOL_CALLS,
        max_steps: int = MAX_STEPS,
        on_confirmation_required: Callable[[ConfirmationRequest], None] | None = None,
    ) -> None:
        if int(max_tool_calls) < 0 or int(max_steps) < 1:
            raise ValueError("Grenzen müssen sinnvoll sein")
        self._adapter = adapter
        self._registry = registry
        self._trace_dir = trace_dir
        self._max_tool_calls = int(max_tool_calls)
        self._max_steps = int(max_steps)
        self._on_confirmation = on_confirmation_required
        self._active: AgentRun | None = None
        self._token: CancelToken | None = None
        self._pending = PendingConfirmation()
        self._decision: asyncio.Event | None = None
        self._approved: ConfirmationRequest | None = None

    @property
    def active_run_id(self) -> str | None:
        return self._active.id if self._active is not None else None

    @property
    def busy(self) -> bool:
        return self._active is not None

    @property
    def pending_confirmation(self) -> ConfirmationRequest | None:
        return self._pending.pending

    def confirm(self, run_id: str, call_id: str, fingerprint: str) -> None:
        """Redeem exactly one approval. Raises `ConfirmationError` otherwise.

        The fingerprint is spent here, so a second press, a replayed dialog or a
        retried call cannot produce a second write.
        """
        request = self._pending.approve(run_id, call_id, fingerprint)
        self._approved = request
        self._release()

    def reject(self, run_id: str, call_id: str) -> None:
        self._pending.reject(run_id, call_id)
        self._approved = None
        self._release()

    def abandon_confirmation(self) -> None:
        """Shutdown or window close: whatever was waiting can never be written."""
        self._pending.clear()
        self._approved = None
        if self._token is not None:
            self._token.cancel()
        self._release()

    def _release(self) -> None:
        if self._decision is not None:
            self._decision.set()

    def cancel(self, run_id: str) -> bool:
        """Cancel one run by id. Unknown or finished ids do nothing.

        Idempotent, and deliberately narrow: it can only reach the run whose id
        was named, never whatever happens to be active.
        """
        if self._active is None or self._active.id != run_id or self._token is None:
            return False
        self._token.cancel()
        # A pending proposal dies with the run it belonged to.
        self._pending.clear()
        self._approved = None
        self._release()
        return True

    async def run(self, user_text: str) -> AgentRun:
        if self._active is not None:
            raise RunBusyError("ein Run läuft bereits")
        run = AgentRun(user_text=user_text)
        token = CancelToken()
        self._active = run
        self._token = token
        self._pending.clear()
        self._approved = None
        try:
            return await self._drive(run, token)
        finally:
            self._active = None
            self._token = None
            self._pending.clear()
            self._approved = None
            self._decision = None

    # --- internals ---------------------------------------------------------

    async def _drive(self, run: AgentRun, token: CancelToken) -> AgentRun:
        trace = TraceRecorder(self._trace_dir, run.id)
        began = time.monotonic()
        observations: list[ToolResult] = []
        schemas = self._registry.schemas()
        try:
            # Length only: the text itself is the one thing a trace never holds.
            trace.write(
                "run_started",
                user_text_length=len(run.user_text),
                tools=list(self._registry.names),
                max_steps=self._max_steps,
                max_tool_calls=self._max_tool_calls,
            )
            run.start()
            for step in range(self._max_steps + 1):
                if token.cancelled:
                    return self._cancel(run, trace, began)
                if step >= self._max_steps:
                    return self._stop(run, trace, began, RunStatus.LIMIT_REACHED, "step_limit")

                action = await self._adapter.next_action(
                    user_text=run.user_text,
                    tool_schemas=schemas,
                    observations=list(observations),
                    cancel_token=token,
                )
                if token.cancelled:
                    return self._cancel(run, trace, began)

                problem = validate_action(action)
                trace.write(
                    "model_action_received",
                    kind=getattr(getattr(action, "kind", None), "value", "invalid"),
                    valid=not problem,
                )
                if problem:
                    return self._stop(run, trace, began, RunStatus.FAILED, problem)

                if action.kind is ActionKind.FINAL:
                    run_text = action.final_text or ""
                    self._finish(run, trace, began, RunStatus.COMPLETED, final_text=run_text)
                    return run

                call = action.tool_call
                assert call is not None  # validate_action guaranteed it
                if run.tool_calls >= self._max_tool_calls:
                    return self._stop(
                        run, trace, began, RunStatus.LIMIT_REACHED, "tool_call_limit"
                    )

                problem = self._registry.validate(call)
                trace.write(
                    "tool_requested",
                    tool=call.name,
                    call_id=call.id,
                    # Shape, not content: for a write tool the arguments *are*
                    # the payload, and a note's text has no business in a trace.
                    arguments=_argument_shape(call.arguments),
                    accepted=not problem,
                )
                if problem:
                    return self._stop(run, trace, began, RunStatus.FAILED, problem)
                if token.cancelled:
                    # Checked again right here: a cancel that arrived while the
                    # model was thinking must not still cost a tool run.
                    return self._cancel(run, trace, began)

                tool = self._registry.get(call.name)
                if getattr(tool, "confirmation_required", False):
                    verdict = await self._ask_human(run, trace, call, tool, token)
                    if verdict is None:
                        return self._cancel(run, trace, began)
                    if verdict is False:
                        return self._stop(
                            run, trace, began, RunStatus.FAILED, "confirmation_rejected"
                        )

                result = await self._registry.execute(call)
                if getattr(tool, "confirmation_required", False) and result.ok:
                    trace.write("write_executed", tool=call.name, call_id=call.id)
                run.tool_calls += 1
                trace.write(
                    "tool_completed",
                    tool=result.name,
                    call_id=result.call_id,
                    ok=result.ok,
                    error_code=result.error_code,
                )
                if not result.ok:
                    return self._stop(
                        run, trace, began, RunStatus.FAILED, result.error_code or "tool_failed"
                    )
                observations.append(result)
            return self._stop(run, trace, began, RunStatus.LIMIT_REACHED, "step_limit")
        except TraceWriteError:
            # The one failure that cannot be traced. Settle the run honestly and
            # try once to say so; a second failure has nowhere left to go.
            if not run.is_terminal:
                run.finish(RunStatus.FAILED, error_code="trace_write_failed")
            _try_finished(trace, run, began)
            return run

    async def _ask_human(
        self,
        run: AgentRun,
        trace: TraceRecorder,
        call: Any,
        tool: Any,
        token: CancelToken,
    ) -> bool | None:
        """Pause the run until a person decides. True, False, or None for gone.

        The proposal is built from the tool itself, so what the user is shown is
        what the tool would write — not a paraphrase the model supplied.
        """
        try:
            target, content = tool.preview(dict(call.arguments))
        except Exception:
            trace.write("tool_completed", tool=call.name, call_id=call.id, ok=False,
                        error_code="invalid_arguments")
            return False
        request = ConfirmationRequest.build(
            run.id, call, title=tool.name, target=target, content=content
        )
        self._pending.arm(request)
        self._decision = asyncio.Event()
        run.await_confirmation()
        # Target and fingerprint only: the content is for the dialog, never the
        # trace.
        trace.write(
            "confirmation_requested",
            tool=call.name,
            call_id=call.id,
            target=target,
            content_chars=len(content),
            fingerprint=request.fingerprint,
        )
        if self._on_confirmation is not None:
            try:
                self._on_confirmation(request)
            except Exception:
                log.debug("confirmation listener failed", exc_info=True)
        await self._decision.wait()
        self._decision = None
        if token.cancelled or self._approved is None:
            if self._approved is None and not token.cancelled:
                trace.write("confirmation_rejected", tool=call.name, call_id=call.id)
                run.resume()
                return False
            return None
        approved = self._approved
        self._approved = None
        run.resume()
        trace.write(
            "confirmation_approved",
            tool=call.name,
            call_id=call.id,
            fingerprint=approved.fingerprint,
        )
        return True

    def _finish(
        self,
        run: AgentRun,
        trace: TraceRecorder,
        began: float,
        status: RunStatus,
        *,
        final_text: str | None = None,
        error_code: str | None = None,
    ) -> None:
        run.finish(status, final_text=final_text, error_code=error_code)
        trace.write(
            "run_finished",
            status=run.status.value,
            error_code=run.error_code,
            tool_calls=run.tool_calls,
            duration_ms=int((time.monotonic() - began) * 1000),
        )

    def _stop(
        self,
        run: AgentRun,
        trace: TraceRecorder,
        began: float,
        status: RunStatus,
        error_code: str,
    ) -> AgentRun:
        trace.write("run_failed", status=status.value, error_code=error_code)
        self._finish(run, trace, began, status, error_code=error_code)
        return run

    def _cancel(self, run: AgentRun, trace: TraceRecorder, began: float) -> AgentRun:
        trace.write("run_cancelled")
        self._finish(run, trace, began, RunStatus.CANCELLED)
        return run


def _argument_shape(arguments: dict[str, Any]) -> dict[str, int]:
    """Which arguments were passed and how big each was. Never their values."""
    return {str(key): len(str(value)) for key, value in list(arguments.items())[:20]}


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


def observations_summary(observations: Sequence[ToolResult]) -> list[dict[str, Any]]:
    """What an adapter is handed back. Data only, no exception text."""
    return [
        {"name": item.name, "ok": item.ok, "data": item.data, "error_code": item.error_code}
        for item in observations
    ]
