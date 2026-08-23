"""The loop: ask the model, run one tool, ask again, stop.

Bounded on two axes and cancellable at every boundary. One run at a time — a
second is refused rather than queued, and refusing does not touch the first.

The runner knows nothing about any particular model. It receives a
`ModelAdapter`, validates whatever comes back, and treats a malformed action as
`model_protocol_error` instead of letting it become a traceback.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

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
    ) -> None:
        if int(max_tool_calls) < 0 or int(max_steps) < 1:
            raise ValueError("Grenzen müssen sinnvoll sein")
        self._adapter = adapter
        self._registry = registry
        self._trace_dir = trace_dir
        self._max_tool_calls = int(max_tool_calls)
        self._max_steps = int(max_steps)
        self._active: AgentRun | None = None
        self._token: CancelToken | None = None

    @property
    def active_run_id(self) -> str | None:
        return self._active.id if self._active is not None else None

    @property
    def busy(self) -> bool:
        return self._active is not None

    def cancel(self, run_id: str) -> bool:
        """Cancel one run by id. Unknown or finished ids do nothing.

        Idempotent, and deliberately narrow: it can only reach the run whose id
        was named, never whatever happens to be active.
        """
        if self._active is None or self._active.id != run_id or self._token is None:
            return False
        self._token.cancel()
        return True

    async def run(self, user_text: str) -> AgentRun:
        if self._active is not None:
            raise RunBusyError("ein Run läuft bereits")
        run = AgentRun(user_text=user_text)
        token = CancelToken()
        self._active = run
        self._token = token
        try:
            return await self._drive(run, token)
        finally:
            self._active = None
            self._token = None

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
                    arguments=call.arguments,
                    accepted=not problem,
                )
                if problem:
                    return self._stop(run, trace, began, RunStatus.FAILED, problem)
                if token.cancelled:
                    # Checked again right here: a cancel that arrived while the
                    # model was thinking must not still cost a tool run.
                    return self._cancel(run, trace, began)

                result = await self._registry.execute(call)
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
