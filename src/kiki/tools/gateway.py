"""The one door every tool call goes through.

Not a second executor. The executor knows how to validate, confirm and run a
tool; the gateway knows *the world it runs in* — whether panic is on right now,
whether integrations are enabled right now, which profile is active. It holds
those as live sources rather than values, and hands them down so the executor
can ask again immediately before the side effect.

That is the point. A snapshot taken when a dialog opened says nothing about the
moment the action actually runs, and a confirmation dialog can stand open for a
long time:

    13:20:00  the card is shown
    13:20:05  the person approves
    13:20:06  privacy panic is switched on
    13:20:07  the tool wants to run     ← must not

A human "yes" is an authorisation. It is never a policy override.

Everything a caller needs is one `ToolInvocation`: what to run, with which
arguments, and on whose behalf. Model, routines, the PC-control UI and future
direct actions are meant to arrive here and nowhere else.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kiki.tools.executor import ConfirmFn, ToolExecutor, ToolResult
from kiki.tools.policy import Origin

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolInvocation:
    """One request to run one tool. Says who is asking, not whether they may."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    # Who produced this call. `Origin.MODEL` is the untrusted case and the
    # reason `model_callable` exists; `Origin.ROUTINE` is a standing
    # authorisation; `Origin.USER` is a person clicking something.
    actor: Origin = Origin.USER
    run_id: str = ""
    call_id: str = ""
    profile: str = "observe"


class ToolGateway:
    """The security facade in front of `ToolExecutor`.

    `panic_check` and `integrations_check` are callables on purpose. Passing
    booleans would reintroduce exactly the stale snapshot this exists to remove.
    """

    def __init__(
        self,
        executor: ToolExecutor,
        *,
        panic_check: Callable[[], bool],
        integrations_check: Callable[[], bool],
    ) -> None:
        self._executor = executor
        self._panic_check = panic_check
        self._integrations_check = integrations_check

    @property
    def executor(self) -> ToolExecutor:
        return self._executor

    @property
    def registry(self) -> Any:
        return self._executor.registry

    @property
    def policy(self) -> Any:
        return self._executor.policy

    @property
    def confirmations(self) -> Any:
        return self._executor.confirmations

    async def invoke(
        self,
        invocation: ToolInvocation,
        *,
        confirm: ConfirmFn | None = None,
    ) -> ToolResult:
        """Run one tool through validation, policy, confirmation and recheck.

        The gateway reads the world twice — once to decide, once again before
        the side effect — and never caches the answer in between.
        """
        return await self._executor.run(
            invocation.tool,
            dict(invocation.arguments),
            panic=self._panic_check(),
            integrations_enabled=self._integrations_check(),
            confirm=confirm,
            profile=invocation.profile,
            origin=invocation.actor,
            run_id=invocation.run_id,
            call_id=invocation.call_id,
            panic_check=self._panic_check,
            integrations_check=self._integrations_check,
        )

    def panic_now(self) -> bool:
        """The panic switch as it stands right now, not as it stood at startup."""
        return self._panic_check()

    def integrations_now(self) -> bool:
        """The integration lockout as it stands right now."""
        return self._integrations_check()

    def cancel_run(self, run_id: str) -> int:
        """A cancelled run takes its pending and granted authorisations with it."""
        return self._executor.confirmations.cancel_run(run_id)

    def shutdown(self) -> None:
        """Nothing outstanding may survive the process."""
        self._executor.confirmations.clear()

