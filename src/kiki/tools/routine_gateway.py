"""The one door for routine fires, and the authorization that guards it.

The routine engine holds the power of `Origin.ROUTINE`: tools whose author
allowed it run without a card, because a person confirmed the *recipe* once.
That power must never become a free pass. This module hands the engine an
object with the executor's `run()` shape -- the engine stays untouched --
but every call goes through the `ToolGateway` like every other execution,
with the live panic and integration sources, and only for a recipe the
repository holds, exactly as confirmed:

* the tool name and the canonical arguments must match a stored, enabled
  routine -- anything else is a fire nobody confirmed, and it is denied;
* a repository that cannot be read authorizes nothing: a damaged
  `RoutineAuthorization` fails closed, never open;
* the snapshots the engine passes are accepted and ignored -- the gateway
  re-reads the world through its live sources immediately before the side
  effect, so a switch flipped after the engine's tick still stops the fire.

A denial here is a `DecisionKind.DENY`, because a mismatched or unreadable
authorization is a permanent condition, not a transient one: the engine's
existing rule then disables the routine instead of re-denying it every tick.
"""

from __future__ import annotations

import logging
from typing import Any

from kiki.tools.confirmation import canonical_arguments
from kiki.tools.executor import ToolResult
from kiki.tools.gateway import ToolGateway, ToolInvocation
from kiki.tools.policy import DecisionKind, Origin, PolicyDecision

log = logging.getLogger(__name__)


class RoutineToolGateway:
    """Executor-shaped, gateway-backed, recipe-bound. For routine fires only."""

    def __init__(
        self,
        gateway: ToolGateway,
        recipes: Any,
        activity: Any = None,
        *,
        pause: Any = None,
    ) -> None:
        self._gateway = gateway
        self._recipes = recipes
        # Optional observation sink: what fired, as identifiers. Never content.
        self._activity = activity
        # Optional session gate: a paused assistant fires no routine. The
        # refusal is transient on purpose -- a pause is not a policy decision,
        # and the routine must survive it enabled.
        self._pause = pause

    async def run(
        self,
        name: str,
        params: dict[str, Any] | None,
        *,
        panic: bool = False,
        integrations_enabled: bool = True,
        origin: Origin = Origin.ROUTINE,
        **_unused: Any,
    ) -> ToolResult:
        del panic, integrations_enabled  # snapshots; the gateway reads it live
        if self._pause is not None and self._pause.paused:
            # Transient, deliberately not a DENY: the engine disables a
            # routine on policy denials, and a pause must not cost the user
            # their recipe. The tool never runs; the routine stays enabled.
            self._note("paused", name, "")
            return ToolResult(
                ok=False,
                decision=PolicyDecision(
                    kind=DecisionKind.CONFIRM,
                    reason="Die Assistentin macht Pause — die Routine bleibt aktiv.",
                ),
                error="Assistant pausiert.",
            )
        if origin is not Origin.ROUTINE:
            # This door is for routines. Anything else asking for their
            # card-free power is a wiring bug and gets the closed door.
            return self._refuse(
                name,
                params,
                "Dies ist die Routine-Tür — andere Aufrufer gehen durch den Gateway.",
            )
        try:
            authorized = self._authorized(name, params or {})
        except Exception:
            log.warning("routine authorization unreadable; refusing")
            return self._refuse(
                name, params, "Beschädigte Routineautorisierung — fail closed."
            )
        if authorized is None:
            self._note("refused", name, "")
            return self._refuse(
                name,
                params,
                "Keine aktive Routine deckt Werkzeug und Argumente — nicht bestätigt.",
            )
        run_id = f"routine-{authorized}"
        outcome = await self._gateway.invoke(
            ToolInvocation(
                tool=name,
                arguments=dict(params or {}),
                actor=Origin.ROUTINE,
                run_id=run_id,
                profile="observe",
            )
        )
        self._note("fired" if outcome.ok else "blocked", name, run_id)
        return outcome

    def _note(self, code: str, tool: str, run_id: str) -> None:
        if self._activity is not None:
            self._activity.record_routine(code=code, tool=tool, run_id=run_id)

    # --- the authorization --------------------------------------------------

    def _authorized(self, name: str, params: dict[str, Any]) -> str | None:
        """The routine id whose confirmed recipe covers exactly this fire.

        Tool name and canonical arguments must match a stored, enabled
        routine. A re-ordered dictionary is the same recipe; one changed
        character is not.
        """
        canonical = canonical_arguments(params)
        for routine in self._recipes.list():
            if not routine.enabled:
                continue
            if routine.tool_name != name:
                continue
            if canonical_arguments(routine.arguments) == canonical:
                return str(routine.id)
        return None

    def _refuse(self, name: str, params: dict[str, Any] | None, reason: str) -> ToolResult:
        decision = PolicyDecision(
            kind=DecisionKind.DENY,
            reason=reason,
        )
        executor = self._gateway.executor
        spec = executor.registry.get(name)
        executor.audit.record(
            name,
            params or {},
            "denied",
            spec=spec,
            error="routine_authorization",
        )
        return ToolResult(ok=False, decision=decision, error=reason)
