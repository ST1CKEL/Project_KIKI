"""Run a tool only after policy evaluation (and confirmation when needed)."""

from __future__ import annotations

import inspect
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from kiki.tools.audit import AuditLog, error_code, result_code
from kiki.tools.autonomy import sharpen, spec_for
from kiki.tools.confirmation import ConfirmationBroker, ConfirmationError
from kiki.tools.policy import DecisionKind, Origin, PolicyDecision, ToolPolicy
from kiki.tools.registry import ActionPreview, ToolRegistry, ToolSpec

log = logging.getLogger(__name__)

ConfirmFn = Callable[[ActionPreview], Awaitable[bool] | bool]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    decision: PolicyDecision
    data: dict[str, Any] | None = None
    error: str | None = None


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: ToolPolicy,
        audit: AuditLog,
        confirmations: ConfirmationBroker | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.audit = audit
        # Every confirmed action is redeemed against this broker just before it
        # runs. Default-constructed so no existing caller has to change.
        self.confirmations = confirmations or ConfirmationBroker()

    async def run(
        self,
        name: str,
        params: dict[str, Any] | None,
        *,
        panic: bool,
        integrations_enabled: bool,
        confirm: ConfirmFn | None = None,
        profile: str = "observe",
        origin: Origin = Origin.USER,
        run_id: str = "",
        call_id: str = "",
        panic_check: Callable[[], bool] | None = None,
        integrations_check: Callable[[], bool] | None = None,
    ) -> ToolResult:
        """Decide, possibly ask, then run — with the yes bound to this call.

        `run_id` and `call_id` are optional while the shared run model is still
        being introduced. Without them each invocation gets its own call id, so
        a confirmation is already tied to one execution; once runs exist the
        caller passes the real pair and the binding covers those too.

        `panic_check` and `integrations_check` are read again immediately before
        the side effect. `panic` and `integrations_enabled` are a snapshot the
        caller took, and a dialog can stand open for a long time — re-reading a
        snapshot proves nothing, so the live sources are what make a switch
        flipped during the dialog actually stop the action. Passing them is how
        `ToolGateway` closes that window; without them the old behaviour stands.
        """
        call_id = call_id or f"call-{uuid.uuid4().hex[:12]}"
        spec: ToolSpec | None = self.registry.get(name)
        decision = self.policy.evaluate(
            name=name,
            params=params,
            spec=spec,
            panic=panic,
            integrations_enabled=integrations_enabled,
            profile=profile,
            origin=origin,
        )
        # The tool-sharp limit on the level's blanket: before anything is
        # recorded or run, a jarvis-level model call the autonomy spec does
        # not cover becomes a card over the same validated parameters. It
        # runs before the audit entry so the log names the real decision.
        decision = sharpen(
            decision,
            name=name,
            spec=spec,
            origin=origin,
            autonomy=spec_for(self.policy.autonomy),
        )
        self.audit.record(
            name,
            params or {},
            decision.kind.value,
            spec=spec,
            result=f"[{origin.value}] {decision.reason}",
        )
        if decision.kind is DecisionKind.DENY:
            return ToolResult(ok=False, decision=decision, error=decision.reason)
        assert spec is not None
        cleaned = decision.params or {}
        grant = None
        preview = None
        if decision.kind is DecisionKind.CONFIRM:
            if confirm is None:
                self.audit.record(
                    name, cleaned, "denied", spec=spec, error="no_confirmation_ui"
                )
                return ToolResult(
                    ok=False,
                    decision=decision,
                    error="Bestätigung erforderlich, aber kein Dialog vorhanden.",
                )
            preview = self.registry.preview(spec, cleaned, decision.reason)
            # Registered before the dialog opens, so what is shown is what the
            # grant will be checked against.
            request = self.confirmations.request(
                run_id=run_id,
                call_id=call_id,
                spec=spec,
                arguments=cleaned,
                preview=preview,
            )
            # So a dialog can answer with the id it was handed, instead of
            # computing an authorisation value of its own.
            preview.request_id = request.id
            allowed = confirm(preview)
            if inspect.isawaitable(allowed):
                allowed = await allowed
            if not allowed:
                self.confirmations.reject(request.id)
                self.audit.record(name, cleaned, "cancelled", spec=spec)
                return ToolResult(ok=False, decision=decision, error="Vom Nutzer abgebrochen.")
            try:
                # The UI answered with a request id; the grant is minted here,
                # never carried through the interface.
                grant = self.confirmations.approve(request.id)
            except ConfirmationError as exc:
                self.audit.record(name, cleaned, "denied", spec=spec, error=exc.code)
                return ToolResult(
                    ok=False, decision=decision, error="Die Bestätigung gilt nicht mehr."
                )
            self.audit.record(name, cleaned, "confirmed", spec=spec)
        blocked = self._recheck(
            name=name,
            spec=spec,
            profile=profile,
            origin=origin,
            panic=panic,
            integrations_enabled=integrations_enabled,
            panic_check=panic_check,
            integrations_check=integrations_check,
        )
        if blocked is not None:
            # The world changed between the decision and the side effect. A
            # human "yes" is an authorisation, never a policy override.
            self.audit.record(name, cleaned, "denied", spec=spec, error="policy_recheck")
            return ToolResult(ok=False, decision=blocked, error=blocked.reason)
        if grant is not None:
            try:
                # Immediately before the side effect, against the values that
                # are really about to be used: an edit in between is impossible
                # rather than merely unlikely.
                self.confirmations.redeem(
                    grant,
                    run_id=run_id,
                    call_id=call_id,
                    spec=spec,
                    arguments=cleaned,
                    preview=preview,
                )
            except ConfirmationError as exc:
                self.audit.record(name, cleaned, "denied", spec=spec, error=exc.code)
                return ToolResult(
                    ok=False, decision=decision, error="Die Bestätigung gilt nicht mehr."
                )
        try:
            raw = spec.handler(cleaned)
            payload = await raw if inspect.isawaitable(raw) else raw
        except Exception as exc:
            # Not log.exception: the traceback carries the message, and with
            # it whatever the tool choked on — a path, a value, a token. The
            # class name is what a reader of the log can act on anyway.
            log.warning("tool %s failed: %s", name, type(exc).__name__)
            # The category, not the message: an exception quotes whatever it
            # choked on, which is regularly a path or a value.
            self.audit.record(name, cleaned, "error", spec=spec, error=error_code(exc))
            return ToolResult(ok=False, decision=decision, error=str(exc))
        if not isinstance(payload, dict):
            payload = {"result": payload}
        # Which fields came back, never their contents: a tool result can hold a
        # file, a note body or a network detail, and none of that belongs in a
        # long-lived security log.
        self.audit.record(name, cleaned, "executed", spec=spec, result=result_code(payload))
        return ToolResult(ok=True, decision=decision, data=payload)

    def _recheck(
        self,
        *,
        name: str,
        spec: ToolSpec | None,
        profile: str,
        origin: Origin,
        panic: bool,
        integrations_enabled: bool,
        panic_check: Callable[[], bool] | None,
        integrations_check: Callable[[], bool] | None,
    ) -> PolicyDecision | None:
        """Ask the policy again, with the world as it is now.

        Only the gate is re-run, not the whole evaluation: whether the action
        needs confirmation was already settled and paid for. What can still have
        changed is whether it is allowed at all — panic, the integration
        lockout, a hard deny, a profile switch.
        """
        if panic_check is None and integrations_check is None:
            return None
        live_panic = panic_check() if panic_check is not None else panic
        live_integrations = (
            integrations_check() if integrations_check is not None else integrations_enabled
        )
        return self.policy.gate(
            name=name,
            spec=spec,
            panic=live_panic,
            integrations_enabled=live_integrations,
            profile=profile,
            origin=origin,
        )
