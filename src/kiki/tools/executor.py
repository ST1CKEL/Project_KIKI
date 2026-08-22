"""Run a tool only after policy evaluation (and confirmation when needed)."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from kiki.tools.audit import AuditLog
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
    def __init__(self, registry: ToolRegistry, policy: ToolPolicy, audit: AuditLog) -> None:
        self.registry = registry
        self.policy = policy
        self.audit = audit

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
    ) -> ToolResult:
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
        self.audit.record(
            name, params or {}, decision.kind.value, result=f"[{origin.value}] {decision.reason}"
        )
        if decision.kind is DecisionKind.DENY:
            return ToolResult(ok=False, decision=decision, error=decision.reason)
        assert spec is not None
        cleaned = decision.params or {}
        if decision.kind is DecisionKind.CONFIRM:
            if confirm is None:
                self.audit.record(name, cleaned, "denied", error="no confirmation UI")
                return ToolResult(
                    ok=False,
                    decision=decision,
                    error="Bestätigung erforderlich, aber kein Dialog vorhanden.",
                )
            preview = self.registry.preview(spec, cleaned, decision.reason)
            allowed = confirm(preview)
            if inspect.isawaitable(allowed):
                allowed = await allowed
            if not allowed:
                self.audit.record(name, cleaned, "cancelled")
                return ToolResult(ok=False, decision=decision, error="Vom Nutzer abgebrochen.")
            self.audit.record(name, cleaned, "confirmed")
        try:
            raw = spec.handler(cleaned)
            payload = await raw if inspect.isawaitable(raw) else raw
        except Exception as exc:
            log.exception("tool %s failed", name)
            self.audit.record(name, cleaned, "error", error=str(exc))
            return ToolResult(ok=False, decision=decision, error=str(exc))
        if not isinstance(payload, dict):
            payload = {"result": payload}
        self.audit.record(
            name,
            cleaned,
            "executed",
            result=json.dumps(payload, ensure_ascii=False, default=str)[:2000],
        )
        return ToolResult(ok=True, decision=decision, data=payload)
