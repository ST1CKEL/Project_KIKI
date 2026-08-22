"""Read-only system snapshot tools. Invoked only by the user (status card)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kiki.integrations.base import Integration
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec


def _wrap(integration: Integration) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _handler(_params: dict[str, Any]) -> dict[str, Any]:
        snap = integration.snapshot()
        payload: dict[str, Any] = {"available": snap.available, **snap.data}
        if snap.error:
            payload["error"] = snap.error
        return payload

    return _handler


class SystemStatusSkill:
    id = "system_status"
    name = "Systemstatus"
    description = "Uhrzeit, Akku, Netzwerk und Speicherplatz — nur lesend."

    def __init__(self, integrations: list[Integration]) -> None:
        self._integrations = integrations

    def tools(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for integ in self._integrations:
            specs.append(
                ToolSpec(
                    name=f"status_{integ.id}",
                    title=integ.title,
                    description=f"Liest {integ.title} (read-only).",
                    risk=RiskLevel.READ,
                    parameters={"type": "object", "properties": {}, "additionalProperties": False},
                    handler=_wrap(integ),
                    effect=f"Liest den aktuellen Wert von {integ.title}. Keine Änderung am System.",
                    target="local",
                    auto_allow=True,
                    requires_integration=True,
                    model_callable=True,
                )
            )
        return specs
