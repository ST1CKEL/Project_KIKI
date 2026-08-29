"""Power tools: logind mapping, failure handling, and the risk split."""

from __future__ import annotations

from typing import Any

import pytest

from kiki.platform.dbus import BusError
from kiki.tools.policy import AutonomyLevel, DecisionKind, Origin, RiskLevel, ToolPolicy
from kiki.tools.power_tools import LogindClient, PowerControlSkill


class FakeLogind:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[str] = []

    def suspend(self) -> None:
        self._record("Suspend")

    def reboot(self) -> None:
        self._record("Reboot")

    def power_off(self) -> None:
        self._record("PowerOff")

    def _record(self, name: str) -> None:
        if self.error:
            raise self.error
        self.calls.append(name)


def _skill(client: FakeLogind) -> dict[str, Any]:
    return {spec.name: spec for spec in PowerControlSkill(client).tools()}


def test_suspend_is_control_and_unattended_in_balanced() -> None:
    spec = _skill(FakeLogind())["power.suspend"]
    assert spec.risk is RiskLevel.CONTROL
    decision = ToolPolicy(AutonomyLevel.BALANCED.value).evaluate(
        name=spec.name,
        params={},
        spec=spec,
        panic=False,
        integrations_enabled=True,
        origin=Origin.MODEL,
    )
    assert decision.kind is DecisionKind.ALLOW


@pytest.mark.parametrize("slug", ["reboot", "poweroff"])
def test_destructive_actions_are_write_card_outside_jarvis(slug: str) -> None:
    spec = _skill(FakeLogind())[f"power.{slug}"]
    assert spec.risk is RiskLevel.WRITE
    balanced = ToolPolicy(AutonomyLevel.BALANCED.value).evaluate(
        name=spec.name,
        params={},
        spec=spec,
        panic=False,
        integrations_enabled=True,
        origin=Origin.MODEL,
    )
    assert balanced.kind is DecisionKind.CONFIRM
    jarvis = ToolPolicy(AutonomyLevel.JARVIS.value).evaluate(
        name=spec.name,
        params={},
        spec=spec,
        panic=False,
        integrations_enabled=True,
        origin=Origin.MODEL,
    )
    assert jarvis.kind is DecisionKind.ALLOW


def test_handlers_call_the_right_logind_method() -> None:
    client = FakeLogind()
    skill = _skill(client)
    assert skill["power.suspend"].handler({}) == {"ok": True, "action": "suspend"}
    assert skill["power.reboot"].handler({}) == {"ok": True, "action": "reboot"}
    assert skill["power.poweroff"].handler({}) == {"ok": True, "action": "power_off"}
    assert client.calls == ["Suspend", "Reboot", "PowerOff"]


def test_bus_error_is_a_clean_tool_result() -> None:
    skill = _skill(FakeLogind(error=BusError("logind verweigert")))
    result = skill["power.suspend"].handler({})
    assert result["ok"] is False
    assert "logind" in result["error"]


def test_client_maps_actions_to_logind_methods() -> None:
    calls: list[str] = []

    def caller(method: str) -> None:
        calls.append(method)

    client = LogindClient(caller)
    client.suspend()
    client.reboot()
    client.power_off()
    assert calls == ["Suspend", "Reboot", "PowerOff"]
