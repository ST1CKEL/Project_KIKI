"""Screen lock tool: dialect cascade and clean unavailability."""

from __future__ import annotations

import pytest

from kiki.tools.policy import AutonomyLevel, DecisionKind, Origin, RiskLevel, ToolPolicy
from kiki.tools.session_tools import ScreenLockClient, ScreenLockError, SessionControlSkill


def _skill(client: ScreenLockClient) -> dict[str, object]:
    return {spec.name: spec for spec in SessionControlSkill(client).tools()}


def test_lock_prefers_freedesktop_name() -> None:
    calls: list[tuple[str, str, str]] = []

    def caller(destination: str, path: str, interface: str) -> None:
        calls.append((destination, path, interface))

    result = _skill(ScreenLockClient(caller))["session.lock"].handler({})
    assert result == {"ok": True, "locked_via": "org.freedesktop.ScreenSaver"}
    assert len(calls) == 1
    assert calls[0][2] == "org.freedesktop.ScreenSaver"


def test_lock_falls_back_to_gnome_name() -> None:
    def caller(destination: str, _path: str, _interface: str) -> None:
        if destination == "org.freedesktop.ScreenSaver":
            raise RuntimeError("ServiceUnknown")

    result = _skill(ScreenLockClient(caller))["session.lock"].handler({})
    assert result == {"ok": True, "locked_via": "org.gnome.ScreenSaver"}


def test_lock_reports_failure_when_nobody_answers() -> None:
    def caller(_destination: str, _path: str, _interface: str) -> None:
        raise RuntimeError("ServiceUnknown")

    result = _skill(ScreenLockClient(caller))["session.lock"].handler({})
    assert result["ok"] is False
    assert "nicht erreichbar" in result["error"]


def test_client_lock_raises_when_all_targets_fail() -> None:
    def caller(_destination: str, _path: str, _interface: str) -> None:
        raise RuntimeError("kaputt")

    with pytest.raises(ScreenLockError):
        ScreenLockClient(caller).lock()


def test_lock_is_control_and_unattended_in_balanced() -> None:
    spec = _skill(ScreenLockClient())["session.lock"]
    assert spec.risk is RiskLevel.CONTROL
    assert spec.model_callable is True
    assert spec.auto_allow is True
    decision = ToolPolicy(AutonomyLevel.BALANCED.value).evaluate(
        name=spec.name,
        params={},
        spec=spec,
        panic=False,
        integrations_enabled=True,
        origin=Origin.MODEL,
    )
    assert decision.kind is DecisionKind.ALLOW
