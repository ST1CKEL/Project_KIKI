"""Network tools: SSID decoding, AP ranking, VPN uuid resolution, policy."""

from __future__ import annotations

from typing import Any

import pytest

from kiki.tools.network_tools import (
    NetworkControlSkill,
    NetworkError,
    decode_ssid,
)
from kiki.tools.policy import AutonomyLevel, DecisionKind, Origin, RiskLevel, ToolPolicy


class FakeNMClient:
    def __init__(
        self,
        *,
        points: list[dict[str, Any]] | None = None,
        vpns: list[dict[str, str]] | None = None,
        wifi_enabled: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.points = points or []
        self.vpns = vpns or []
        self._wifi_enabled = wifi_enabled
        self.error = error
        self.calls: list[str] = []

    def wifi_enabled(self) -> bool:
        return self._wifi_enabled

    def set_wifi_enabled(self, enabled: bool) -> None:
        if self.error:
            raise self.error
        self.calls.append(f"wifi:{enabled}")
        self._wifi_enabled = enabled

    def wifi_access_points(self) -> list[dict[str, Any]]:
        if self.error:
            raise self.error
        return list(self.points)

    def vpn_connections(self) -> list[dict[str, str]]:
        if self.error:
            raise self.error
        return list(self.vpns)

    def activate_connection(self, uuid: str) -> str:
        if self.error:
            raise self.error
        if not any(v["uuid"] == uuid for v in self.vpns):
            raise NetworkError(f"Keine VPN-Verbindung mit UUID „{uuid}“ gefunden.")
        self.calls.append(f"activate:{uuid}")
        return "/connections/vpn1"

    def deactivate_connection(self, uuid: str) -> None:
        if self.error:
            raise self.error
        self.calls.append(f"deactivate:{uuid}")


def _skill(client: FakeNMClient) -> dict[str, Any]:
    return {spec.name: spec for spec in NetworkControlSkill(client).tools()}


def test_decode_ssid_handles_bytes_empty_and_strings() -> None:
    assert decode_ssid(b"Mein WLAN") == "Mein WLAN"
    assert decode_ssid([72, 101]) == "He"
    assert decode_ssid(b"") is None  # hidden network
    assert decode_ssid("  ") is None
    assert decode_ssid(42) is None


def test_wifi_list_ranks_and_reports_state() -> None:
    client = FakeNMClient(
        points=[
            {"ssid": "Schwach", "strength": 20, "secure": True},
            {"ssid": "Stark", "strength": 90, "secure": False},
        ]
    )
    result = _skill(client)["network.wifi_list"].handler({})
    assert result["ok"] is True
    assert result["wifi_enabled"] is True
    assert result["networks"][0]["ssid"] == "Stark"


def test_wifi_list_reports_bus_error() -> None:
    client = FakeNMClient(error=NetworkError("NetworkManager nicht erreichbar"))
    result = _skill(client)["network.wifi_list"].handler({})
    assert result["ok"] is False
    assert "nicht erreichbar" in result["error"]


def test_wifi_set_toggles_radio() -> None:
    client = FakeNMClient()
    result = _skill(client)["network.wifi_set"].handler({"enabled": False})
    assert result == {"ok": True, "wifi_enabled": False}
    assert client.calls == ["wifi:False"]


def test_vpn_list_returns_connections() -> None:
    client = FakeNMClient(vpns=[{"id": "Firma", "uuid": "u-1", "type": "wireguard"}])
    result = _skill(client)["network.vpn_list"].handler({})
    assert result["count"] == 1
    assert result["connections"][0]["uuid"] == "u-1"


def test_vpn_connect_uses_uuid() -> None:
    client = FakeNMClient(vpns=[{"id": "Firma", "uuid": "u-1", "type": "vpn"}])
    assert _skill(client)["network.vpn_connect"].handler({"uuid": "u-1"})["ok"] is True
    assert client.calls == ["activate:u-1"]


def test_vpn_connect_rejects_unknown_uuid() -> None:
    client = FakeNMClient(vpns=[])
    result = _skill(client)["network.vpn_connect"].handler({"uuid": "nope"})
    assert result["ok"] is False


def test_vpn_disconnect_passes_uuid() -> None:
    client = FakeNMClient()
    assert _skill(client)["network.vpn_disconnect"].handler({"uuid": "u-9"})["ok"] is True
    assert client.calls == ["deactivate:u-9"]


@pytest.mark.parametrize(
    ("name", "risk", "params"),
    [
        ("network.wifi_list", RiskLevel.READ, {}),
        ("network.vpn_list", RiskLevel.READ, {}),
        ("network.wifi_set", RiskLevel.CONTROL, {"enabled": True}),
        ("network.vpn_connect", RiskLevel.CONTROL, {"uuid": "u-1"}),
        ("network.vpn_disconnect", RiskLevel.CONTROL, {"uuid": "u-1"}),
    ],
)
def test_risks_and_model_reachability(name: str, risk: RiskLevel, params: dict) -> None:
    spec = _skill(FakeNMClient())[name]
    assert spec.risk is risk
    assert spec.model_callable is True
    assert spec.auto_allow is True
    decision = ToolPolicy(AutonomyLevel.BALANCED.value).evaluate(
        name=name,
        params=params,
        spec=spec,
        panic=False,
        integrations_enabled=True,
        origin=Origin.MODEL,
    )
    assert decision.kind is DecisionKind.ALLOW
