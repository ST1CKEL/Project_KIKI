"""Wi-Fi and VPN control over NetworkManager's system-bus interface.

The passive status integration (`integrations/networkmanager.py`) deliberately
reports no SSID — it runs unasked on every status card. These tools are
different: they run because the user asked for networks, so naming them is the
answer, not a leak. Writing is limited to two declared controls — the Wi-Fi
radio switch and activating/deactivating an existing connection. Creating or
editing connections stays out: that is settings territory, not chat territory.
"""

from __future__ import annotations

import logging
from typing import Any

from kiki.platform import dbus
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec

log = logging.getLogger(__name__)

_NM = "org.freedesktop.NetworkManager"
_NM_PATH = "/org/freedesktop/NetworkManager"
_NM_SETTINGS = f"{_NM}.Settings"
_NM_DEVICE = "org.freedesktop.NetworkManager.Device"
_NM_WIRELESS = "org.freedesktop.NetworkManager.Device.Wireless"
_NM_AP = "org.freedesktop.NetworkManager.AccessPoint"
_NM_CONN_ACTIVE = "org.freedesktop.NetworkManager.Connection.Active"

# NM device types we care about; mirrors the integration's mapping.
_DEVICE_TYPE_WIFI = 2

# Connection types that count as VPN for the tools below.
_VPN_TYPES = frozenset({"vpn", "wireguard"})

_MAX_APS = 12


class NetworkError(RuntimeError):
    """The bus or NetworkManager refused the request."""


def decode_ssid(raw: Any) -> str | None:
    """NM delivers SSIDs as byte arrays; a hidden AP has an empty one."""
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, (bytes, bytearray, list)):
        text = bytes(raw).decode("utf-8", errors="replace")
    else:
        return None
    return text.strip() or None


def rank_access_points(
    points: list[dict[str, Any]], limit: int = _MAX_APS
) -> list[dict[str, Any]]:
    """Deduplicate by SSID (strongest wins) and rank by signal strength.

    Pure presentation logic, kept out of the D-Bus client so it is testable
    without a bus.
    """
    best: dict[str, dict[str, Any]] = {}
    for point in points:
        ssid = str(point.get("ssid") or "").strip()
        if not ssid:
            continue
        known = best.get(ssid)
        if known is None or int(point.get("strength") or 0) > int(known.get("strength") or 0):
            best[ssid] = point
    ranked = sorted(best.values(), key=lambda p: int(p.get("strength") or 0), reverse=True)
    return ranked[:limit]


def _variant(signature: str):
    from gi.repository import GLib

    return GLib.VariantType(signature)


def _object_path_arg(path: Any) -> Any:
    from gi.repository import GLib

    return GLib.Variant("(o)", (str(path),))


class NetworkManagerClient:
    """Thin NM wrapper over the system bus. Replaceable in tests."""

    def __init__(self, bus_factory=None) -> None:
        self._bus_factory = bus_factory or dbus.system_bus

    def wifi_enabled(self) -> bool:
        raw = dbus.property_get(self._bus_factory(), _NM, _NM_PATH, _NM, "WirelessEnabled")
        return bool(raw)

    def set_wifi_enabled(self, enabled: bool) -> None:
        from gi.repository import GLib

        dbus.property_set(
            self._bus_factory(),
            _NM,
            _NM_PATH,
            _NM,
            "WirelessEnabled",
            GLib.Variant("b", bool(enabled)),
        )

    def _wifi_device_paths(self) -> list[str]:
        bus = self._bus_factory()
        reply = dbus.call(bus, _NM, _NM_PATH, _NM, "GetDevices", None, _variant("(ao)"))
        paths: list[str] = []
        for path in reply.unpack()[0]:
            try:
                dtype = dbus.property_get(bus, _NM, str(path), _NM_DEVICE, "DeviceType")
            except dbus.BusError:
                continue
            if int(dtype) == _DEVICE_TYPE_WIFI:
                paths.append(str(path))
        return paths

    def wifi_access_points(self) -> list[dict[str, Any]]:
        """Raw access-point dicts; ranking and capping live in the skill."""
        bus = self._bus_factory()
        points: list[dict[str, Any]] = []
        for device in self._wifi_device_paths():
            self._request_scan(bus, device)
            try:
                reply = dbus.call(
                    bus, _NM, device, _NM_WIRELESS, "GetAllAccessPoints", None, _variant("(ao)")
                )
            except dbus.BusError as exc:
                raise NetworkError(f"WLAN-Netzwerke nicht abrufbar: {exc}") from exc
            for path in reply.unpack()[0]:
                point = self._read_ap(bus, str(path))
                if point is not None:
                    points.append(point)
        return points

    def _request_scan(self, bus: Any, device: str) -> None:
        from gi.repository import GLib

        # Best effort: a scan refreshes the AP cache but may be denied or busy;
        # listing what the cache already holds is still a useful answer.
        try:
            dbus.call(
                bus,
                _NM,
                device,
                _NM_WIRELESS,
                "RequestScan",
                # new_tuple because pygobject cannot build "(a{sv})" from {}.
                GLib.Variant.new_tuple(GLib.Variant("a{sv}", None)),
                None,
            )
        except dbus.BusError as exc:
            log.debug("wifi scan skipped: %s", exc)

    def _read_ap(self, bus: Any, path: str) -> dict[str, Any] | None:
        try:
            ssid = decode_ssid(dbus.property_get(bus, _NM, path, _NM_AP, "Ssid"))
            strength = int(dbus.property_get(bus, _NM, path, _NM_AP, "Strength"))
            wpa = int(dbus.property_get(bus, _NM, path, _NM_AP, "WpaFlags") or 0)
            rsn = int(dbus.property_get(bus, _NM, path, _NM_AP, "RsnFlags") or 0)
        except dbus.BusError:
            return None
        if ssid is None:
            return None
        return {"ssid": ssid, "strength": strength, "secure": bool(wpa or rsn)}

    def vpn_connections(self) -> list[dict[str, str]]:
        bus = self._bus_factory()
        reply = dbus.call(
            bus, _NM, f"{_NM_PATH}/Settings", _NM_SETTINGS, "ListConnections", None, _variant("(ao)")
        )
        found: list[dict[str, str]] = []
        for path in reply.unpack()[0]:
            settings = self._connection_settings(bus, str(path))
            if settings is None:
                continue
            if settings.get("type") in _VPN_TYPES:
                found.append(settings)
        return found

    def _connection_settings(self, bus: Any, path: str) -> dict[str, str] | None:
        from gi.repository import GLib

        try:
            reply = dbus.call(
                bus,
                _NM,
                path,
                f"{_NM}.Settings.Connection",
                "GetSettings",
                None,
                GLib.VariantType("(a{sa{sv}})"),
            )
        except dbus.BusError:
            return None
        connection = (reply.unpack()[0] or {}).get("connection") or {}
        conn_type = str(connection.get("type") or "")
        if conn_type not in _VPN_TYPES:
            return None
        return {
            "id": str(connection.get("id") or ""),
            "uuid": str(connection.get("uuid") or ""),
            "type": conn_type,
        }

    def _connection_path_by_uuid(self, uuid: str) -> str:
        bus = self._bus_factory()
        reply = dbus.call(
            bus, _NM, f"{_NM_PATH}/Settings", _NM_SETTINGS, "ListConnections", None, _variant("(ao)")
        )
        for path in reply.unpack()[0]:
            settings = self._connection_settings(bus, str(path))
            if settings is not None and settings["uuid"] == uuid:
                return str(path)
        raise NetworkError(f"Keine VPN-Verbindung mit UUID „{uuid}“ gefunden.")

    def activate_connection(self, uuid: str) -> str:
        from gi.repository import GLib

        path = self._connection_path_by_uuid(uuid)
        dbus.call(
            self._bus_factory(),
            _NM,
            _NM_PATH,
            _NM,
            "ActivateConnection",
            GLib.Variant("(ooo)", (path, "/", "/")),
            None,
        )
        return path

    def deactivate_connection(self, uuid: str) -> None:
        bus = self._bus_factory()
        reply = dbus.call(
            bus, _NM, _NM_PATH, _NM, "GetActiveConnections", None, _variant("(ao)")
        )
        for path in reply.unpack()[0]:
            try:
                active_uuid = dbus.property_get(
                    bus, _NM, str(path), _NM_CONN_ACTIVE, "Uuid"
                )
            except dbus.BusError:
                continue
            if str(active_uuid) == uuid:
                dbus.call(bus, _NM, _NM_PATH, _NM, "DeactivateConnection", _object_path_arg(path))
                return
        raise NetworkError("Diese Verbindung ist gerade nicht aktiv.")


class NetworkControlSkill:
    id = "network_control"
    name = "Netzwerk"
    description = "WLAN-Gerät schalten, Netzwerke anzeigen und VPN-Verbindungen verbinden/trennen."

    def __init__(self, client: NetworkManagerClient | None = None) -> None:
        self._client = client or NetworkManagerClient()

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="network.wifi_list",
                title="WLAN-Netzwerke anzeigen",
                description=(
                    "Listet die WLAN-Netzwerke in Reichweite mit Signalstärke und "
                    "ob sie verschlüsselt sind."
                ),
                risk=RiskLevel.READ,
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=self._wifi_list,
                effect="Fragt einen WLAN-Scan über NetworkManager ab. Keine Änderung.",
                target="NetworkManager",
                auto_allow=True,
                model_callable=True,
            ),
            ToolSpec(
                name="network.wifi_set",
                title="WLAN ein-/ausschalten",
                description="Schaltet das WLAN-Funkgerät an (enabled=true) oder aus (enabled=false).",
                risk=RiskLevel.CONTROL,
                parameters={
                    "type": "object",
                    "properties": {"enabled": {"type": "boolean"}},
                    "required": ["enabled"],
                    "additionalProperties": False,
                },
                handler=self._wifi_set,
                effect="Schaltet das WLAN-Funkgerät der Maschine.",
                target="NetworkManager",
                auto_allow=True,
                model_callable=True,
            ),
            ToolSpec(
                name="network.vpn_list",
                title="VPN-Verbindungen anzeigen",
                description=(
                    "Listet die eingerichteten VPN-Verbindungen (inkl. WireGuard) mit "
                    "Namen und uuid. Die uuid gehört in network.vpn_connect."
                ),
                risk=RiskLevel.READ,
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=self._vpn_list,
                effect="Liest die Verbindungsliste des NetworkManager. Keine Änderung.",
                target="NetworkManager",
                auto_allow=True,
                model_callable=True,
            ),
            ToolSpec(
                name="network.vpn_connect",
                title="VPN verbinden",
                description="Baut eine bestehende VPN-Verbindung auf. Die uuid kommt von network.vpn_list.",
                risk=RiskLevel.CONTROL,
                parameters={
                    "type": "object",
                    "properties": {"uuid": {"type": "string", "minLength": 1, "maxLength": 64}},
                    "required": ["uuid"],
                    "additionalProperties": False,
                },
                handler=self._vpn_connect,
                effect="Aktiviert eine gespeicherte VPN-Verbindung.",
                target="NetworkManager",
                auto_allow=True,
                model_callable=True,
            ),
            ToolSpec(
                name="network.vpn_disconnect",
                title="VPN trennen",
                description="Trennt eine aktive VPN-Verbindung. Die uuid kommt von network.vpn_list.",
                risk=RiskLevel.CONTROL,
                parameters={
                    "type": "object",
                    "properties": {"uuid": {"type": "string", "minLength": 1, "maxLength": 64}},
                    "required": ["uuid"],
                    "additionalProperties": False,
                },
                handler=self._vpn_disconnect,
                effect="Deaktiviert eine aktive VPN-Verbindung.",
                target="NetworkManager",
                auto_allow=True,
                model_callable=True,
            ),
        ]

    def _wifi_list(self, _params: dict[str, Any]) -> dict[str, Any]:
        try:
            points = self._client.wifi_access_points()
            enabled = self._client.wifi_enabled()
        except (dbus.BusError, NetworkError) as exc:
            return {"ok": False, "error": str(exc)}
        ranked = rank_access_points(points)
        return {
            "ok": True,
            "wifi_enabled": enabled,
            "count": len(ranked),
            "networks": ranked,
        }

    def _wifi_set(self, params: dict[str, Any]) -> dict[str, Any]:
        enabled = bool(params["enabled"])
        try:
            self._client.set_wifi_enabled(enabled)
        except (dbus.BusError, NetworkError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "wifi_enabled": enabled}

    def _vpn_list(self, _params: dict[str, Any]) -> dict[str, Any]:
        try:
            vpns = self._client.vpn_connections()
        except (dbus.BusError, NetworkError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "count": len(vpns), "connections": vpns}

    def _vpn_connect(self, params: dict[str, Any]) -> dict[str, Any]:
        uuid = str(params["uuid"])
        try:
            self._client.activate_connection(uuid)
        except (dbus.BusError, NetworkError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "uuid": uuid}

    def _vpn_disconnect(self, params: dict[str, Any]) -> dict[str, Any]:
        uuid = str(params["uuid"])
        try:
            self._client.deactivate_connection(uuid)
        except (dbus.BusError, NetworkError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "uuid": uuid}
