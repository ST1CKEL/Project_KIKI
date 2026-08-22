"""NetworkManager connectivity. No SSID, no IP — those are identifying."""

from __future__ import annotations

from kiki.integrations.base import IntegrationSnapshot

_NM_STATE = {
    0: "unknown",
    10: "asleep",
    20: "disconnected",
    30: "disconnecting",
    40: "connecting",
    50: "connected_local",
    60: "connected_site",
    70: "connected_global",
}

_CONNECTIVITY = {
    0: "unknown",
    1: "none",
    2: "portal",
    3: "limited",
    4: "full",
}

_DEVICE_TYPE = {
    1: "ethernet",
    2: "wifi",
    16: "tun",
    23: "wireguard",
}


class NetworkManagerIntegration:
    id = "networkmanager"
    title = "Netzwerk"

    def snapshot(self) -> IntegrationSnapshot:
        try:
            from gi.repository import Gio  # type: ignore
        except Exception as exc:
            return IntegrationSnapshot(self.id, self.title, False, {}, f"Gio fehlt: {exc}")
        try:
            bus = Gio.bus_get_sync(Gio.BusType.SYSTEM, None)
            proxy = Gio.DBusProxy.new_sync(
                bus,
                Gio.DBusProxyFlags.NONE,
                None,
                "org.freedesktop.NetworkManager",
                "/org/freedesktop/NetworkManager",
                "org.freedesktop.NetworkManager",
                None,
            )
        except Exception as exc:
            return IntegrationSnapshot(
                self.id, self.title, False, {}, f"NetworkManager nicht erreichbar: {exc}"
            )

        def _prop(name: str):
            value = proxy.get_cached_property(name)
            return value.unpack() if value is not None else None

        state = int(_prop("State") or 0)
        connectivity = int(_prop("Connectivity") or 0)
        primary = _prop("PrimaryConnectionType")
        wireless = bool(_prop("WirelessEnabled") or False)
        return IntegrationSnapshot(
            self.id,
            self.title,
            True,
            {
                "state": _NM_STATE.get(state, str(state)),
                "connectivity": _CONNECTIVITY.get(connectivity, str(connectivity)),
                "primary_type": str(primary) if primary else None,
                "wireless_enabled": wireless,
                "connected": state >= 50,
            },
        )
