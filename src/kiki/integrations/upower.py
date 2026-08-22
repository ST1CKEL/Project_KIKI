"""Battery status via UPower DisplayDevice. Safe no-op on desktops without a battery."""

from __future__ import annotations

import logging

from kiki.integrations.base import IntegrationSnapshot

log = logging.getLogger(__name__)

_STATE = {
    0: "unknown",
    1: "charging",
    2: "discharging",
    3: "empty",
    4: "fully_charged",
    5: "pending_charge",
    6: "pending_discharge",
}


class UPowerIntegration:
    id = "upower"
    title = "Akku"

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
                "org.freedesktop.UPower",
                "/org/freedesktop/UPower/devices/DisplayDevice",
                "org.freedesktop.UPower.Device",
                None,
            )
        except Exception as exc:
            return IntegrationSnapshot(self.id, self.title, False, {}, f"UPower nicht erreichbar: {exc}")

        def _prop(name: str):
            value = proxy.get_cached_property(name)
            return value.unpack() if value is not None else None

        present = bool(_prop("IsPresent") or False)
        kind = int(_prop("Type") or 0)
        # Type 2 = battery
        if not present or kind not in {0, 2}:
            return IntegrationSnapshot(
                self.id,
                self.title,
                True,
                {"present": False, "note": "Kein Akku (Desktop oder VM)."},
            )
        state = int(_prop("State") or 0)
        percentage = _prop("Percentage")
        return IntegrationSnapshot(
            self.id,
            self.title,
            True,
            {
                "present": True,
                "percentage": float(percentage) if percentage is not None else None,
                "state": _STATE.get(state, str(state)),
                "time_to_empty_sec": _prop("TimeToEmpty"),
                "time_to_full_sec": _prop("TimeToFull"),
            },
        )
