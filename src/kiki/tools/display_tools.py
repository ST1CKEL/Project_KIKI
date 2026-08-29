"""Display brightness over the session bus. GNOME first, Plasma as fallback.

There is no cross-desktop brightness API. GNOME exposes a percent property on
its power daemon; Plasma 6.3+ exposes per-display objects with their own
0..MaxBrightness scale. Both are wrapped behind one percent interface so the
tool never cares which desktop answered — and a desktop without either backend
gets a clean "unavailable", not a broken control.
"""

from __future__ import annotations

import logging
from typing import Any

from kiki.platform import dbus
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec

log = logging.getLogger(__name__)

_GNOME_DEST = "org.gnome.SettingsDaemon.Power"
_GNOME_PATH = "/org/gnome/SettingsDaemon/Power"
_GNOME_IFACE = "org.gnome.SettingsDaemon.Power.Screen"

_KDE_DEST = "org.kde.ScreenBrightness"
_KDE_PATH = "/org/kde/ScreenBrightness"
_KDE_MANAGER_IFACE = "org.kde.ScreenBrightness"
_KDE_DISPLAY_IFACE = "org.kde.ScreenBrightness.Display"


class BrightnessError(RuntimeError):
    """No backend answered, or the one that did refused the value."""


def clamp_percent(value: int) -> int:
    return max(0, min(100, int(value)))


class GnomeBrightnessBackend:
    """GNOME: the Screen.Brightness property is already a percent (0–100)."""

    name = "gnome"

    def available(self) -> bool:
        try:
            self._get()
        except BrightnessError:
            return False
        return True

    def get(self) -> int:
        return clamp_percent(int(self._get()))

    def set(self, percent: int) -> int:
        from gi.repository import GLib

        value = clamp_percent(percent)
        dbus.property_set(
            dbus.session_bus(),
            _GNOME_DEST,
            _GNOME_PATH,
            _GNOME_IFACE,
            "Brightness",
            GLib.Variant("i", value),
        )
        return value

    def _get(self) -> Any:
        try:
            raw = dbus.property_get(
                dbus.session_bus(), _GNOME_DEST, _GNOME_PATH, _GNOME_IFACE, "Brightness"
            )
        except Exception as exc:
            raise BrightnessError(f"GNOME-Helligkeit nicht erreichbar: {exc}") from exc
        if not isinstance(raw, int):
            raise BrightnessError("GNOME-Helligkeit hat ein unerwartetes Format.")
        return raw


class KdeBrightnessBackend:
    """Plasma 6.3+: per-display objects scaled by their own MaxBrightness."""

    name = "kde"

    def available(self) -> bool:
        try:
            return bool(self._display_names())
        except BrightnessError:
            return False

    def _display_names(self) -> list[str]:
        try:
            names = dbus.property_get(
                dbus.session_bus(), _KDE_DEST, _KDE_PATH, _KDE_MANAGER_IFACE, "DisplaysDBusNames"
            )
        except Exception as exc:
            raise BrightnessError(f"KDE-Helligkeit nicht erreichbar: {exc}") from exc
        if not isinstance(names, (list, tuple)) or not names:
            raise BrightnessError("KDE-Helligkeit meldet keine Displays.")
        return [str(n) for n in names]

    def _display_value(self, display: str, prop: str) -> int:
        raw = dbus.property_get(
            dbus.session_bus(), _KDE_DEST, f"{_KDE_PATH}/{display}", _KDE_DISPLAY_IFACE, prop
        )
        if not isinstance(raw, int):
            raise BrightnessError(f"KDE-Display {display}: unerwartetes Format für {prop}.")
        return raw

    def get(self) -> int:
        names = self._display_names()
        current = self._display_value(names[0], "Brightness")
        maximum = max(1, self._display_value(names[0], "MaxBrightness"))
        return clamp_percent(round(current * 100 / maximum))

    def set(self, percent: int) -> int:
        value_percent = clamp_percent(percent)
        names = self._display_names()
        for display in names:
            maximum = max(1, self._display_value(display, "MaxBrightness"))
            target = round(maximum * value_percent / 100)
            dbus.call(
                dbus.session_bus(),
                _KDE_DEST,
                f"{_KDE_PATH}/{display}",
                _KDE_DISPLAY_IFACE,
                "SetBrightness",
                _int_pair_variant(target, 0),
                None,
            )
        return value_percent


def _int_pair_variant(first: int, second: int) -> Any:
    from gi.repository import GLib

    return GLib.Variant("(iu)", (first, second))


class BrightnessController:
    """First available backend wins. One desktop answers; the rest stay quiet."""

    def __init__(self, backends: tuple[GnomeBrightnessBackend | KdeBrightnessBackend, ...] | None = None) -> None:
        self._backends = backends or (GnomeBrightnessBackend(), KdeBrightnessBackend())

    def _active(self):
        for backend in self._backends:
            if backend.available():
                return backend
        return None

    def get(self) -> tuple[int | None, str]:
        backend = self._active()
        if backend is None:
            return None, "Keine Helligkeits-Steuerung gefunden (weder GNOME noch KDE)."
        try:
            return backend.get(), backend.name
        except BrightnessError as exc:
            return None, str(exc)

    def set(self, percent: int) -> dict[str, Any]:
        backend = self._active()
        if backend is None:
            return {"ok": False, "error": "Keine Helligkeits-Steuerung gefunden (weder GNOME noch KDE)."}
        try:
            applied = backend.set(percent)
        except BrightnessError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            return {"ok": False, "error": f"Helligkeit nicht gesetzt: {exc}"}
        return {"ok": True, "percent": applied, "backend": backend.name}


class DisplayControlSkill:
    id = "display_control"
    name = "Helligkeit"
    description = "Bildschirmhelligkeit abfragen und setzen (0–100 Prozent)."

    def __init__(self, controller: BrightnessController | None = None) -> None:
        self._controller = controller or BrightnessController()

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="display.brightness_get",
                title="Helligkeit abfragen",
                description="Nennt die aktuelle Bildschirmhelligkeit in Prozent.",
                risk=RiskLevel.READ,
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=self._brightness_get,
                effect="Liest die Helligkeit über den Sitzungsbus. Keine Änderung.",
                target="Bildschirm",
                auto_allow=True,
                model_callable=True,
            ),
            ToolSpec(
                name="display.brightness_set",
                title="Helligkeit setzen",
                description=(
                    "Setzt die Bildschirmhelligkeit auf 0–100 Prozent. Werte außerhalb "
                    "werden auf den Bereich begrenzt."
                ),
                risk=RiskLevel.CONTROL,
                parameters={
                    "type": "object",
                    "properties": {"percent": {"type": "integer"}},
                    "required": ["percent"],
                    "additionalProperties": False,
                },
                handler=self._brightness_set,
                effect="Ändert die Bildschirmhelligkeit.",
                target="Bildschirm",
                auto_allow=True,
                model_callable=True,
            ),
        ]

    def _brightness_get(self, _params: dict[str, Any]) -> dict[str, Any]:
        percent, backend_or_error = self._controller.get()
        if percent is None:
            return {"ok": False, "error": backend_or_error}
        return {"ok": True, "percent": percent, "backend": backend_or_error}

    def _brightness_set(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._controller.set(int(params["percent"]))
