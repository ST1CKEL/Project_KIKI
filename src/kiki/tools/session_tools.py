"""Lock the screen. One method call, two desktop dialects.

The XDG screensaver interface answers on KDE and most other desktops; GNOME
ships its own name. Locking is the only session action in this slice —
suspend, reboot and logout interrupt KIKI herself and stay a later decision.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from kiki.platform import dbus
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec

log = logging.getLogger(__name__)

LockCaller = Callable[[str, str, str], None]


class ScreenLockError(RuntimeError):
    """No desktop answered the lock request."""


# (destination, path, interface) — tried in order, first one that answers wins.
_LOCK_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("org.freedesktop.ScreenSaver", "/org/freedesktop/ScreenSaver", "org.freedesktop.ScreenSaver"),
    ("org.gnome.ScreenSaver", "/org/gnome/ScreenSaver", "org.gnome.ScreenSaver"),
)


def _dbus_lock(destination: str, path: str, interface: str) -> None:
    dbus.call(dbus.session_bus(), destination, path, interface, "Lock")


class ScreenLockClient:
    def __init__(self, caller: LockCaller | None = None) -> None:
        self._caller = caller or _dbus_lock

    def lock(self) -> str:
        """Returns the destination that locked. Raises when nobody answers."""
        problems: list[str] = []
        for destination, path, interface in _LOCK_TARGETS:
            try:
                self._caller(destination, path, interface)
                return destination
            except Exception as exc:
                problems.append(f"{destination}: {exc}")
        raise ScreenLockError("Bildschirmsperre nicht erreichbar. " + " | ".join(problems))


class SessionControlSkill:
    id = "session_control"
    name = "Sitzung"
    description = "Den Bildschirm sperren."

    def __init__(self, client: ScreenLockClient | None = None) -> None:
        self._client = client or ScreenLockClient()

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="session.lock",
                title="Bildschirm sperren",
                description="Sperrt den Bildschirm sofort.",
                risk=RiskLevel.CONTROL,
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=self._lock,
                effect="Sperrt die aktuelle Sitzung. Keine Datenänderung.",
                target="Bildschirmsperre",
                auto_allow=True,
                model_callable=True,
            ),
        ]

    def _lock(self, _params: dict[str, Any]) -> dict[str, Any]:
        try:
            target = self._client.lock()
        except ScreenLockError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "locked_via": target}
