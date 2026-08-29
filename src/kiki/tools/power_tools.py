"""Power actions over logind: suspend, reboot, power off.

Suspend is CONTROL — the machine sleeps and wakes, nothing is lost, and a
Jarvis-style "Schlafmodus" should just work. Reboot and power off end every
unsaved thing on the desktop, KIKI included, so they are WRITE: outside the
jarvis level they always show the approval card, inside it they run — that is
exactly the trade the level describes. polkit grants all three to the active
local session without a password.

Login/logout are deliberately absent: they end the KIKI process itself before
it could even report success.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from kiki.platform import dbus
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec

log = logging.getLogger(__name__)

_LOGIND = "org.freedesktop.login1"
_LOGIND_PATH = "/org/freedesktop/login1"
_LOGIND_MANAGER = "org.freedesktop.login1.Manager"

PowerCaller = Callable[[str], None]


def _logind_call(method: str) -> None:
    from gi.repository import GLib

    dbus.call(
        dbus.system_bus(),
        _LOGIND,
        _LOGIND_PATH,
        _LOGIND_MANAGER,
        method,
        GLib.Variant("(b)", (False,)),
    )


class LogindClient:
    def __init__(self, caller: PowerCaller | None = None) -> None:
        self._caller = caller or _logind_call

    def suspend(self) -> None:
        self._caller("Suspend")

    def reboot(self) -> None:
        self._caller("Reboot")

    def power_off(self) -> None:
        self._caller("PowerOff")


class PowerControlSkill:
    id = "power_control"
    name = "Energie"
    description = "Ruhezustand, Neustart und Ausschalten der Maschine."

    def __init__(self, client: LogindClient | None = None) -> None:
        self._client = client or LogindClient()

    def _spec(
        self, slug: str, title: str, description: str, risk: RiskLevel, action: str
    ) -> ToolSpec:
        return ToolSpec(
            name=f"power.{slug}",
            title=title,
            description=description,
            risk=risk,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=lambda _params, action=action: self._run(action),
            effect=f"{title} der gesamten Maschine.",
            target="logind",
            auto_allow=True,
            model_callable=True,
        )

    def tools(self) -> list[ToolSpec]:
        return [
            self._spec(
                "suspend",
                "Ruhezustand",
                "Versetzt die Maschine in den Ruhezustand (Suspend-to-RAM).",
                RiskLevel.CONTROL,
                "suspend",
            ),
            self._spec(
                "reboot",
                "Neustart",
                "Startet die Maschine neu. Alles ungespeicherte geht verloren.",
                RiskLevel.WRITE,
                "reboot",
            ),
            self._spec(
                "poweroff",
                "Ausschalten",
                "Fährt die Maschine herunter. Alles ungespeicherte geht verloren.",
                RiskLevel.WRITE,
                "power_off",
            ),
        ]

    def _run(self, action: str) -> dict[str, Any]:
        try:
            getattr(self._client, action)()
        except dbus.BusError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "action": action}
