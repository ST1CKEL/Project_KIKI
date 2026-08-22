"""Concrete watchers.

Each one is **edge triggered**: it reports when a condition *becomes* true and
stays silent until the condition clears again. Without that a battery sitting at
19 % would produce a notice on every single poll, and the notifier's cooldown
would be doing work the watcher should have done itself.

Re-arming uses a margin so a value hovering exactly on the threshold cannot
flap between armed and fired.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from kiki.watch.models import Notice, Severity

log = logging.getLogger(__name__)


class Watcher(Protocol):
    id: str

    def check(self) -> Notice | None:
        """Return a notice when something changed for the worse. Never raises."""
        ...


def _fmt_percent(value: float) -> str:
    return f"{value:.0f}"


class BatteryWatcher:
    """Warns once when the battery runs low while unplugged."""

    id = "battery"

    def __init__(
        self,
        integration: Any,
        *,
        threshold_percent: float = 20.0,
        urgent_percent: float = 10.0,
        rearm_margin: float = 5.0,
    ) -> None:
        self._integration = integration
        self._threshold = float(threshold_percent)
        self._urgent = float(urgent_percent)
        self._margin = float(rearm_margin)
        self._fired_at: float | None = None

    def check(self) -> Notice | None:
        try:
            snapshot = self._integration.snapshot()
        except Exception:
            log.debug("battery watcher failed", exc_info=True)
            return None
        if not getattr(snapshot, "available", False):
            return None
        data = getattr(snapshot, "data", {}) or {}
        if not data.get("present"):
            return None
        percentage = data.get("percentage")
        if percentage is None:
            return None
        level = float(percentage)
        charging = str(data.get("state") or "") in {"charging", "fully_charged", "pending_charge"}

        if charging or level > self._threshold + self._margin:
            # Condition cleared — allow the next drop to speak up again.
            self._fired_at = None
            return None
        if level > self._threshold:
            return None
        if self._fired_at is not None and level >= self._fired_at - self._margin:
            # Already reported at this level; wait for a further drop.
            return None
        self._fired_at = level

        urgent = level <= self._urgent
        return Notice(
            key="battery.low",
            watcher=self.id,
            title="Akku niedrig",
            spoken=(
                f"Der Akku ist bei {_fmt_percent(level)} Prozent. Zeit zum Anstecken."
                if urgent
                else f"Der Akku ist bei {_fmt_percent(level)} Prozent."
            ),
            detail=f"Akkustand {_fmt_percent(level)} %, Zustand: {data.get('state')}.",
            severity=Severity.URGENT if urgent else Severity.WARNING,
        )


class DiskWatcher:
    """Warns once when the watched filesystem gets full."""

    id = "disk"

    def __init__(
        self,
        integration: Any,
        *,
        threshold_percent: float = 90.0,
        urgent_percent: float = 96.0,
        rearm_margin: float = 3.0,
    ) -> None:
        self._integration = integration
        self._threshold = float(threshold_percent)
        self._urgent = float(urgent_percent)
        self._margin = float(rearm_margin)
        self._fired_at: float | None = None

    def check(self) -> Notice | None:
        try:
            snapshot = self._integration.snapshot()
        except Exception:
            log.debug("disk watcher failed", exc_info=True)
            return None
        if not getattr(snapshot, "available", False):
            return None
        data = getattr(snapshot, "data", {}) or {}
        used = data.get("used_percent")
        if used is None:
            return None
        level = float(used)

        if level < self._threshold - self._margin:
            self._fired_at = None
            return None
        if level < self._threshold:
            return None
        if self._fired_at is not None and level <= self._fired_at + self._margin:
            return None
        self._fired_at = level

        free = str(data.get("free_human") or "?")
        urgent = level >= self._urgent
        return Notice(
            key="disk.full",
            watcher=self.id,
            title="Speicher wird knapp",
            spoken=(f"Die Platte ist zu {_fmt_percent(level)} Prozent voll. Nur noch {free} frei."),
            detail=f"{data.get('path')}: {_fmt_percent(level)} % belegt, {free} frei.",
            severity=Severity.URGENT if urgent else Severity.WARNING,
        )
