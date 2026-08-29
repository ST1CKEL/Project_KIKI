"""Metric values the routine engine compares triggers against.

Reads the same integrations the watchers read — no second D-Bus surface, no
caching: a routine decision is never made from stale numbers. Blocking calls
belong to a worker thread; the engine takes care of that.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from kiki.integrations.base import IntegrationSnapshot

log = logging.getLogger(__name__)

MetricProvider = Callable[[], dict[str, float]]


class IntegrationMetrics:
    """Pull routine metrics from the live UPower and disk integrations."""

    def __init__(self, upower: Any, disk: Any) -> None:
        self._upower = upower
        self._disk = disk

    def snapshot(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        metrics.update(self._battery_percent(self._safe_snapshot(self._upower)))
        metrics.update(self._disk_used_percent(self._safe_snapshot(self._disk)))
        return metrics

    def _safe_snapshot(self, integration: Any) -> IntegrationSnapshot:
        try:
            return integration.snapshot()
        except Exception:
            # An integration that throws reports nothing; it never stops the
            # other metrics from being collected.
            log.exception("metrics snapshot failed for %s", getattr(integration, "id", "?"))
            return IntegrationSnapshot("error", "Fehler", False, {}, "exception")

    @staticmethod
    def _battery_percent(snapshot: IntegrationSnapshot) -> dict[str, float]:
        data = snapshot.data if snapshot.available else {}
        if not data.get("present"):
            return {}
        percentage = data.get("percentage")
        # Discharging only: a charging battery must not fire low-battery
        # routines just because the number is still low on the way up.
        if data.get("state") != "discharging" or not isinstance(percentage, (int, float)):
            return {}
        return {"battery.percent": float(percentage)}

    @staticmethod
    def _disk_used_percent(snapshot: IntegrationSnapshot) -> dict[str, float]:
        data = snapshot.data if snapshot.available else {}
        used = data.get("used_percent")
        if not isinstance(used, (int, float)):
            return {}
        return {"disk.used_percent": float(used)}
