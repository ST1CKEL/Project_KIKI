"""Fires approved routines on the asyncio thread.

The engine decides *when*; *what* still goes through the tool executor with
`Origin.ROUTINE`, so policy, panic, audit and the auto_allow veto all apply to
a background fire exactly as they do to a chat request. Cooldowns keep a stuck
metric from hammering a tool every tick.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from kiki.routines.metrics import MetricProvider
from kiki.routines.models import Routine
from kiki.routines.repository import RoutineRepository
from kiki.tools.executor import ToolExecutor
from kiki.tools.policy import DecisionKind, Origin

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RoutineEngine:
    def __init__(
        self,
        repository: RoutineRepository,
        executor: ToolExecutor,
        metrics: MetricProvider,
        *,
        panic_check: Callable[[], bool],
        integrations_check: Callable[[], bool],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._repository = repository
        self._executor = executor
        self._metrics = metrics
        self._panic_check = panic_check
        self._integrations_check = integrations_check
        self._clock = clock

    async def tick(self) -> list[dict[str, Any]]:
        """One evaluation pass. Returns what fired, for logging and tests."""
        if self._panic_check() or not self._integrations_check():
            return []
        try:
            metrics = await asyncio.to_thread(self._metrics)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("routine metrics failed")
            return []
        fired: list[dict[str, Any]] = []
        for routine in self._repository.list():
            if not routine.enabled:
                continue
            current = metrics.get(routine.trigger.metric)
            if current is None or not routine.trigger.matches(current):
                continue
            if not self._cooldown_elapsed(routine):
                continue
            fired.append(await self._fire(routine))
        return fired

    async def _fire(self, routine: Routine) -> dict[str, Any]:
        result = await self._executor.run(
            routine.tool_name,
            routine.arguments,
            panic=self._panic_check(),
            integrations_enabled=self._integrations_check(),
            origin=Origin.ROUTINE,
        )
        summary = {
            "routine": routine.name,
            "routine_id": routine.id,
            "tool": routine.tool_name,
            "ok": bool(result.ok),
        }
        if result.ok:
            self._repository.record_fired(routine.id, _now_iso())
            log.info("routine fired: %s → %s", routine.name, routine.tool_name)
        else:
            summary["error"] = result.error
            log.warning("routine %s failed: %s", routine.name, result.error)
            if result.decision.kind is DecisionKind.DENY:
                # A policy deny is permanent (tool gone, auto_allow withdrawn).
                # Disable the routine instead of re-denying it into the audit
                # log every tick; the user can delete it in the settings.
                self._repository.set_enabled(routine.id, False)
                summary["disabled"] = True
                log.warning("routine %s disabled by policy deny", routine.name)
        return summary

    def _cooldown_elapsed(self, routine: Routine) -> bool:
        if not routine.last_fired_at:
            return True
        try:
            last = datetime.fromisoformat(routine.last_fired_at).timestamp()
        except ValueError:
            return True
        return (self._clock() - last) >= routine.cooldown_min * 60
