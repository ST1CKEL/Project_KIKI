"""Poll loop for the routine engine, mirroring the watch service."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from kiki.routines.engine import RoutineEngine

log = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 60.0


class RoutineService:
    def __init__(
        self,
        engine: RoutineEngine,
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
        enabled: Callable[[], bool] | None = None,
    ) -> None:
        self._engine = engine
        self._interval = max(5.0, float(interval_s))
        self._enabled = enabled or (lambda: True)
        self._handle = None

    @property
    def running(self) -> bool:
        return self._handle is not None

    def start(self, bridge) -> None:
        if self._handle is not None:
            return
        self._handle = bridge.submit(self._loop())

    def stop(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            handle.cancel()

    async def _loop(self) -> None:
        await asyncio.sleep(min(self._interval, 15.0))
        while True:
            try:
                if self._enabled():
                    await self._engine.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("routine loop iteration failed")
            await asyncio.sleep(self._interval)
