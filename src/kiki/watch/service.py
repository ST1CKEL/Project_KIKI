"""Polls the watchers on the asyncio thread and hands notices to a sink.

Watchers do blocking I/O (D-Bus, statvfs), so each poll runs in a worker thread.
A watcher that throws is logged and skipped — one broken observer must not stop
the others or kill the loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable

from kiki.watch.models import Notice
from kiki.watch.watchers import Watcher

log = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 60.0


class WatchService:
    def __init__(
        self,
        watchers: Iterable[Watcher],
        *,
        on_notice: Callable[[Notice], None],
        interval_s: float = DEFAULT_INTERVAL_S,
        enabled: Callable[[], bool] | None = None,
    ) -> None:
        self._watchers = list(watchers)
        self._on_notice = on_notice
        self._interval = max(5.0, float(interval_s))
        self._enabled = enabled or (lambda: True)
        self._handle = None

    @property
    def running(self) -> bool:
        return self._handle is not None

    def start(self, bridge) -> None:
        if self._handle is not None or not self._watchers:
            return
        self._handle = bridge.submit(self._loop())

    def stop(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            handle.cancel()

    async def _loop(self) -> None:
        # A short first delay keeps startup quiet and avoids a notice landing
        # before the pet window is even on screen.
        await asyncio.sleep(min(self._interval, 15.0))
        while True:
            try:
                if self._enabled():
                    await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("watch loop iteration failed")
            await asyncio.sleep(self._interval)

    async def poll_once(self) -> list[Notice]:
        """Run every watcher once. Exposed separately so tests need no loop."""
        found: list[Notice] = []
        for watcher in self._watchers:
            try:
                notice = await asyncio.to_thread(watcher.check)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("watcher %s failed", getattr(watcher, "id", "?"))
                continue
            if notice is not None:
                found.append(notice)
                self._on_notice(notice)
        return found
