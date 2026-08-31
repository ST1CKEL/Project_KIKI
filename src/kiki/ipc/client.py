"""Auto-reconnecting JSON-lines Unix client used by the orchestrator."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from kiki.ipc.protocol import dumps, loads

log = logging.getLogger("kiki.ipc")

OnMessage = Callable[[dict[str, Any]], Awaitable[None] | None]
OnLink = Callable[[bool], Awaitable[None] | None]


class JsonUnixClient:
    """One outbound connection to a KIKI service socket.

    Supervisors stay alive across service restarts. A missing socket is not a
    silent success — ``connected`` is False until a real handshake happens.
    """

    def __init__(self, path: Path, *, name: str = "service") -> None:
        self.path = Path(path)
        self.name = name
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._running = True

    @property
    def connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def send(self, payload: dict[str, Any]) -> bool:
        writer = self._writer
        if writer is None or writer.is_closing():
            return False
        try:
            writer.write(dumps(payload))
            await writer.drain()
            return True
        except Exception:
            log.debug("%s send failed", self.name, exc_info=True)
            await self._drop()
            return False

    async def close(self) -> None:
        self._running = False
        await self._drop()

    async def _drop(self) -> None:
        writer = self._writer
        self._reader = None
        self._writer = None
        if writer is None:
            return
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

    async def run(
        self,
        on_message: OnMessage,
        *,
        on_link: OnLink | None = None,
        retry_s: float = 1.0,
    ) -> None:
        while self._running:
            if not self.path.exists():
                await asyncio.sleep(retry_s)
                continue
            try:
                reader, writer = await asyncio.open_unix_connection(str(self.path))
            except Exception as exc:
                log.debug("%s connect failed: %s", self.name, exc)
                await asyncio.sleep(retry_s)
                continue
            self._reader = reader
            self._writer = writer
            log.info("connected to %s at %s", self.name, self.path)
            if on_link is not None:
                result = on_link(True)
                if asyncio.iscoroutine(result):
                    await result
            try:
                while self._running:
                    line = await reader.readline()
                    if not line:
                        break
                    try:
                        msg = loads(line)
                    except Exception:
                        continue
                    result = on_message(msg)
                    if asyncio.iscoroutine(result):
                        await result
            except Exception as exc:
                log.debug("%s read loop ended: %s", self.name, exc)
            finally:
                await self._drop()
                if on_link is not None:
                    result = on_link(False)
                    if asyncio.iscoroutine(result):
                        await result
            await asyncio.sleep(retry_s)
