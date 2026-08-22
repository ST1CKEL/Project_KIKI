from __future__ import annotations

import asyncio
import threading

from kiki.runtime.async_bridge import AsyncBridge, SubmitHandle


def test_submit_returns_handle_that_cancels_running_coroutine() -> None:
    async def _run() -> None:
        bridge = AsyncBridge()
        bridge._loop = asyncio.get_running_loop()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def _wait_forever() -> None:
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        handle = bridge.submit(_wait_forever())
        await asyncio.wait_for(started.wait(), timeout=1)

        handle.cancel()

        assert handle.cancelled is True
        await asyncio.wait_for(cancelled.wait(), timeout=1)

    asyncio.run(_run())


def test_submit_handle_honours_cancel_before_task_binding() -> None:
    bridge = AsyncBridge()
    handle = SubmitHandle(bridge)

    class _Task:
        cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    task = _Task()
    handle.cancel()
    handle.bind(task)  # type: ignore[arg-type]

    assert task.cancelled is True


def test_bridge_stop_cancels_and_joins_pending_tasks() -> None:
    bridge = AsyncBridge()
    started = threading.Event()
    cancelled = threading.Event()

    async def _wait_forever() -> None:
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    bridge.start()
    bridge.submit(_wait_forever())
    assert started.wait(timeout=2)

    bridge.stop()

    assert cancelled.wait(timeout=2)
    assert bridge.loop is None
