"""Dedicated asyncio loop so GTK never blocks on HTTP or SQLite."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


def _invoke_ui(callback: Callable[..., None] | None, *args: Any) -> None:
    if callback is None:
        return
    try:
        from gi.repository import GLib  # type: ignore

        def _run() -> bool:
            try:
                callback(*args)
            except Exception:
                log.exception("UI callback failed")
            return False

        GLib.idle_add(_run, priority=GLib.PRIORITY_DEFAULT)
    except Exception:
        # Tests and headless callers run callbacks inline.
        callback(*args)


class StreamHandle:
    """Cancel a running async generator from the GTK thread."""

    def __init__(self, bridge: AsyncBridge) -> None:
        self._bridge = bridge
        self._task: asyncio.Task[Any] | None = None
        self._cancelled = False

    def bind(self, task: asyncio.Task[Any]) -> None:
        self._task = task

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True
        if self._task is not None and self._bridge.loop is not None:
            self._bridge.loop.call_soon_threadsafe(self._task.cancel)


class SubmitHandle:
    """Thread-safe cancellation handle for a submitted coroutine."""

    def __init__(self, bridge: AsyncBridge) -> None:
        self._bridge = bridge
        self._task: asyncio.Task[Any] | None = None
        self._cancelled = False
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def bind(self, task: asyncio.Task[Any]) -> None:
        with self._lock:
            self._task = task
            cancelled = self._cancelled
        if cancelled:
            # bind() runs on the asyncio thread, so direct cancellation is safe.
            task.cancel()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            task = self._task
        loop = self._bridge.loop
        if task is not None and loop is not None:
            loop.call_soon_threadsafe(task.cancel)


class AsyncBridge:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop_requested = threading.Event()

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        return self._loop

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._ready.clear()
        self._stop_requested.clear()
        self._thread = threading.Thread(target=self._run, name="kiki-asyncio", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("asyncio bridge failed to start")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        try:
            # A small safety tick prevents a platform/sandbox from stranding
            # callbacks when asyncio's cross-thread self-pipe wakeup is lost.
            # Regular I/O still wakes the selector immediately.
            while not self._stop_requested.is_set():
                loop.call_later(0.05, loop.stop)
                loop.run_forever()
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    def stop(self) -> None:
        loop = self._loop
        if loop is None:
            return
        if self._thread is threading.current_thread():
            self._stop_requested.set()
            loop.stop()
            return

        async def _cancel_pending() -> None:
            current = asyncio.current_task()
            pending = [
                task
                for task in asyncio.all_tasks(loop)
                if task is not current and not task.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            cleanup = asyncio.run_coroutine_threadsafe(_cancel_pending(), loop)
            cleanup.result(timeout=6)
        except Exception:
            log.exception("asyncio bridge cleanup failed")
        self._stop_requested.set()
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._loop = None
        self._thread = None

    def submit(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        on_success: Callable[[T], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> SubmitHandle:
        loop = self._require_loop()
        handle = SubmitHandle(self)

        def _done(task: asyncio.Task[T]) -> None:
            try:
                result = task.result()
            except asyncio.CancelledError:
                _invoke_ui(on_complete)
                return
            except Exception as exc:
                log.exception("async task failed")
                _invoke_ui(on_error, exc)
                _invoke_ui(on_complete)
                return
            _invoke_ui(on_success, result)
            _invoke_ui(on_complete)

        def _schedule() -> None:
            task = loop.create_task(coro)
            handle.bind(task)
            task.add_done_callback(_done)

        loop.call_soon_threadsafe(_schedule)
        return handle

    def stream(
        self,
        agen: AsyncIterator[T],
        *,
        on_item: Callable[[T], None],
        on_error: Callable[[BaseException], None] | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> StreamHandle:
        handle = StreamHandle(self)
        loop = self._require_loop()

        async def _consume() -> None:
            try:
                async for item in agen:
                    if handle.cancelled:
                        break
                    _invoke_ui(on_item, item)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.exception("async stream failed")
                _invoke_ui(on_error, exc)
            finally:
                _invoke_ui(on_complete)

        def _schedule() -> None:
            task = loop.create_task(_consume())
            handle.bind(task)

        loop.call_soon_threadsafe(_schedule)
        return handle

    async def ask_ui(
        self,
        present: Callable[[Callable[[bool], None]], None],
        *,
        timeout: float = 300.0,
    ) -> bool:
        """From the asyncio thread: ask the GTK thread a yes/no question.

        `present` runs on the UI thread and receives a callback to settle the
        answer with. An unanswered question resolves to False once the timeout
        expires, so a dismissed dialog denies instead of stalling the caller
        forever.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()

        def _settle(value: bool) -> None:
            def _apply() -> None:
                if not future.done():
                    future.set_result(bool(value))

            loop.call_soon_threadsafe(_apply)

        _invoke_ui(present, _settle)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            log.warning("UI question timed out after %.0fs — denying", timeout)
            return False

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("AsyncBridge is not running")
        return self._loop
