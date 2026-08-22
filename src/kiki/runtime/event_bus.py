"""Tiny synchronous pub/sub. UI layers marshal to the GTK thread themselves."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

Listener = Callable[..., None]


class EventBus:
    def __init__(self) -> None:
        self._listeners: dict[str, list[Listener]] = defaultdict(list)

    def subscribe(self, event: str, listener: Listener) -> Callable[[], None]:
        self._listeners[event].append(listener)

        def _off() -> None:
            bucket = self._listeners.get(event)
            if bucket and listener in bucket:
                bucket.remove(listener)

        return _off

    def emit(self, event: str, **payload: Any) -> None:
        for listener in list(self._listeners.get(event, ())):
            listener(**payload)
