"""The assistant pause: no new work, and that is all it is.

The character pause stops an animation. The panic switch kills integrations
and every tool. Between them there was nothing for "KIKI, halt mal kurz":
stop taking new work without touching the pet, without destroying anything,
and without the privacy weight of panic. This is that switch.

What a pause stops: new runs (agent and voice), routine fires, delivered
watch notices. What it leaves alone: the run that is already going -- it was
asked for, it is bounded, and it settles on its own; an approval card that
is on screen stays answerable; typed chat stays open, because someone typing
is someone explicitly asking, not background work. It is session state, not
a setting: a restart begins unpaused, on purpose.

Thread context like the activity view: flipped from GTK, read from asyncio.
One flag, one lock, no listeners of its own -- whoever wants the state asks.
"""

from __future__ import annotations

import threading


class AssistantPause:
    """A thread-safe session flag: paused means no new assistant work."""

    def __init__(self) -> None:
        self._paused = False
        self._lock = threading.Lock()

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def toggle(self) -> bool:
        """Flip and report the new state."""
        with self._lock:
            self._paused = not self._paused
            return self._paused
