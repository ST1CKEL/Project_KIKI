"""GTK-freie Character-State-Machine.

Unbekannte Namen fallen auf idle. Unerlaubte Übergänge werden verworfen
(fail closed: der aktuelle Zustand bleibt).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum

Listener = Callable[["CharacterState"], None]


class CharacterState(StrEnum):
    IDLE = "idle"
    IDLE_BLINK = "idle_blink"
    GREET = "greet"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    HAPPY = "happy"
    SURPRISED = "surprised"
    SLEEPING = "sleeping"
    ERROR = "error"
    NOTIFICATION = "notification"
    PAUSED = "paused"


MVP_STATES: frozenset[CharacterState] = frozenset(
    {
        CharacterState.IDLE,
        CharacterState.THINKING,
        CharacterState.SPEAKING,
        CharacterState.ERROR,
        CharacterState.PAUSED,
    }
)

TRANSITORY: frozenset[CharacterState] = frozenset(
    {
        CharacterState.IDLE_BLINK,
        CharacterState.GREET,
        CharacterState.HAPPY,
        CharacterState.SURPRISED,
        CharacterState.ERROR,
        CharacterState.NOTIFICATION,
    }
)

DEFAULT_HOLD_MS: dict[CharacterState, int] = {
    CharacterState.IDLE_BLINK: 180,
    CharacterState.GREET: 1800,
    CharacterState.HAPPY: 1600,
    CharacterState.SURPRISED: 1400,
    CharacterState.ERROR: 2400,
    CharacterState.NOTIFICATION: 2000,
}

_ACTIVE: frozenset[CharacterState] = frozenset(s for s in CharacterState if s is not CharacterState.PAUSED)

ALLOWED_TRANSITIONS: dict[CharacterState, frozenset[CharacterState]] = {
    CharacterState.PAUSED: frozenset({CharacterState.PAUSED, CharacterState.IDLE}),
}
for _state in _ACTIVE:
    ALLOWED_TRANSITIONS[_state] = _ACTIVE


def resolve_state(value: CharacterState | str) -> CharacterState:
    """Map a name to a known state. Unknown values become idle."""
    if isinstance(value, CharacterState):
        return value
    try:
        return CharacterState(str(value))
    except ValueError:
        return CharacterState.IDLE


class CharacterStateMachine:
    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._state = CharacterState.IDLE
        self._base_state = CharacterState.IDLE
        self._until: float | None = None
        self._listeners: list[Listener] = []

    @property
    def state(self) -> CharacterState:
        return self._state

    @property
    def base_state(self) -> CharacterState:
        """Persistent activity restored after a short reaction finishes."""
        return self._base_state

    @property
    def paused(self) -> bool:
        return self._state is CharacterState.PAUSED

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.append(listener)

        def _off() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return _off

    def can_transition(self, target: CharacterState | str) -> bool:
        dest = resolve_state(target)
        allowed = ALLOWED_TRANSITIONS.get(self._state, _ACTIVE)
        return dest in allowed

    def set(self, state: CharacterState | str, *, hold_ms: int | None = None) -> CharacterState:
        target = resolve_state(state)
        if not self.can_transition(target):
            return self._state
        changed = target != self._state
        self._state = target
        if target not in TRANSITORY:
            self._base_state = target
        if hold_ms is None and target in TRANSITORY:
            hold_ms = DEFAULT_HOLD_MS.get(target)
        if hold_ms and hold_ms > 0:
            self._until = self._clock() + hold_ms / 1000.0
        else:
            self._until = None
        if changed:
            self._notify()
        return self._state

    def pause(self) -> CharacterState:
        if self._state is CharacterState.PAUSED:
            return self._state
        self._until = None
        self._state = CharacterState.PAUSED
        self._notify()
        return self._state

    def resume(self) -> CharacterState:
        if self._state is not CharacterState.PAUSED:
            return self._state
        return self.set(CharacterState.IDLE, hold_ms=0)

    def tick(self, now: float | None = None) -> CharacterState:
        if self._until is None or self._state is CharacterState.PAUSED:
            return self._state
        ts = self._clock() if now is None else now
        if ts >= self._until:
            self._until = None
            if self._state != self._base_state:
                self._state = self._base_state
                self._notify()
        return self._state

    def _notify(self) -> None:
        for listener in list(self._listeners):
            listener(self._state)
