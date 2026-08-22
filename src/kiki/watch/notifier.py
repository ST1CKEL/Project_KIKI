"""Decides whether a notice reaches the user, and how loudly.

Pure decision logic with an injected clock, so the awkward cases — quiet hours
crossing midnight, a watcher stuck repeating itself, KIKI about to talk over the
user — are testable without waiting for real time to pass.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time

from kiki.watch.models import SPOKEN_SEVERITIES, Notice, Severity

log = logging.getLogger(__name__)

DEFAULT_COOLDOWN_S = 1800.0
DEFAULT_MAX_PER_HOUR = 6


def parse_clock(value: str, fallback: time) -> time:
    """Read "HH:MM". Anything unreadable keeps the fallback."""
    text = str(value or "").strip()
    if not text:
        return fallback
    try:
        hour, _, minute = text.partition(":")
        return time(hour=int(hour), minute=int(minute or 0))
    except (TypeError, ValueError):
        return fallback


def in_quiet_hours(now: datetime, start: time, end: time) -> bool:
    """True inside the quiet window, which normally wraps past midnight."""
    if start == end:
        return False
    current = now.time()
    if start < end:
        return start <= current < end
    # Wrapping window, e.g. 22:00–08:00.
    return current >= start or current < end


@dataclass
class NotifierPolicy:
    speak: bool = True
    quiet_start: time = time(22, 0)
    quiet_end: time = time(8, 0)
    # Even inside quiet hours, an urgent notice still appears silently.
    quiet_allows_urgent_notification: bool = True
    cooldown_s: float = DEFAULT_COOLDOWN_S
    max_per_hour: int = DEFAULT_MAX_PER_HOUR


@dataclass(frozen=True)
class Delivery:
    """What should actually happen for one notice."""

    notify: bool = False
    speak: bool = False
    reason: str = ""

    @property
    def silent(self) -> bool:
        return not self.notify and not self.speak


class Notifier:
    def __init__(
        self,
        policy: NotifierPolicy,
        *,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._policy = policy
        self._clock = clock or datetime.now
        self._monotonic = monotonic or _default_monotonic
        self._last_seen: dict[str, float] = {}
        self._recent: list[float] = []

    def update_policy(self, policy: NotifierPolicy) -> None:
        self._policy = policy

    def decide(
        self,
        notice: Notice,
        *,
        panic: bool = False,
        busy: bool = False,
    ) -> Delivery:
        """`busy` means the user is mid-conversation with KIKI right now."""
        if panic:
            return Delivery(reason="panic")

        now = self._monotonic()
        last = self._last_seen.get(notice.key)
        if last is not None and now - last < self._policy.cooldown_s:
            return Delivery(reason="cooldown")

        self._recent = [t for t in self._recent if now - t < 3600.0]
        if len(self._recent) >= max(0, self._policy.max_per_hour):
            log.info("notice %s dropped: hourly budget spent", notice.key)
            return Delivery(reason="rate_limit")

        quiet = in_quiet_hours(self._clock(), self._policy.quiet_start, self._policy.quiet_end)
        if quiet and not (
            notice.severity is Severity.URGENT and self._policy.quiet_allows_urgent_notification
        ):
            return Delivery(reason="quiet_hours")

        speak = (
            self._policy.speak
            and notice.severity in SPOKEN_SEVERITIES
            and not quiet
            # Never talk over the user or over KIKI's own answer.
            and not busy
        )
        self._last_seen[notice.key] = now
        self._recent.append(now)
        return Delivery(notify=True, speak=speak, reason="delivered")

    def reset(self) -> None:
        self._last_seen.clear()
        self._recent.clear()


def _default_monotonic() -> float:
    import time as _time

    return _time.monotonic()
