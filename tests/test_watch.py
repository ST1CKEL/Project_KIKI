"""Proactive notices: watcher edges, delivery policy, and the poll service."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any

import pytest

from kiki.watch.models import Notice, Severity
from kiki.watch.notifier import Notifier, NotifierPolicy, in_quiet_hours, parse_clock
from kiki.watch.service import WatchService
from kiki.watch.watchers import BatteryWatcher, DiskWatcher


@dataclass
class FakeSnapshot:
    available: bool = True
    data: dict[str, Any] = field(default_factory=dict)


class FakeIntegration:
    """Replays a queue of snapshots, holding the last one."""

    def __init__(self, snapshots: list[FakeSnapshot]) -> None:
        self._queue = list(snapshots)
        self._last = snapshots[0] if snapshots else FakeSnapshot(available=False)

    def snapshot(self) -> FakeSnapshot:
        if self._queue:
            self._last = self._queue.pop(0)
        return self._last


def _battery(percentage, state="discharging") -> FakeSnapshot:
    return FakeSnapshot(data={"present": True, "percentage": percentage, "state": state})


def _disk(used_percent) -> FakeSnapshot:
    return FakeSnapshot(data={"path": "/home", "used_percent": used_percent, "free_human": "12.0 GiB"})


# --- battery ----------------------------------------------------------------


def test_battery_reports_once_per_drop_not_per_poll() -> None:
    watcher = BatteryWatcher(FakeIntegration([_battery(18)]), threshold_percent=20)
    first = watcher.check()
    assert first is not None
    assert first.severity is Severity.WARNING
    assert "18" in first.spoken
    # Same level on the next polls must stay quiet.
    assert watcher.check() is None
    assert watcher.check() is None


def test_battery_stays_quiet_above_the_threshold() -> None:
    watcher = BatteryWatcher(FakeIntegration([_battery(55)]), threshold_percent=20)
    assert watcher.check() is None


def test_battery_ignores_a_charging_machine() -> None:
    watcher = BatteryWatcher(FakeIntegration([_battery(9, state="charging")]), threshold_percent=20)
    assert watcher.check() is None


def test_battery_becomes_urgent_when_very_low() -> None:
    watcher = BatteryWatcher(FakeIntegration([_battery(7)]), threshold_percent=20, urgent_percent=10)
    notice = watcher.check()
    assert notice is not None
    assert notice.severity is Severity.URGENT
    assert "Anstecken" in notice.spoken


def test_battery_rearms_after_charging() -> None:
    integration = FakeIntegration([_battery(18), _battery(18), _battery(80, state="charging"), _battery(18)])
    watcher = BatteryWatcher(integration, threshold_percent=20)
    assert watcher.check() is not None  # first drop
    assert watcher.check() is None  # unchanged
    assert watcher.check() is None  # charging clears the condition
    assert watcher.check() is not None  # dropped again after unplugging


def test_battery_speaks_again_after_a_further_drop() -> None:
    watcher = BatteryWatcher(
        FakeIntegration([_battery(19), _battery(11)]), threshold_percent=20, rearm_margin=5
    )
    assert watcher.check() is not None
    second = watcher.check()
    assert second is not None and "11" in second.spoken


def test_battery_handles_machines_without_one() -> None:
    absent = FakeIntegration([FakeSnapshot(data={"present": False})])
    assert BatteryWatcher(absent).check() is None
    unavailable = FakeIntegration([FakeSnapshot(available=False)])
    assert BatteryWatcher(unavailable).check() is None


def test_a_broken_integration_never_escapes_the_watcher() -> None:
    class Exploding:
        def snapshot(self):
            raise RuntimeError("D-Bus weg")

    assert BatteryWatcher(Exploding()).check() is None
    assert DiskWatcher(Exploding()).check() is None


# --- disk -------------------------------------------------------------------


def test_disk_reports_once_when_it_fills_up() -> None:
    watcher = DiskWatcher(FakeIntegration([_disk(93)]), threshold_percent=90)
    notice = watcher.check()
    assert notice is not None
    assert notice.severity is Severity.WARNING
    assert "12.0 GiB" in notice.spoken
    assert watcher.check() is None


def test_disk_rearms_after_cleanup() -> None:
    integration = FakeIntegration([_disk(93), _disk(70), _disk(93)])
    watcher = DiskWatcher(integration, threshold_percent=90)
    assert watcher.check() is not None
    assert watcher.check() is None  # cleaned up
    assert watcher.check() is not None  # filled again


def test_disk_urgent_when_nearly_full() -> None:
    watcher = DiskWatcher(FakeIntegration([_disk(98)]), threshold_percent=90, urgent_percent=96)
    notice = watcher.check()
    assert notice is not None and notice.severity is Severity.URGENT


# --- quiet hours ------------------------------------------------------------


@pytest.mark.parametrize(
    ("hour", "expected"),
    [(23, True), (2, True), (7, True), (8, False), (12, False), (21, False), (22, True)],
)
def test_quiet_hours_wrap_around_midnight(hour, expected) -> None:
    now = datetime(2026, 8, 22, hour, 30)
    assert in_quiet_hours(now, time(22, 0), time(8, 0)) is expected


def test_quiet_hours_within_one_day() -> None:
    assert in_quiet_hours(datetime(2026, 8, 22, 13, 0), time(12, 0), time(14, 0)) is True
    assert in_quiet_hours(datetime(2026, 8, 22, 15, 0), time(12, 0), time(14, 0)) is False


def test_equal_bounds_mean_no_quiet_hours() -> None:
    assert in_quiet_hours(datetime(2026, 8, 22, 3, 0), time(8, 0), time(8, 0)) is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("22:00", time(22, 0)), ("7:5", time(7, 5)), ("08", time(8, 0))],
)
def test_parse_clock_reads_common_forms(raw, expected) -> None:
    assert parse_clock(raw, time(0, 0)) == expected


@pytest.mark.parametrize("raw", ["", "   ", "nonsense", "25:00", "12:99", None])
def test_unreadable_clock_keeps_the_fallback(raw) -> None:
    assert parse_clock(raw, time(22, 0)) == time(22, 0)


# --- notifier ---------------------------------------------------------------


def _notice(key="disk.full", severity=Severity.WARNING) -> Notice:
    return Notice(key=key, watcher="test", title="Titel", spoken="Kurzer Satz.", severity=severity)


class Clock:
    def __init__(self, start=0.0, wall=datetime(2026, 8, 22, 14, 0)) -> None:
        self.t = start
        self.wall = wall

    def mono(self) -> float:
        return self.t

    def now(self) -> datetime:
        return self.wall


def _notifier(policy=None, clock=None) -> tuple[Notifier, Clock]:
    c = clock or Clock()
    return Notifier(policy or NotifierPolicy(), clock=c.now, monotonic=c.mono), c


def test_a_warning_is_delivered_and_spoken() -> None:
    notifier, _c = _notifier()
    delivery = notifier.decide(_notice())
    assert delivery.notify is True
    assert delivery.speak is True


def test_info_notices_never_reach_the_speakers() -> None:
    notifier, _c = _notifier()
    delivery = notifier.decide(_notice(severity=Severity.INFO))
    assert delivery.notify is True
    assert delivery.speak is False


def test_panic_suppresses_everything() -> None:
    notifier, _c = _notifier()
    delivery = notifier.decide(_notice(severity=Severity.URGENT), panic=True)
    assert delivery.silent is True
    assert delivery.reason == "panic"


def test_kiki_does_not_talk_over_an_active_conversation() -> None:
    notifier, _c = _notifier()
    delivery = notifier.decide(_notice(), busy=True)
    assert delivery.notify is True
    assert delivery.speak is False


def test_the_same_notice_is_not_repeated_inside_the_cooldown() -> None:
    notifier, clock = _notifier(NotifierPolicy(cooldown_s=1800))
    assert notifier.decide(_notice()).notify is True

    clock.t = 600
    assert notifier.decide(_notice()).reason == "cooldown"

    clock.t = 2000
    assert notifier.decide(_notice()).notify is True


def test_different_keys_do_not_share_a_cooldown() -> None:
    notifier, _c = _notifier()
    assert notifier.decide(_notice(key="disk.full")).notify is True
    assert notifier.decide(_notice(key="battery.low")).notify is True


def test_hourly_budget_caps_a_runaway_watcher() -> None:
    notifier, clock = _notifier(NotifierPolicy(cooldown_s=0, max_per_hour=3))
    for index in range(3):
        clock.t = index
        assert notifier.decide(_notice(key=f"k{index}")).notify is True
    clock.t = 4
    assert notifier.decide(_notice(key="k4")).reason == "rate_limit"

    # The window slides, so an hour later it recovers.
    clock.t = 4000
    assert notifier.decide(_notice(key="k5")).notify is True


def test_quiet_hours_silence_a_warning_completely() -> None:
    clock = Clock(wall=datetime(2026, 8, 22, 23, 30))
    notifier, _c = _notifier(NotifierPolicy(), clock=clock)
    delivery = notifier.decide(_notice(severity=Severity.WARNING))
    assert delivery.silent is True
    assert delivery.reason == "quiet_hours"


def test_urgent_still_notifies_at_night_but_stays_silent() -> None:
    clock = Clock(wall=datetime(2026, 8, 22, 3, 0))
    notifier, _c = _notifier(NotifierPolicy(), clock=clock)
    delivery = notifier.decide(_notice(severity=Severity.URGENT))
    assert delivery.notify is True
    assert delivery.speak is False


def test_speaking_can_be_turned_off_entirely() -> None:
    notifier, _c = _notifier(NotifierPolicy(speak=False))
    delivery = notifier.decide(_notice(severity=Severity.URGENT))
    assert delivery.notify is True
    assert delivery.speak is False


def test_a_suppressed_notice_does_not_consume_the_budget() -> None:
    notifier, clock = _notifier(NotifierPolicy(cooldown_s=1800, max_per_hour=2))
    assert notifier.decide(_notice(key="a")).notify is True
    clock.t = 1
    assert notifier.decide(_notice(key="a")).reason == "cooldown"
    clock.t = 2
    # The cooldown hit must not have counted against the hourly budget.
    assert notifier.decide(_notice(key="b")).notify is True


# --- notice model -----------------------------------------------------------


def test_a_notice_needs_text() -> None:
    with pytest.raises(ValueError):
        Notice(key="", watcher="w", title="t", spoken="s")
    with pytest.raises(ValueError):
        Notice(key="k", watcher="w", title="t", spoken="")


# --- service ----------------------------------------------------------------


def test_service_collects_from_every_watcher() -> None:
    seen: list[Notice] = []
    service = WatchService(
        [
            BatteryWatcher(FakeIntegration([_battery(9)]), threshold_percent=20),
            DiskWatcher(FakeIntegration([_disk(95)]), threshold_percent=90),
        ],
        on_notice=seen.append,
    )
    found = asyncio.run(service.poll_once())
    assert {n.key for n in found} == {"battery.low", "disk.full"}
    assert len(seen) == 2


def test_one_broken_watcher_does_not_stop_the_others() -> None:
    class Broken:
        id = "broken"

        def check(self):
            raise RuntimeError("kaputt")

    seen: list[Notice] = []
    service = WatchService(
        [Broken(), DiskWatcher(FakeIntegration([_disk(95)]), threshold_percent=90)],
        on_notice=seen.append,
    )
    found = asyncio.run(service.poll_once())
    assert [n.key for n in found] == ["disk.full"]


def test_service_without_watchers_never_starts() -> None:
    service = WatchService([], on_notice=lambda _n: None)

    class Bridge:
        def submit(self, coro):
            coro.close()
            raise AssertionError("should not have been submitted")

    service.start(Bridge())
    assert service.running is False
