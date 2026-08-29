"""The activity view: bounded, content-free, and current.

Runs, routine fires and delivered notices feed one ring; the next slices
read it -- `active_run` for the assistant pause, `recent` for the run bar,
`subscribe` for live views. What it never holds is content: identifiers and
codes in, identifiers and codes out.
"""

from __future__ import annotations

import asyncio
from typing import Any

from kiki.harness.models import HarnessStatusEvent, RunStatus
from kiki.routines.engine import RoutineEngine
from kiki.routines.models import build_routine, build_trigger
from kiki.routines.repository import RoutineRepository
from kiki.runtime.activity import (
    Activity,
    ActivityService,
    first_of_kind,
)
from kiki.storage.database import Database
from kiki.tools.audit import AuditLog
from kiki.tools.executor import ToolExecutor
from kiki.tools.gateway import ToolGateway
from kiki.tools.policy import RiskLevel, ToolPolicy
from kiki.tools.registry import ToolRegistry, ToolSpec
from kiki.tools.routine_gateway import RoutineToolGateway


def _status(run_id: str, status: RunStatus, code: str, *, terminal: bool | None = None) -> HarnessStatusEvent:
    return HarnessStatusEvent(
        run_id=run_id,
        status=status,
        message_code=code,
        terminal=status.is_terminal if terminal is None else terminal,
    )


# --- the ring ------------------------------------------------------------------


def test_recent_is_newest_first_and_bounded():
    service = ActivityService(limit=3)
    for index in range(5):
        service.record(Activity(kind="notice", code=f"n{index}", at=float(index)))
    entries = service.recent()
    assert [e.code for e in entries] == ["n4", "n3", "n2"]
    assert service.recent(limit=1)[0].code == "n4"


def test_limit_zero_is_refused():
    import pytest

    with pytest.raises(ValueError):
        ActivityService(limit=0)


def test_unknown_kinds_are_refused():
    import pytest

    with pytest.raises(ValueError):
        Activity(kind="geheim", code="x", at=1.0)
    with pytest.raises(ValueError):
        Activity(kind="run", code="", at=1.0)


def test_a_clockless_producer_still_gets_an_order():
    service = ActivityService(clock=lambda: 42.0)
    service.record(Activity(kind="notice", code="x", at=0.0))
    assert service.recent()[0].at == 42.0


# --- active run -----------------------------------------------------------------


def test_active_run_tracks_the_one_run():
    service = ActivityService()
    assert service.active_run() is None
    service.record_status(_status("run-1", RunStatus.RUNNING, "working"))
    active = service.active_run()
    assert active is not None and active.run_id == "run-1"
    service.record_status(_status("run-1", RunStatus.COMPLETED, "completed"))
    assert service.active_run() is None


def test_a_terminal_run_closes_even_under_newer_other_activity():
    service = ActivityService()
    service.record_status(_status("run-1", RunStatus.RUNNING, "working"))
    service.record_routine(code="fired", tool="audio.volume_set", run_id="routine-a")
    service.record_notice(key="battery.low", severity="warning")
    assert service.active_run() is not None
    service.record_status(_status("run-1", RunStatus.CANCELLED, "cancelled"))
    # The routine and notice entries must not resurrect the closed run.
    assert service.active_run() is None


# --- content-free by construction -------------------------------------------------


def test_subjects_are_screened():
    service = ActivityService()
    service.record(Activity(kind="routine", code="fired", at=1.0, subject="/home/martin/secret.txt"))
    service.record(Activity(kind="notice", code="n", at=2.0, subject="https://example.invalid/private"))
    service.record(Activity(kind="routine", code="fired", at=3.0, subject="ghp_testtoken"))
    subjects = [e.subject for e in service.recent()]
    assert subjects == ["[entfernt]", "[entfernt]", "[entfernt]"]


def test_entries_carry_no_prose_field():
    # Structural: the record has kinds, codes, ids and a screened subject.
    # There is no field a user text, an argument or an answer could hide in.
    fields = {f for f in Activity.__dataclass_fields__}
    assert fields == {"kind", "code", "at", "run_id", "subject", "terminal"}


# --- listeners --------------------------------------------------------------------


def test_listeners_see_every_entry_and_survive_breakage():
    service = ActivityService()
    seen: list[str] = []

    def _good(entry: Activity) -> None:
        seen.append(entry.code)

    def _broken(_entry: Activity) -> None:
        raise RuntimeError("Ansicht kaputt")

    service.subscribe(_broken)
    service.subscribe(_good)
    service.record(Activity(kind="notice", code="n1", at=1.0))
    service.record(Activity(kind="notice", code="n2", at=2.0))
    assert seen == ["n1", "n2"]
    service.unsubscribe(_good)
    service.record(Activity(kind="notice", code="n3", at=3.0))
    assert seen == ["n1", "n2"]


def test_first_of_kind_answers_one_question():
    service = ActivityService()
    service.record(Activity(kind="notice", code="n", at=1.0))
    service.record(Activity(kind="routine", code="fired", at=2.0, subject="audio.volume_set"))
    service.record(Activity(kind="routine", code="blocked", at=3.0, subject="memory_remember"))
    newest = first_of_kind(service.recent(), "routine")
    assert newest is not None and newest.code == "blocked"


# --- the routine producer, end to end ----------------------------------------------


def _volume_spec(counter: dict) -> ToolSpec:
    def _handler(_params: dict[str, Any]) -> dict[str, Any]:
        counter["runs"] += 1
        return {"ok": True}

    return ToolSpec(
        name="audio.volume_set",
        title="Lautstärke",
        description="Setzt die Lautstärke.",
        risk=RiskLevel.CONTROL,
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_handler,
        effect="Ändert die Lautstärke.",
        auto_allow=True,
        requires_integration=False,
        model_callable=True,
    )


class _Metrics:
    def __call__(self) -> dict[str, float]:
        return {"battery.percent": 10.0}


def test_routine_fires_land_in_the_ring(tmp_path):
    counter = {"runs": 0}
    registry = ToolRegistry()
    registry.register(_volume_spec(counter))
    db = Database(tmp_path / "kiki.db")
    executor = ToolExecutor(registry, ToolPolicy("balanced"), AuditLog(db))
    gateway = ToolGateway(executor, panic_check=lambda: False, integrations_check=lambda: True)
    repo = RoutineRepository(db)
    activity = ActivityService()
    engine = RoutineEngine(
        repo,
        RoutineToolGateway(gateway, repo, activity=activity),
        _Metrics(),
        panic_check=lambda: False,
        integrations_check=lambda: True,
    )
    repo.add(
        build_routine(
            name="Leiser bei leerem Akku",
            trigger=build_trigger("battery.percent", "lt", 15),
            tool_name="audio.volume_set",
            arguments={},
            cooldown_min=30,
        )
    )

    fired = asyncio.run(asyncio.wait_for(engine.tick(), timeout=10))

    assert len(fired) == 1 and fired[0]["ok"] is True
    entries = activity.recent()
    assert len(entries) == 1
    assert entries[0].kind == "routine"
    assert entries[0].code == "fired"
    assert entries[0].subject == "audio.volume_set"
    assert entries[0].run_id.startswith("routine-")


def test_refused_fires_land_too(tmp_path):
    counter = {"runs": 0}
    registry = ToolRegistry()
    registry.register(_volume_spec(counter))
    db = Database(tmp_path / "kiki.db")
    executor = ToolExecutor(registry, ToolPolicy("balanced"), AuditLog(db))
    gateway = ToolGateway(executor, panic_check=lambda: False, integrations_check=lambda: True)
    repo = RoutineRepository(db)
    activity = ActivityService()
    adapter = RoutineToolGateway(gateway, repo, activity=activity)

    async def scenario():
        return await adapter.run(
            "audio.volume_set",
            {"andere": "argumente"},
            panic=False,
            integrations_enabled=True,
        )

    result = asyncio.run(scenario())
    assert result.ok is False
    entries = activity.recent()
    assert len(entries) == 1
    assert entries[0].code == "refused"
    assert entries[0].subject == "audio.volume_set"
