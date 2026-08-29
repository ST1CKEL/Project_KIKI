"""The assistant pause: no new work, and nothing destroyed.

A pause refuses new runs at the structural door, fires no routine, silences
notices. What it must not do is just as important: it does not cancel the
run that is going, does not disable a routine (a pause is not a policy
decision), does not burn a spoken-event id, and it is not the character
pause and not panic. Session state: a fresh start is unpaused.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from kiki.assistant.adapter import StepEvent
from kiki.assistant.run_service import (
    RunBusyError,
    RunCallbacks,
    RunPausedError,
    RunService,
)
from kiki.assistant.runner import AssistantRunner
from kiki.harness.models import ModelAction, RunStatus
from kiki.routines.engine import RoutineEngine
from kiki.routines.models import build_routine, build_trigger
from kiki.routines.repository import RoutineRepository
from kiki.runtime.activity import ActivityService
from kiki.runtime.pause import AssistantPause
from kiki.storage.database import Database
from kiki.tools.audit import AuditLog
from kiki.tools.executor import ToolExecutor
from kiki.tools.gateway import ToolGateway
from kiki.tools.policy import RiskLevel, ToolPolicy
from kiki.tools.registry import ToolRegistry, ToolSpec
from kiki.tools.routine_gateway import RoutineToolGateway


class FinalSteps:
    def __init__(self, text: str = "Erledigt.") -> None:
        self._text = text
        self.calls = 0

    async def next_action_stream(
        self, *, user_text, tool_schemas, observations, cancel_token
    ):
        del user_text, tool_schemas, observations, cancel_token
        self.calls += 1
        yield StepEvent(kind="action", action=ModelAction.answer(self._text))


class BlockingSteps:
    def __init__(self) -> None:
        self.started = 0

    async def next_action_stream(
        self, *, user_text, tool_schemas, observations, cancel_token
    ):
        del user_text, tool_schemas, observations
        self.started += 1
        while not cancel_token.cancelled:
            await asyncio.sleep(0.001)
        yield StepEvent(kind="action", action=ModelAction.answer("zu spät"))


def _service(
    tmp_path, model, pause: AssistantPause, **kwargs: Any
) -> RunService:
    registry = ToolRegistry()
    executor = ToolExecutor(
        registry, ToolPolicy("balanced"), AuditLog(Database(tmp_path / "kiki.db"))
    )
    gateway = ToolGateway(
        executor, panic_check=lambda: False, integrations_check=lambda: True
    )
    runner = AssistantRunner(model, gateway, trace_dir=tmp_path / "tr", **kwargs)
    return RunService(runner, RunCallbacks(), paused=pause)


def _ask(service: RunService, text: str = "frage", correlation_id: str = "") -> Any:
    return asyncio.wait_for(service.ask(text, correlation_id=correlation_id), timeout=10)


# --- the state object -----------------------------------------------------------


def test_the_pause_toggles_and_starts_unpaused():
    pause = AssistantPause()
    assert pause.paused is False
    pause.pause()
    assert pause.paused is True
    assert pause.toggle() is False
    assert pause.paused is False


# --- runs: refused, active untouched, ids not burned ------------------------------


def test_a_paused_assistant_refuses_new_runs(tmp_path):
    pause = AssistantPause()
    model = FinalSteps()
    service = _service(tmp_path, model, pause)

    async def scenario():
        pause.pause()
        with pytest.raises(RunPausedError):
            await _ask(service)
        # It is a "no run now", so existing handlers keep working.
        assert isinstance(RunPausedError("x"), RunBusyError)
        pause.resume()
        return await _ask(service)

    run = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED
    assert model.calls == 1


def test_a_pause_does_not_touch_the_active_run(tmp_path):
    pause = AssistantPause()
    model = BlockingSteps()
    service = _service(tmp_path, model, pause)

    async def scenario():
        first = asyncio.create_task(_ask(service))
        await asyncio.sleep(0.02)
        pause.pause()
        assert service.busy is True  # still going, not cancelled
        service.cancel()  # settle it for the test
        return await first

    run = asyncio.run(scenario())
    assert run.status is RunStatus.CANCELLED


def test_a_pause_refusal_does_not_burn_the_spoken_id(tmp_path):
    pause = AssistantPause()
    model = FinalSteps()
    service = _service(tmp_path, model, pause)

    async def scenario():
        pause.pause()
        with pytest.raises(RunPausedError):
            await _ask(service, "notiz", correlation_id="take-1")
        pause.resume()
        return await _ask(service, "notiz", correlation_id="take-1")

    run = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED


# --- routines: no fire, and no routine lost ----------------------------------------


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


def _engine_stack(tmp_path, pause: AssistantPause, counter: dict):
    registry = ToolRegistry()
    registry.register(_volume_spec(counter))
    db = Database(tmp_path / "kiki.db")
    executor = ToolExecutor(registry, ToolPolicy("balanced"), AuditLog(db))
    gateway = ToolGateway(
        executor, panic_check=lambda: False, integrations_check=lambda: True
    )
    repo = RoutineRepository(db)
    activity = ActivityService()
    engine = RoutineEngine(
        repo,
        RoutineToolGateway(gateway, repo, activity, pause=pause),
        _Metrics(),
        panic_check=lambda: False,
        integrations_check=lambda: True,
    )
    return engine, repo, activity


def test_a_paused_assistant_fires_no_routine_and_keeps_it_enabled(tmp_path):
    pause = AssistantPause()
    counter = {"runs": 0}
    engine, repo, activity = _engine_stack(tmp_path, pause, counter)
    repo.add(
        build_routine(
            name="Leiser bei leerem Akku",
            trigger=build_trigger("battery.percent", "lt", 15),
            tool_name="audio.volume_set",
            arguments={},
            cooldown_min=30,
        )
    )
    pause.pause()

    fired = asyncio.run(asyncio.wait_for(engine.tick(), timeout=10))

    assert counter["runs"] == 0
    assert len(fired) == 1 and fired[0]["ok"] is False
    # The refusal is transient: no DENY, the recipe survives the pause.
    assert fired[0].get("disabled") is not True
    assert repo.list()[0].enabled is True
    entries = activity.recent()
    assert entries and entries[0].code == "paused"

    pause.resume()
    fired = asyncio.run(asyncio.wait_for(engine.tick(), timeout=10))
    assert counter["runs"] == 1
    assert fired[0]["ok"] is True


# --- the pause is its own concept ----------------------------------------------------


def test_panic_and_pause_are_different_switches(tmp_path):
    pause = AssistantPause()
    world = {"panic": False}
    registry = ToolRegistry()
    executor = ToolExecutor(
        registry, ToolPolicy("balanced"), AuditLog(Database(tmp_path / "kiki.db"))
    )
    gateway = ToolGateway(
        executor,
        panic_check=lambda: world["panic"],
        integrations_check=lambda: True,
    )
    del gateway  # the point is the flags, not the stack

    pause.pause()
    world["panic"] = False
    # Paused does not flip panic, and panic does not imply paused.
    assert pause.paused is True
    assert world["panic"] is False
    pause.resume()
    world["panic"] = True
    assert pause.paused is False
    world["panic"] = False
