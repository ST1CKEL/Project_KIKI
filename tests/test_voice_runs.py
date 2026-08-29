"""Voice on the run service: one spoken event, at most one run.

The binding lives where runs are born. `correlation_id` names the event that
asked -- one push-to-talk take, one captured wake command -- and the service
answers the same id at most once, no matter how often it is delivered. A
refusal does not burn the id: an utterance KIKI could not take because she
was busy may be said again.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from kiki.assistant.adapter import StepEvent
from kiki.assistant.run_service import (
    CORRELATION_MEMORY,
    DuplicateCorrelationError,
    RunCallbacks,
    RunService,
    failure_text,
)
from kiki.assistant.runner import AssistantRunner
from kiki.harness.confirmation import ConfirmationRequest
from kiki.harness.models import (
    CancelToken,
    HarnessStatusEvent,
    ModelAction,
    RunBusyError,
    RunStatus,
    ToolResult,
)
from kiki.storage.database import Database
from kiki.tools.audit import AuditLog
from kiki.tools.executor import ToolExecutor
from kiki.tools.gateway import ToolGateway
from kiki.tools.policy import RiskLevel, ToolPolicy
from kiki.tools.registry import ToolRegistry, ToolSpec

READ_SCHEMA = {"type": "object", "properties": {}, "required": []}


class FinalSteps:
    """Answers on the first step."""

    def __init__(self, text: str = "Erledigt.") -> None:
        self._text = text
        self.calls = 0

    async def next_action_stream(
        self,
        *,
        user_text: str,
        tool_schemas: list[dict[str, Any]],
        observations: list[ToolResult],
        cancel_token: CancelToken,
    ) -> Any:
        del user_text, tool_schemas, observations, cancel_token
        self.calls += 1
        yield StepEvent(kind="action", action=ModelAction.answer(self._text))


class BlockOnceThenAnswer:
    """Blocks for the first run until cancelled; answers every later one.

    One fake, two behaviours: the blocker that gets cancelled, and the run
    the re-said utterance deserves -- which must actually finish, or the
    scenario proves nothing about anything but its own model.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def next_action_stream(
        self,
        *,
        user_text: str,
        tool_schemas: list[dict[str, Any]],
        observations: list[ToolResult],
        cancel_token: CancelToken,
    ) -> Any:
        del user_text, tool_schemas, observations
        self.calls += 1
        if self.calls == 1:
            while not cancel_token.cancelled:
                await asyncio.sleep(0.001)
        yield StepEvent(kind="action", action=ModelAction.answer("jetzt fertig"))


class Recorder:
    def __init__(self) -> None:
        self.status: list[HarnessStatusEvent] = []
        self.answers: list[str] = []
        self.spoken: list[str] = []
        self.confirmations: list[ConfirmationRequest] = []

    def callbacks(self) -> RunCallbacks:
        return RunCallbacks(
            on_status=self.status.append,
            on_answer=self.answers.append,
            on_confirmation=self.confirmations.append,
            on_speak=self.spoken.append,
        )


def _service(
    tmp_path: Path,
    model: Any,
    recorder: Recorder | None = None,
    *specs: ToolSpec,
) -> RunService:
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    executor = ToolExecutor(
        registry, ToolPolicy("balanced"), AuditLog(Database(tmp_path / "kiki.db"))
    )
    gateway = ToolGateway(
        executor,
        panic_check=lambda: False,
        integrations_check=lambda: True,
    )
    runner = AssistantRunner(
        model,
        gateway,
        trace_dir=tmp_path / "traces",
        max_steps=4,
        max_tool_calls=6,
    )
    return RunService(runner, (recorder or Recorder()).callbacks())


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        title=name,
        description=f"{name} description",
        risk=RiskLevel.READ,
        parameters=READ_SCHEMA,
        handler=lambda _p: {"ran": name},
        effect=f"{name} effect",
        auto_allow=True,
        requires_integration=False,
        model_callable=True,
    )


def _ask(service: RunService, text: str, correlation_id: str = "") -> Any:
    return asyncio.wait_for(
        service.ask(text, correlation_id=correlation_id), timeout=10
    )


# --- one event, at most one run ------------------------------------------------


def test_the_same_event_never_starts_a_second_run(tmp_path):
    service = _service(tmp_path, FinalSteps())

    async def scenario():
        first = await _ask(service, "stelle die uhr", correlation_id="take-1")
        with pytest.raises(DuplicateCorrelationError):
            await _ask(service, "stelle die uhr", correlation_id="take-1")
        with pytest.raises(DuplicateCorrelationError):
            await _ask(service, "andauernd doppelt", correlation_id="take-1")
        return first

    run = asyncio.run(scenario())
    # The duplicate arrived after the run settled and still did not run.
    assert run.status is RunStatus.COMPLETED


def test_a_duplicate_during_an_active_run_leaves_it_untouched(tmp_path):
    recorder = Recorder()
    service = _service(tmp_path, BlockOnceThenAnswer(), recorder)

    async def scenario():
        first = asyncio.create_task(_ask(service, "lange frage", correlation_id="take-1"))
        await asyncio.sleep(0.02)
        with pytest.raises(RunBusyError):
            await _ask(service, "zweite aufnahme", correlation_id="take-2")
        with pytest.raises(DuplicateCorrelationError):
            await _ask(service, "lange frage", correlation_id="take-1")
        service.cancel()
        return await first

    run = asyncio.run(scenario())
    assert run.status is RunStatus.CANCELLED


def test_distinct_events_run_one_after_the_other(tmp_path):
    model = FinalSteps()
    service = _service(tmp_path, model)

    async def scenario():
        await _ask(service, "erste frage", correlation_id="take-1")
        await _ask(service, "zweite frage", correlation_id="take-2")
        await _ask(service, "ohne id geht auch")

    asyncio.run(scenario())
    assert model.calls == 3


# --- refusals do not burn the id ----------------------------------------------


def test_a_busy_refusal_keeps_the_id_reusable(tmp_path):
    service = _service(tmp_path, BlockOnceThenAnswer())

    async def scenario():
        blocker = asyncio.create_task(_ask(service, "erste aufnahme", correlation_id="take-A"))
        await asyncio.sleep(0.02)
        # A different utterance arrives while the blocker runs: refused. It
        # never got a run, so its id is not burned.
        with pytest.raises(RunBusyError):
            await _ask(service, "zweite aufnahme", correlation_id="take-1")
        service.cancel()
        await blocker
        # Said again once KIKI is free: this time it runs.
        return await _ask(service, "zweite aufnahme", correlation_id="take-1")

    run = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED


def test_a_run_without_an_id_is_never_deduplicated(tmp_path):
    service = _service(tmp_path, FinalSteps())

    async def scenario():
        await _ask(service, "dieselbe frage")
        await _ask(service, "dieselbe frage")

    asyncio.run(scenario())
    # Typed runs carry no event identity; nothing here deduplicates them.


# --- the memory bound ----------------------------------------------------------


def test_remembered_ids_stay_bounded(tmp_path):
    model = FinalSteps()
    service = _service(tmp_path, model)

    async def scenario():
        for index in range(CORRELATION_MEMORY + 40):
            await _ask(service, f"frage {index}", correlation_id=f"take-{index}")
        return service

    service = asyncio.run(scenario())
    remembered = service._correlations  # noqa: SLF001 - the bound is the point
    assert len(remembered) <= CORRELATION_MEMORY
    # Oldest evicted, newest kept.
    assert "take-0" not in remembered
    assert f"take-{CORRELATION_MEMORY + 39}" in remembered


def test_a_duplicate_is_still_refused_after_the_run_settled(tmp_path):
    recorder = Recorder()
    service = _service(tmp_path, FinalSteps("Fertig."), recorder)

    async def scenario():
        await _ask(service, "notiz diktieren", correlation_id="take-7")

    asyncio.run(scenario())
    assert recorder.answers == ["Fertig."]

    async def again():
        with pytest.raises(DuplicateCorrelationError):
            await _ask(service, "notiz diktieren", correlation_id="take-7")

    asyncio.run(again())
    # Still exactly one answer for that event.
    assert recorder.answers == ["Fertig."]
    assert failure_text("provider_error")  # the vocabulary stays importable
