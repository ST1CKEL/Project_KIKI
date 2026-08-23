"""The loop itself: limits, cancellation, the busy guard, and the invariants."""

from __future__ import annotations

import asyncio

import pytest

from kiki.harness.models import (
    ERROR_CODES,
    ActionKind,
    AgentRun,
    CancelToken,
    ModelAction,
    RunStatus,
    ToolCall,
    validate_action,
)
from kiki.harness.runner import AgentRunner, ModelAdapter, RunBusyError
from kiki.harness.system_status import SystemStatusTool
from kiki.harness.tools import ToolRegistry
from tests.agent_fakes import (
    FinalOnlyModel,
    InvalidActionModel,
    OneToolThenFinalModel,
    RepeatedToolModel,
    WaitingModel,
)


def _registry(tool=None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool or SystemStatusTool(uptime=lambda: 5.0))
    return registry


def _runner(adapter, tmp_path, *, registry=None, **kwargs) -> AgentRunner:
    return AgentRunner(
        adapter, registry or _registry(), trace_dir=tmp_path / "traces", **kwargs
    )


# --- the run object's invariants --------------------------------------------


def test_a_run_settles_exactly_once() -> None:
    run = AgentRun(user_text="x")
    run.start()
    run.finish(RunStatus.COMPLETED, final_text="fertig")
    with pytest.raises(RuntimeError):
        run.finish(RunStatus.FAILED, error_code="tool_failed")


def test_a_terminal_run_never_starts_again() -> None:
    run = AgentRun(user_text="x")
    run.start()
    run.finish(RunStatus.CANCELLED)
    assert run.is_terminal
    with pytest.raises(RuntimeError):
        run.start()


def test_completed_needs_an_answer() -> None:
    run = AgentRun(user_text="x")
    run.start()
    with pytest.raises(ValueError):
        run.finish(RunStatus.COMPLETED, final_text="  ")


@pytest.mark.parametrize("status", [RunStatus.FAILED, RunStatus.LIMIT_REACHED])
def test_a_failure_cannot_carry_an_answer(status) -> None:
    """No terminal state but COMPLETED may look like a success."""
    run = AgentRun(user_text="x")
    run.start()
    with pytest.raises(ValueError):
        run.finish(status, final_text="sieht gut aus", error_code="tool_failed")


def test_a_failure_needs_a_known_category() -> None:
    run = AgentRun(user_text="x")
    run.start()
    with pytest.raises(ValueError):
        run.finish(RunStatus.FAILED, error_code="ausgedacht")


def test_running_is_not_terminal() -> None:
    assert RunStatus.RUNNING.is_terminal is False
    assert RunStatus.PENDING.is_terminal is False
    for status in (RunStatus.COMPLETED, RunStatus.CANCELLED, RunStatus.FAILED,
                   RunStatus.LIMIT_REACHED):
        assert status.is_terminal is True


# --- the model action contract ----------------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        ModelAction(ActionKind.FINAL),                                   # no text
        ModelAction(ActionKind.FINAL, final_text="   "),                 # blank
        ModelAction(ActionKind.FINAL, tool_call=ToolCall("system_status"), final_text="x"),
        ModelAction(ActionKind.TOOL_CALL),                               # no call
        ModelAction(ActionKind.TOOL_CALL, tool_call=ToolCall("system_status"), final_text="x"),
        "nicht einmal eine Aktion",
        None,
    ],
)
def test_a_contradictory_action_is_refused(action) -> None:
    assert validate_action(action) == "model_protocol_error"


def test_a_well_formed_action_passes() -> None:
    assert validate_action(ModelAction.answer("fertig")) == ""
    assert validate_action(ModelAction.call("system_status")) == ""


def test_the_adapter_protocol_is_structural() -> None:
    assert isinstance(FinalOnlyModel(), ModelAdapter)


# --- limits -----------------------------------------------------------------

def test_a_model_that_only_asks_for_tools_hits_the_call_limit(tmp_path) -> None:
    adapter = RepeatedToolModel()
    runner = _runner(adapter, tmp_path, max_tool_calls=2, max_steps=10)
    run = asyncio.run(runner.run("wie geht es dir?"))

    assert run.status is RunStatus.LIMIT_REACHED
    assert run.error_code == "tool_call_limit"
    assert run.tool_calls == 2
    assert run.final_text is None


def test_the_step_limit_ends_the_run(tmp_path) -> None:
    adapter = RepeatedToolModel()
    runner = _runner(adapter, tmp_path, max_tool_calls=99, max_steps=3)
    run = asyncio.run(runner.run("status?"))

    assert run.status is RunStatus.LIMIT_REACHED
    assert run.error_code == "step_limit"
    assert adapter.calls == 3


def test_nonsense_limits_are_refused(tmp_path) -> None:
    for kwargs in ({"max_steps": 0}, {"max_tool_calls": -1}):
        with pytest.raises(ValueError):
            _runner(FinalOnlyModel(), tmp_path, **kwargs)


# --- the observations the model sees ----------------------------------------


def test_the_tool_result_comes_back_to_the_model(tmp_path) -> None:
    adapter = OneToolThenFinalModel()
    run = asyncio.run(_runner(adapter, tmp_path).run("status?"))

    assert run.status is RunStatus.COMPLETED
    assert adapter.calls == 2
    assert adapter.seen_observations[0] == []
    observed = adapter.seen_observations[1]
    assert len(observed) == 1
    assert observed[0].data["agent_harness"] == "available"


def test_the_model_only_ever_sees_registered_tools(tmp_path) -> None:
    seen: list[list[dict]] = []

    class _Watching(FinalOnlyModel):
        async def next_action(self, *, user_text, tool_schemas, observations, cancel_token):
            seen.append(tool_schemas)
            return await super().next_action(
                user_text=user_text, tool_schemas=tool_schemas,
                observations=observations, cancel_token=cancel_token,
            )

    asyncio.run(_runner(_Watching(), tmp_path).run("hallo"))
    assert [schema["name"] for schema in seen[0]] == ["system_status"]


# --- cancellation -----------------------------------------------------------


def _cancel_when_running(runner, task, poll=0.001):
    async def _fire():
        for _ in range(2000):
            if runner.active_run_id:
                runner.cancel(runner.active_run_id)
                return
            await asyncio.sleep(poll)
        raise AssertionError("Run wurde nie aktiv")

    return _fire()


def test_a_cancel_while_the_model_thinks_ends_the_run(tmp_path) -> None:
    adapter = WaitingModel()
    runner = _runner(adapter, tmp_path)

    async def go():
        task = asyncio.create_task(runner.run("status?"))
        await _cancel_when_running(runner, task)
        return await task

    run = asyncio.run(go())
    assert run.status is RunStatus.CANCELLED
    assert run.tool_calls == 0
    assert run.final_text is None
    assert run.error_code is None
    assert adapter.released is True


def test_a_cancel_between_validation_and_execution_stops_the_tool(tmp_path) -> None:
    """The check right before execute(): a cancel that arrived while the model
    was thinking must not still cost a tool run."""
    ran: list[int] = []

    class _WatchingTool(SystemStatusTool):
        async def execute(self, arguments):
            ran.append(1)
            return await super().execute(arguments)

    holder: dict = {}

    class _CancellingRegistry(ToolRegistry):
        def validate(self, call):
            holder["runner"].cancel(holder["runner"].active_run_id)
            return super().validate(call)

    registry = _CancellingRegistry()
    registry.register(_WatchingTool(uptime=lambda: 1.0))
    runner = _runner(RepeatedToolModel(), tmp_path, registry=registry)
    holder["runner"] = runner

    run = asyncio.run(runner.run("status?"))
    assert run.status is RunStatus.CANCELLED
    assert run.tool_calls == 0
    assert ran == []


def test_a_cancel_after_the_tool_stops_the_next_model_step(tmp_path) -> None:
    holder: dict = {}

    class _CancellingTool(SystemStatusTool):
        async def execute(self, arguments):
            holder["runner"].cancel(holder["runner"].active_run_id)
            return await super().execute(arguments)

    adapter = RepeatedToolModel()
    runner = _runner(adapter, tmp_path, registry=_registry(_CancellingTool(uptime=lambda: 1.0)))
    holder["runner"] = runner

    run = asyncio.run(runner.run("status?"))
    assert run.status is RunStatus.CANCELLED
    assert run.tool_calls == 1
    assert adapter.calls == 1, "nach dem Cancel darf kein zweiter Modellschritt kommen"


def test_cancelling_twice_is_harmless(tmp_path) -> None:
    adapter = WaitingModel()
    runner = _runner(adapter, tmp_path)

    async def go():
        task = asyncio.create_task(runner.run("status?"))
        for _ in range(2000):
            if runner.active_run_id:
                break
            await asyncio.sleep(0.001)
        run_id = runner.active_run_id
        assert runner.cancel(run_id) is True
        assert runner.cancel(run_id) is True
        return await task

    assert asyncio.run(go()).status is RunStatus.CANCELLED


def test_a_cancel_for_another_id_does_nothing(tmp_path) -> None:
    adapter = OneToolThenFinalModel()
    runner = _runner(adapter, tmp_path)

    async def go():
        assert runner.cancel("run-gibtsnicht") is False
        return await runner.run("status?")

    run = asyncio.run(go())
    assert run.status is RunStatus.COMPLETED
    assert runner.cancel(run.id) is False, "ein beendeter Run nimmt keinen Cancel mehr an"


def test_a_new_run_works_after_a_cancel(tmp_path) -> None:
    runner = _runner(WaitingModel(), tmp_path)

    async def go():
        task = asyncio.create_task(runner.run("erster"))
        await _cancel_when_running(runner, task)
        first = await task
        runner._adapter = FinalOnlyModel("jetzt geht es")
        second = await runner.run("zweiter")
        return first, second

    first, second = asyncio.run(go())
    assert first.status is RunStatus.CANCELLED
    assert second.status is RunStatus.COMPLETED
    assert second.final_text == "jetzt geht es"


def test_no_task_survives_a_cancelled_run(tmp_path) -> None:
    runner = _runner(WaitingModel(), tmp_path)

    async def go():
        before = len(asyncio.all_tasks())
        task = asyncio.create_task(runner.run("status?"))
        await _cancel_when_running(runner, task)
        await task
        await asyncio.sleep(0)
        return before, len(asyncio.all_tasks())

    before, after = asyncio.run(go())
    assert after <= before + 1


# --- one run at a time ------------------------------------------------------


def test_a_second_run_is_refused_while_one_is_active(tmp_path) -> None:
    adapter = WaitingModel()
    runner = _runner(adapter, tmp_path)

    async def go():
        task = asyncio.create_task(runner.run("erster"))
        for _ in range(2000):
            if runner.active_run_id:
                break
            await asyncio.sleep(0.001)
        active = runner.active_run_id
        with pytest.raises(RunBusyError) as excinfo:
            await runner.run("zweiter")
        # The refusal must not disturb the run that was already going.
        assert runner.active_run_id == active
        runner.cancel(active)
        return await task, excinfo.value

    run, error = asyncio.run(go())
    assert run.status is RunStatus.CANCELLED
    assert error.error_code == "run_busy"
    assert error.error_code in ERROR_CODES


def test_the_runner_is_free_again_after_a_terminal_run(tmp_path) -> None:
    runner = _runner(FinalOnlyModel(), tmp_path)

    async def go():
        first = await runner.run("eins")
        assert runner.busy is False
        second = await runner.run("zwei")
        return first, second

    first, second = asyncio.run(go())
    assert first.id != second.id
    assert first.status is second.status is RunStatus.COMPLETED


# --- trace write failure ----------------------------------------------------


def test_a_broken_trace_ends_the_run_instead_of_being_ignored(tmp_path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("Datei, kein Verzeichnis", encoding="utf-8")
    runner = AgentRunner(FinalOnlyModel(), _registry(), trace_dir=blocked)

    run = asyncio.run(runner.run("hallo"))
    assert run.status is RunStatus.FAILED
    assert run.error_code == "trace_write_failed"


def test_a_cancel_token_is_a_flag_not_an_exception() -> None:
    token = CancelToken()
    assert token.cancelled is False
    token.cancel()
    token.cancel()
    assert token.cancelled is True


def test_an_invalid_model_action_never_reaches_a_tool(tmp_path) -> None:
    ran: list[int] = []

    class _WatchingTool(SystemStatusTool):
        async def execute(self, arguments):
            ran.append(1)
            return await super().execute(arguments)

    run = asyncio.run(
        _runner(InvalidActionModel(), tmp_path, registry=_registry(_WatchingTool())).run("x")
    )
    assert run.status is RunStatus.FAILED
    assert run.error_code == "model_protocol_error"
    assert ran == []
