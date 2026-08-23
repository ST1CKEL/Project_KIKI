"""Deterministic end-to-end cases. No model, no GPU, no network, no real /tmp.

Each case runs the whole harness against fakes and is judged on the things that
can be judged without a model: the terminal state, the error category, which
tools ran and in what order, the shape of the trace, and whether anything got
into that trace that should not be there.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from kiki.harness.models import ActionKind, ModelAction, RunStatus, ToolCall
from kiki.harness.runner import AgentRunner, RunBusyError
from kiki.harness.system_status import SystemStatusTool
from kiki.harness.tools import ToolRegistry
from tests.agent_fakes import (
    FailingTool,
    FinalOnlyModel,
    InvalidActionModel,
    OneToolThenFinalModel,
    RepeatedToolModel,
    WaitingModel,
)

SECRET_TEXT = (
    "Bitte prüf https://intern.example.com mit token=sk-live-4711 "
    "in /home/martin/.config/kiki/secrets.toml"
)


def _registry(tool=None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool or SystemStatusTool(uptime=lambda: 5.0))
    return registry


def _runner(adapter, tmp_path, *, registry=None, **kwargs) -> AgentRunner:
    return AgentRunner(
        adapter, registry or _registry(), trace_dir=tmp_path / "traces", **kwargs
    )


def _trace(tmp_path, run) -> list[dict]:
    path = tmp_path / "traces" / f"{run.id}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _events(records) -> list[str]:
    return [record["event"] for record in records]


def _assert_trace_is_well_formed(records, run) -> None:
    events = _events(records)
    assert events.count("run_started") == 1, events
    assert events.count("run_finished") == 1, events
    assert events[0] == "run_started"
    assert events[-1] == "run_finished"
    assert [record["sequence"] for record in records] == list(range(len(records)))

    finished = records[-1]
    assert finished["status"] == run.status.value
    assert finished["error_code"] == run.error_code
    assert finished["tool_calls"] == run.tool_calls
    assert isinstance(finished["duration_ms"], int)

    for record in records:
        blob = json.dumps(record, ensure_ascii=False)
        for forbidden in ("sk-live", "intern.example.com", "/home/", "secrets.toml", "token="):
            assert forbidden not in blob, (forbidden, blob)
    # The user's words are never in there; only how many there were.
    assert records[0]["user_text_length"] == len(SECRET_TEXT)


def _tool_events(records) -> list[str]:
    return [record["tool"] for record in records if record["event"] == "tool_completed"]


# --- the twelve cases -------------------------------------------------------


def test_case_direct_answer(tmp_path) -> None:
    run = asyncio.run(_runner(FinalOnlyModel("Mir geht es gut."), tmp_path).run(SECRET_TEXT))
    records = _trace(tmp_path, run)

    assert run.status is RunStatus.COMPLETED
    assert run.error_code is None
    assert run.final_text == "Mir geht es gut."
    assert run.tool_calls == 0
    assert _tool_events(records) == []
    assert _events(records) == ["run_started", "model_action_received", "run_finished"]
    _assert_trace_is_well_formed(records, run)


def test_case_status_question_uses_the_tool_once(tmp_path) -> None:
    adapter = OneToolThenFinalModel()
    run = asyncio.run(_runner(adapter, tmp_path).run(SECRET_TEXT))
    records = _trace(tmp_path, run)

    assert run.status is RunStatus.COMPLETED
    assert run.tool_calls == 1
    assert _tool_events(records) == ["system_status"]
    assert _events(records) == [
        "run_started", "model_action_received", "tool_requested", "tool_completed",
        "model_action_received", "run_finished",
    ]
    _assert_trace_is_well_formed(records, run)


def test_case_unknown_tool(tmp_path) -> None:
    adapter = OneToolThenFinalModel(tool="run_shell")
    run = asyncio.run(_runner(adapter, tmp_path).run(SECRET_TEXT))
    records = _trace(tmp_path, run)

    assert run.status is RunStatus.FAILED
    assert run.error_code == "unknown_tool"
    assert run.tool_calls == 0
    assert _tool_events(records) == []
    assert [r for r in records if r["event"] == "tool_requested"][0]["accepted"] is False
    _assert_trace_is_well_formed(records, run)


def test_case_arguments_for_a_tool_that_takes_none(tmp_path) -> None:
    adapter = OneToolThenFinalModel(arguments={"verbose": True})
    run = asyncio.run(_runner(adapter, tmp_path).run(SECRET_TEXT))
    records = _trace(tmp_path, run)

    assert run.status is RunStatus.FAILED
    assert run.error_code == "invalid_arguments"
    assert run.tool_calls == 0
    assert _tool_events(records) == []
    _assert_trace_is_well_formed(records, run)


def test_case_the_tool_itself_fails(tmp_path) -> None:
    run = asyncio.run(
        _runner(OneToolThenFinalModel(), tmp_path, registry=_registry(FailingTool())).run(
            SECRET_TEXT
        )
    )
    records = _trace(tmp_path, run)

    assert run.status is RunStatus.FAILED
    assert run.error_code == "tool_failed"
    assert run.tool_calls == 1
    assert [r for r in records if r["event"] == "tool_completed"][0]["ok"] is False
    _assert_trace_is_well_formed(records, run)


@pytest.mark.parametrize(
    "action",
    [
        ModelAction(ActionKind.FINAL),
        ModelAction(ActionKind.TOOL_CALL),
        ModelAction(ActionKind.FINAL, tool_call=ToolCall("system_status"), final_text="x"),
        "gar keine Aktion",
    ],
)
def test_case_invalid_model_action(tmp_path, action) -> None:
    run = asyncio.run(_runner(InvalidActionModel(action), tmp_path).run(SECRET_TEXT))
    records = _trace(tmp_path, run)

    assert run.status is RunStatus.FAILED
    assert run.error_code == "model_protocol_error"
    assert run.tool_calls == 0
    assert [r for r in records if r["event"] == "model_action_received"][0]["valid"] is False
    _assert_trace_is_well_formed(records, run)


def test_case_repeated_tool_calls_hit_the_limit(tmp_path) -> None:
    run = asyncio.run(
        _runner(RepeatedToolModel(), tmp_path, max_tool_calls=3, max_steps=10).run(SECRET_TEXT)
    )
    records = _trace(tmp_path, run)

    assert run.status is RunStatus.LIMIT_REACHED
    assert run.error_code == "tool_call_limit"
    assert run.tool_calls == 3
    assert _tool_events(records) == ["system_status"] * 3
    assert run.final_text is None
    _assert_trace_is_well_formed(records, run)


def test_case_cancel_before_any_model_action(tmp_path) -> None:
    """The run is cancelled while the first model step is still pending, so no
    tool ever gets the chance to run."""
    adapter = WaitingModel()
    runner = _runner(adapter, tmp_path)

    async def go():
        task = asyncio.create_task(runner.run(SECRET_TEXT))
        for _ in range(2000):
            if runner.active_run_id:
                break
            await asyncio.sleep(0.001)
        runner.cancel(runner.active_run_id)
        return await task

    run = asyncio.run(go())
    records = _trace(tmp_path, run)

    assert run.status is RunStatus.CANCELLED
    assert run.error_code is None
    assert run.final_text is None
    assert run.tool_calls == 0
    assert _tool_events(records) == []
    assert "run_cancelled" in _events(records)
    _assert_trace_is_well_formed(records, run)


def test_case_cancel_before_the_tool_runs(tmp_path) -> None:
    ran: list[int] = []
    holder: dict = {}

    class _WatchingTool(SystemStatusTool):
        async def execute(self, arguments):
            ran.append(1)
            return await super().execute(arguments)

    class _CancellingRegistry(ToolRegistry):
        def validate(self, call):
            holder["runner"].cancel(holder["runner"].active_run_id)
            return super().validate(call)

    registry = _CancellingRegistry()
    registry.register(_WatchingTool(uptime=lambda: 1.0))
    runner = _runner(RepeatedToolModel(), tmp_path, registry=registry)
    holder["runner"] = runner

    run = asyncio.run(runner.run(SECRET_TEXT))
    records = _trace(tmp_path, run)

    assert run.status is RunStatus.CANCELLED
    assert run.tool_calls == 0
    assert ran == []
    assert _tool_events(records) == []
    _assert_trace_is_well_formed(records, run)


def test_case_cancel_after_the_tool_result(tmp_path) -> None:
    holder: dict = {}

    class _CancellingTool(SystemStatusTool):
        async def execute(self, arguments):
            holder["runner"].cancel(holder["runner"].active_run_id)
            return await super().execute(arguments)

    adapter = RepeatedToolModel()
    runner = _runner(adapter, tmp_path, registry=_registry(_CancellingTool(uptime=lambda: 1.0)))
    holder["runner"] = runner

    run = asyncio.run(runner.run(SECRET_TEXT))
    records = _trace(tmp_path, run)

    assert run.status is RunStatus.CANCELLED
    assert run.tool_calls == 1
    assert adapter.calls == 1, "kein zweiter Modellschritt nach dem Cancel"
    assert _tool_events(records) == ["system_status"]
    _assert_trace_is_well_formed(records, run)


def test_case_a_second_run_is_refused_and_the_first_survives(tmp_path) -> None:
    adapter = WaitingModel()
    runner = _runner(adapter, tmp_path)

    async def go():
        task = asyncio.create_task(runner.run(SECRET_TEXT))
        for _ in range(2000):
            if runner.active_run_id:
                break
            await asyncio.sleep(0.001)
        first_id = runner.active_run_id
        with pytest.raises(RunBusyError) as excinfo:
            await runner.run("zweiter")
        assert runner.active_run_id == first_id
        runner.cancel(first_id)
        return await task, excinfo.value, first_id

    run, error, first_id = asyncio.run(go())
    assert error.error_code == "run_busy"
    assert run.id == first_id
    assert run.status is RunStatus.CANCELLED
    # The refused run left no trace of its own; only the real one exists.
    assert [path.name for path in (tmp_path / "traces").iterdir()] == [f"{first_id}.jsonl"]


def test_case_a_new_run_after_a_terminal_one(tmp_path) -> None:
    runner = _runner(OneToolThenFinalModel(), tmp_path)

    async def go():
        first = await runner.run(SECRET_TEXT)
        runner._adapter = FinalOnlyModel("Und noch einmal.")
        second = await runner.run(SECRET_TEXT)
        return first, second

    first, second = asyncio.run(go())
    assert first.status is second.status is RunStatus.COMPLETED
    assert first.id != second.id
    for run in (first, second):
        _assert_trace_is_well_formed(_trace(tmp_path, run), run)


# --- audits -----------------------------------------------------------------


def test_no_run_writes_outside_the_trace_directory(tmp_path) -> None:
    """The only thing the harness may touch on disk is the directory it was
    handed, and only under the run's own name."""
    runner = _runner(OneToolThenFinalModel(), tmp_path)
    run = asyncio.run(runner.run(SECRET_TEXT))

    written = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert written == ["traces", f"traces/{run.id}.jsonl"]


def test_the_harness_imports_nothing_it_should_not() -> None:
    import subprocess
    import sys

    probe = """
import sys
started = []
sys.addaudithook(
    lambda event, args: started.append(event)
    if event in {"subprocess.Popen", "os.system", "os.exec", "socket.connect", "exec", "compile"}
    else None
)
import kiki.harness  # noqa: F401
forbidden = [
    name for name in (
        "gi", "gi.repository", "torch", "numpy", "qwen_tts", "httpx",
        "subprocess", "socket", "kiki.voice", "kiki.ui", "kiki.ai",
    )
    if name in sys.modules
]
assert not forbidden, forbidden
assert "subprocess.Popen" not in started, started
print("sauber")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "sauber" in result.stdout


def test_the_source_holds_no_dangerous_construct() -> None:
    from pathlib import Path

    package = Path(__file__).resolve().parents[1] / "src" / "kiki" / "agent"
    for path in package.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for forbidden in ("shell=True", "subprocess", "os.system", "eval(", "exec("):
            assert forbidden not in source, (path.name, forbidden)


def test_every_reported_error_code_is_in_the_vocabulary(tmp_path) -> None:
    from kiki.harness.models import ERROR_CODES

    cases = [
        (OneToolThenFinalModel(tool="nope"), _registry()),
        (OneToolThenFinalModel(arguments={"x": 1}), _registry()),
        (OneToolThenFinalModel(), _registry(FailingTool())),
        (InvalidActionModel(), _registry()),
    ]
    for adapter, registry in cases:
        run = asyncio.run(_runner(adapter, tmp_path, registry=registry).run("x"))
        assert run.error_code in ERROR_CODES, run.error_code
