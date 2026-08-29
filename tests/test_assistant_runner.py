"""The unified assistant runner: run shell, streaming steps, one gateway.

These tests carry over the promises of both predecessors and prove them
against the one class that now holds them: exactly one terminal state, cancel
by id, a trace without content, refusals that fail closed, and every
execution behind the `ToolGateway` with the live policy read again.
"""

from __future__ import annotations

import ast
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from kiki.ai.provider import StreamChunk
from kiki.assistant.adapter import ProviderStepAdapter, StepEvent
from kiki.assistant.runner import AssistantRunner, RunnerEvent
from kiki.harness.adapter import ProviderError
from kiki.harness.confirmation import ConfirmationError
from kiki.harness.models import (
    ActionKind,
    CancelToken,
    ModelAction,
    RunStatus,
    ToolResult,
)
from kiki.harness.trace import TraceRecorder
from kiki.storage.database import Database
from kiki.tools.audit import AuditLog
from kiki.tools.executor import ToolExecutor
from kiki.tools.gateway import ToolGateway
from kiki.tools.policy import RiskLevel, ToolPolicy
from kiki.tools.registry import ToolRegistry, ToolSpec

SRC = Path(__file__).resolve().parent.parent / "src"

READ_SCHEMA = {"type": "object", "properties": {}, "required": []}
TEXT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}

USER_TEXT = "Bitte behalte sk-test-secret für mich."


# -- fakes --------------------------------------------------------------------


class World:
    """The live sources the gateway reads. A test flips them when it wants."""

    def __init__(self) -> None:
        self.panic = False
        self.integrations = True


class ScriptedModel:
    """Plays scripted steps. Each *argument* is one step: a list of StepEvents
    that ends in exactly one action. Records what every round was given.

    `on_step` runs when an adapter round starts -- the one deterministic point
    between two steps, since the runner builds the tool list before calling in
    and re-reads the policy again before every execution.
    """

    def __init__(self, *steps: list[StepEvent], on_step: Any = None) -> None:
        self._steps = [list(step) for step in steps]
        self._on_step = on_step
        self.calls = 0
        self.seen_schemas: list[list[dict[str, Any]]] = []
        self.seen_observations: list[list[ToolResult]] = []

    async def next_action_stream(
        self,
        *,
        user_text: str,
        tool_schemas: list[dict[str, Any]],
        observations: list[ToolResult],
        cancel_token: CancelToken,
    ) -> Any:
        del user_text, cancel_token
        self.seen_schemas.append([dict(s) for s in tool_schemas])
        self.seen_observations.append(list(observations))
        if self._on_step is not None:
            self._on_step(self.calls)
        events = self._steps[self.calls] if self.calls < len(self._steps) else []
        self.calls += 1
        for event in events:
            yield event


class WaitingStepModel:
    """Blocks inside the model step until the token is cancelled."""

    def __init__(self) -> None:
        self.released = False

    async def next_action_stream(
        self,
        *,
        user_text: str,
        tool_schemas: list[dict[str, Any]],
        observations: list[ToolResult],
        cancel_token: CancelToken,
    ) -> Any:
        del user_text, tool_schemas, observations
        while not cancel_token.cancelled:
            await asyncio.sleep(0.001)
        self.released = True
        yield StepEvent(kind="action", action=ModelAction.answer("zu spät"))


class CrashingStepModel:
    """The adapter itself breaks -- not the network, the protocol."""

    async def next_action_stream(self, **_kwargs: Any) -> Any:
        raise RuntimeError("kaputter Adapter")
        yield  # pragma: no cover - makes this an async generator


class ChunkProvider:
    """A tool-capable provider that replays fixed StreamChunks."""

    def __init__(self, *chunks: StreamChunk) -> None:
        self._chunks = chunks

    async def stream_chat_tools(
        self,
        messages: list[Any],
        *,
        model: str,
        temperature: float,
        tools: list[dict[str, Any]],
        num_ctx: int | None = None,
    ) -> Any:
        del messages, model, temperature, tools, num_ctx
        for chunk in self._chunks:
            yield chunk


# -- helpers ------------------------------------------------------------------


def _spec(
    name: str,
    risk: RiskLevel,
    *,
    model_callable: bool = True,
    handler: Any = None,
    parameters: dict[str, Any] | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        title=name,
        description=f"{name} description",
        risk=risk,
        parameters=parameters if parameters is not None else READ_SCHEMA,
        handler=handler or (lambda params: {"ran": name}),
        effect=f"{name} effect",
        auto_allow=True,
        requires_integration=False,
        model_callable=model_callable,
    )


def _write_spec(name: str, counter: dict[str, int]) -> ToolSpec:
    """A write tool that always asks. The card content is whatever was passed."""

    def _handler(params: dict[str, Any]) -> dict[str, Any]:
        counter["writes"] += 1
        return {"ran": name, "chars": len(params.get("text", ""))}

    return ToolSpec(
        name=name,
        title=name,
        description=f"{name} description",
        risk=RiskLevel.WRITE,
        parameters=TEXT_SCHEMA,
        handler=_handler,
        effect=f"{name} effect",
        auto_allow=False,
        requires_integration=False,
        model_callable=True,
        sensitive_parameters=("text",),
    )


def _gateway_only(tmp_path: Path, world: World | None = None) -> ToolGateway:
    world = world or World()
    db = Database(tmp_path / "kiki.db")
    executor = ToolExecutor(ToolRegistry(), ToolPolicy("balanced"), AuditLog(db))
    return ToolGateway(
        executor,
        panic_check=lambda: world.panic,
        integrations_check=lambda: world.integrations,
    )


def _runner(tmp_path: Path, adapter: Any, world: World, *specs: ToolSpec, **kwargs: Any) -> AssistantRunner:
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    db = Database(tmp_path / "kiki.db")
    executor = ToolExecutor(registry, ToolPolicy("balanced"), AuditLog(db))
    gateway = ToolGateway(
        executor,
        panic_check=lambda: world.panic,
        integrations_check=lambda: world.integrations,
    )
    kwargs.setdefault("max_steps", 4)
    kwargs.setdefault("max_tool_calls", 6)
    return AssistantRunner(
        adapter,
        gateway,
        trace_dir=tmp_path / "trace",
        **kwargs,
    )


def delta(text: str) -> StepEvent:
    return StepEvent(kind="delta", text=text)


def acts(action: ModelAction) -> StepEvent:
    return StepEvent(kind="action", action=action)


def answer(text: str) -> StepEvent:
    return acts(ModelAction.answer(text))


def calls(name: str, **arguments: Any) -> StepEvent:
    return acts(ModelAction.call(name, arguments))


async def _consume(
    runner: AssistantRunner,
    run: Any,
    on_event: Any = None,
) -> list[RunnerEvent]:
    events: list[RunnerEvent] = []
    async for event in runner.drive(run):
        events.append(event)
        if on_event is not None:
            on_event(event)
    return events


def _kinds(events: list[RunnerEvent]) -> list[str]:
    return [event.kind for event in events]


# -- the happy paths ----------------------------------------------------------


def test_a_final_answer_completes_the_run(tmp_path):
    model = ScriptedModel(
        [delta("Hallo "), delta("Welt"), answer("Hallo Welt")]
    )
    runner = _runner(tmp_path, model, World(), _spec("status_tool", RiskLevel.READ))

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, events = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED
    assert run.final_text == "Hallo Welt"
    assert run.is_terminal
    deltas = [event.text for event in events if event.kind == "delta"]
    assert "".join(deltas) == "Hallo Welt"
    assert _kinds(events)[-1] == "finished"
    assert events[-1].run is run


def test_status_events_flow_from_working_to_terminal(tmp_path):
    model = ScriptedModel([answer("fertig")])
    runner = _runner(tmp_path, model, World())

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, events = asyncio.run(scenario())
    statuses = [event.status for event in events if event.kind == "status"]
    assert statuses[0].status is RunStatus.RUNNING
    assert statuses[0].message_code == "working"
    assert statuses[-1].status is RunStatus.COMPLETED
    assert statuses[-1].terminal is True
    assert run.status is RunStatus.COMPLETED


def test_a_read_tool_runs_and_the_run_completes(tmp_path):
    counter = {"reads": 0}

    def _handler(params: dict[str, Any]) -> dict[str, Any]:
        counter["reads"] += 1
        return {"ok": True}

    model = ScriptedModel(
        [calls("status_tool")], [answer("Der Status ist ok.")]
    )
    runner = _runner(
        tmp_path,
        model,
        World(),
        _spec("status_tool", RiskLevel.READ, handler=_handler),
    )

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, events = asyncio.run(scenario())
    assert counter["reads"] == 1
    assert run.status is RunStatus.COMPLETED
    assert [e for e in _kinds(events) if e == "tool_start"] == ["tool_start"]
    tool_ends = [e for e in events if e.kind == "tool_end"]
    assert tool_ends[0].ok is True


def test_next_action_adapters_still_work(tmp_path):
    from tests.agent_fakes import FinalOnlyModel

    runner = _runner(tmp_path, FinalOnlyModel("Alles gut."), World())

    async def scenario():
        return await runner.run(USER_TEXT)

    run = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED
    assert run.final_text == "Alles gut."


# -- fail closed on model misbehaviour ----------------------------------------


def test_an_invalid_action_fails_the_run(tmp_path):
    model = ScriptedModel([StepEvent(kind="action", action=None)])
    runner = _runner(tmp_path, model, World())

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, _events = asyncio.run(scenario())
    assert run.status is RunStatus.FAILED
    assert run.error_code == "model_protocol_error"


def test_an_empty_final_answer_fails_the_run(tmp_path):
    model = ScriptedModel([answer("   ")])
    runner = _runner(tmp_path, model, World())

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, _events = asyncio.run(scenario())
    assert run.status is RunStatus.FAILED
    assert run.error_code == "model_protocol_error"


def test_more_than_one_call_in_a_stream_is_a_protocol_error(tmp_path):
    from kiki.ai.provider import ToolCall as ProviderToolCall

    provider = ChunkProvider(
        StreamChunk(
            tool_calls=(
                ProviderToolCall(id="1", name="status_tool", arguments={}),
                ProviderToolCall(id="2", name="status_tool", arguments={}),
            )
        )
    )
    adapter = ProviderStepAdapter(provider, model="m")
    counter = {"reads": 0}

    def _handler(params: dict[str, Any]) -> dict[str, Any]:
        counter["reads"] += 1
        return {"ok": True}

    runner = _runner(
        tmp_path,
        adapter,
        World(),
        _spec("status_tool", RiskLevel.READ, handler=_handler),
    )

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, _events = asyncio.run(scenario())
    assert run.status is RunStatus.FAILED
    assert run.error_code == "model_protocol_error"
    # A pile of calls is refused, not executed in some order.
    assert counter["reads"] == 0


def test_an_adapter_crash_fails_the_run(tmp_path):
    runner = _runner(tmp_path, CrashingStepModel(), World())

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, _events = asyncio.run(scenario())
    assert run.status is RunStatus.FAILED
    assert run.error_code == "model_protocol_error"


def test_two_actions_in_one_step_are_a_protocol_error(tmp_path):
    # The step contract is exactly one decision. A step that carries a call
    # *and* an answer is refused as a whole, not resolved in favour of either.
    model = ScriptedModel([calls("status_tool"), answer("oder doch fertig")])
    counter = {"reads": 0}

    def _handler(params: dict[str, Any]) -> dict[str, Any]:
        counter["reads"] += 1
        return {"ok": True}

    runner = _runner(
        tmp_path,
        model,
        World(),
        _spec("status_tool", RiskLevel.READ, handler=_handler),
    )

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, _events = asyncio.run(scenario())
    assert run.status is RunStatus.FAILED
    assert run.error_code == "model_protocol_error"
    assert counter["reads"] == 0


def test_a_provider_error_fails_the_run(tmp_path):
    class _Broken:
        async def next_action_stream(self, **_kwargs: Any) -> Any:
            raise ProviderError("provider_error")
            yield  # pragma: no cover - makes this an async generator

    runner = _runner(tmp_path, _Broken(), World())

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, _events = asyncio.run(scenario())
    assert run.status is RunStatus.FAILED
    assert run.error_code == "provider_error"


def test_a_crashing_machine_still_settles_the_run(tmp_path):
    # No expected failure reaches this path, and that is the point: whatever
    # breaks, the run settles, the consumer gets `finished`, and the runner
    # becomes free for the next run instead of hanging everyone.
    class _CrashingRunner(AssistantRunner):
        async def _drive(self, run, token, emit):  # type: ignore[override]
            raise RuntimeError("bug in der Maschine")

    runner = _CrashingRunner(
        ScriptedModel([answer("nie")]),
        _gateway_only(tmp_path),
        trace_dir=tmp_path / "trace",
    )

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, events = asyncio.run(scenario())
    assert run.is_terminal
    assert run.status is RunStatus.FAILED
    assert _kinds(events)[-1] == "finished"
    assert runner.busy is False


# -- limits --------------------------------------------------------------------


def test_the_step_limit_ends_the_run_visibly(tmp_path):
    model = ScriptedModel(*([[calls("status_tool")]] * 10))
    runner = _runner(
        tmp_path,
        model,
        World(),
        _spec("status_tool", RiskLevel.READ),
        max_steps=2,
    )

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, _events = asyncio.run(scenario())
    assert run.status is RunStatus.LIMIT_REACHED
    assert run.error_code == "step_limit"


def test_the_call_budget_ends_the_run_visibly(tmp_path):
    counter = {"reads": 0}

    def _handler(params: dict[str, Any]) -> dict[str, Any]:
        counter["reads"] += 1
        return {"ok": True}

    model = ScriptedModel(
        [calls("status_tool")], [calls("status_tool")], [answer("nie")]
    )
    runner = _runner(
        tmp_path,
        model,
        World(),
        _spec("status_tool", RiskLevel.READ, handler=_handler),
        max_tool_calls=1,
    )

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, _events = asyncio.run(scenario())
    assert run.status is RunStatus.LIMIT_REACHED
    assert run.error_code == "tool_call_limit"
    assert run.tool_calls == 1
    assert counter["reads"] == 1


def test_a_repeated_call_is_answered_without_running_twice(tmp_path):
    counter = {"reads": 0}

    def _handler(params: dict[str, Any]) -> dict[str, Any]:
        counter["reads"] += 1
        return {"ok": True}

    model = ScriptedModel(
        [calls("status_tool")],
        [calls("status_tool")],
        [answer("schon geschehen")],
    )
    runner = _runner(
        tmp_path,
        model,
        World(),
        _spec("status_tool", RiskLevel.READ, handler=_handler),
    )

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, _events = asyncio.run(scenario())
    assert counter["reads"] == 1
    assert run.status is RunStatus.COMPLETED
    # The model was told the call had already run.
    third_round = model.seen_observations[2]
    assert any(result.data == {"already_ran": True} for result in third_round)


# -- refusals feed the model, they do not kill the run -------------------------


def test_a_tool_the_policy_hides_does_not_exist(tmp_path):
    model = ScriptedModel(
        [calls("hidden_tool")], [answer("geht nicht")]
    )
    runner = _runner(
        tmp_path,
        model,
        World(),
        _spec("hidden_tool", RiskLevel.READ, model_callable=False),
    )

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, _events = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED
    second_round = model.seen_observations[1]
    assert second_round[-1].error_code == "unknown_tool"


def test_unknown_tool_is_reported_as_a_category(tmp_path):
    model = ScriptedModel([calls("no_such_tool")], [answer("okay")])
    runner = _runner(tmp_path, model, World())

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, events = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED
    second_round = model.seen_observations[1]
    assert second_round[-1].error_code == "unknown_tool"
    refusal = [e for e in events if e.kind == "tool_end"]
    assert refusal[0].text == "unknown_tool"


def test_invalid_arguments_are_refused_before_any_execution(tmp_path):
    counter = {"writes": 0}
    model = ScriptedModel(
        [calls("write_tool", wrong="x")], [answer("versucht")]
    )
    runner = _runner(tmp_path, model, World(), _write_spec("write_tool", counter))

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, _events = asyncio.run(scenario())
    assert counter["writes"] == 0
    assert run.status is RunStatus.COMPLETED
    second_round = model.seen_observations[1]
    assert second_round[-1].error_code == "invalid_arguments"


# -- the gateway is the only door ----------------------------------------------


def test_every_execution_goes_through_the_gateway():
    source = (SRC / "kiki" / "assistant" / "runner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    invoked: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            invoked.append(node.func.attr)
    # No direct handler call: the registry's handlers are the executor's job.
    assert "handler" not in invoked
    # Exactly one door: the gateway invocation in _tool_step.
    assert invoked.count("invoke") == 1


def test_the_runner_imports_no_gtk():
    code = (
        "import sys; import kiki.assistant.runner; "
        "sys.stdout.write(','.join(sorted("
        "m for m in sys.modules if m == 'gi' or m.startswith('gi.'))))"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


# -- cancellation --------------------------------------------------------------


def test_cancellation_during_model_wait_cancels(tmp_path):
    model = WaitingStepModel()
    runner = _runner(tmp_path, model, World())

    async def scenario():
        run = runner.begin(USER_TEXT)

        async def _cancel_soon() -> None:
            await asyncio.sleep(0.02)
            assert runner.cancel(run.id) is True

        task = asyncio.create_task(_cancel_soon())
        events = await _consume(runner, run)
        await task
        return run, events

    run, _events = asyncio.run(scenario())
    assert run.status is RunStatus.CANCELLED
    assert model.released
    assert runner.busy is False


def test_cancelling_unknown_ids_does_nothing(tmp_path):
    model = ScriptedModel([answer("fertig")])
    runner = _runner(tmp_path, model, World())

    async def scenario():
        run = runner.begin(USER_TEXT)
        assert runner.cancel("run-keins") is False
        return run, await _consume(runner, run)

    run, _events = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED


def test_a_second_run_is_refused_while_one_is_active(tmp_path):
    model = WaitingStepModel()
    runner = _runner(tmp_path, model, World())

    async def scenario():
        run = runner.begin(USER_TEXT)
        refusals: list[Exception] = []

        async def _try_second() -> None:
            await asyncio.sleep(0.01)
            from kiki.harness.models import RunBusyError

            try:
                runner.begin("zweite Aufgabe")
            except RunBusyError as exc:
                refusals.append(exc)
            assert runner.cancel(run.id) is True

        task = asyncio.create_task(_try_second())
        events = await _consume(runner, run)
        await task
        return run, events, refusals

    run, _events, refusals = asyncio.run(scenario())
    assert len(refusals) == 1
    assert run.status is RunStatus.CANCELLED


# -- confirmation --------------------------------------------------------------


def test_an_approval_runs_the_write_exactly_once(tmp_path):
    counter = {"writes": 0}
    model = ScriptedModel(
        [calls("write_tool", text="meine Notiz")],
        [answer("gemacht")],
    )
    runner = _runner(tmp_path, model, World(), _write_spec("write_tool", counter))
    seen: list[Any] = []

    def _on_event(event: RunnerEvent) -> None:
        if event.kind == "confirmation_requested":
            request = event.request
            assert request is not None and request.request_id != ""
            seen.append(request)
            runner.confirm(request.run_id, request.call_id, request.request_id)

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run, on_event=_on_event)

    run, events = asyncio.run(scenario())
    assert len(seen) == 1
    assert counter["writes"] == 1
    assert run.status is RunStatus.COMPLETED
    statuses = [e.status for e in events if e.kind == "status"]
    assert any(s.status is RunStatus.NEEDS_CONFIRMATION for s in statuses)
    trace = TraceRecorder(tmp_path / "trace", run.id).read()
    names = [record["event"] for record in trace]
    assert "confirmation_requested" in names
    assert "confirmation_approved" in names
    assert "write_executed" in names


def test_a_rejection_keeps_the_run_alive(tmp_path):
    counter = {"writes": 0}
    model = ScriptedModel(
        [calls("write_tool", text="meine Notiz")],
        [answer("dann eben nicht")],
    )
    runner = _runner(tmp_path, model, World(), _write_spec("write_tool", counter))

    def _on_event(event: RunnerEvent) -> None:
        if event.kind == "confirmation_requested":
            assert event.request is not None
            runner.reject(event.request.run_id, event.request.call_id)

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run, on_event=_on_event)

    run, _events = asyncio.run(scenario())
    assert counter["writes"] == 0
    assert run.status is RunStatus.COMPLETED
    second_round = model.seen_observations[1]
    assert second_round[-1].error_code == "confirmation_rejected"


def test_a_wrong_request_id_is_refused_and_the_card_stays(tmp_path):
    counter = {"writes": 0}
    model = ScriptedModel(
        [calls("write_tool", text="meine Notiz")],
        [answer("gemacht")],
    )
    runner = _runner(tmp_path, model, World(), _write_spec("write_tool", counter))
    answered: list[bool] = []

    def _on_event(event: RunnerEvent) -> None:
        if event.kind == "confirmation_requested":
            request = event.request
            assert request is not None
            with pytest.raises(ConfirmationError):
                runner.confirm(request.run_id, request.call_id, "erfundene-id")
            with pytest.raises(ConfirmationError):
                runner.confirm(request.run_id, request.call_id, "")
            answered.append(True)
            runner.confirm(request.run_id, request.call_id, request.request_id)

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run, on_event=_on_event)

    run, _events = asyncio.run(scenario())
    assert answered == [True]
    assert counter["writes"] == 1
    assert run.status is RunStatus.COMPLETED


def test_cancelling_a_pending_confirmation_wastes_it(tmp_path):
    counter = {"writes": 0}
    model = ScriptedModel(
        [calls("write_tool", text="meine Notiz")],
        [answer("nie gefragt")],
    )
    runner = _runner(tmp_path, model, World(), _write_spec("write_tool", counter))

    def _on_event(event: RunnerEvent) -> None:
        if event.kind == "confirmation_requested":
            assert event.request is not None
            assert runner.cancel(event.request.run_id) is True

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run, on_event=_on_event)

    run, _events = asyncio.run(scenario())
    assert counter["writes"] == 0
    assert run.status is RunStatus.CANCELLED
    with pytest.raises(ConfirmationError):
        runner.confirm(run.id, "call-egal", "irgendeine-id")


def test_abandon_cancels_the_pending_write(tmp_path):
    counter = {"writes": 0}
    model = ScriptedModel(
        [calls("write_tool", text="meine Notiz")],
        [answer("nie")],
    )
    runner = _runner(tmp_path, model, World(), _write_spec("write_tool", counter))

    def _on_event(event: RunnerEvent) -> None:
        if event.kind == "confirmation_requested":
            runner.abandon_confirmation()

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run, on_event=_on_event)

    run, _events = asyncio.run(scenario())
    assert counter["writes"] == 0
    assert run.status is RunStatus.CANCELLED


# -- the live policy -----------------------------------------------------------


def test_panic_mid_run_removes_the_tools(tmp_path):
    world = World()
    model = ScriptedModel(
        [calls("status_tool")],
        [answer("ohne werkzeuge")],
        # Panic flips while the first round is under way. The list was built
        # before it, so the first round keeps its tool; the next read of the
        # live policy -- the very next step -- must not offer it any more.
        on_step=lambda index: setattr(world, "panic", True) if index == 0 else None,
    )
    runner = _runner(tmp_path, model, world, _spec("status_tool", RiskLevel.READ))

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run)

    run, _events = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED
    assert model.seen_schemas[0] != []
    assert model.seen_schemas[1] == []


def test_panic_stops_an_approved_not_yet_run_side_effect(tmp_path):
    counter = {"writes": 0}
    world = World()
    model = ScriptedModel(
        [calls("write_tool", text="meine Notiz")],
        [answer("kann nicht mehr")],
    )
    runner = _runner(tmp_path, model, world, _write_spec("write_tool", counter))

    def _on_event(event: RunnerEvent) -> None:
        if event.kind == "confirmation_requested":
            request = event.request
            assert request is not None
            # Approved while the card was open, but the world has moved: the
            # recheck right before the side effect must stop it.
            world.panic = True
            runner.confirm(request.run_id, request.call_id, request.request_id)

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run, on_event=_on_event)

    run, _events = asyncio.run(scenario())
    assert counter["writes"] == 0
    assert run.status is RunStatus.COMPLETED
    second_round = model.seen_observations[1]
    assert second_round[-1].error_code == "tool_unavailable"


# -- the trace -----------------------------------------------------------------


def test_the_trace_holds_no_content(tmp_path):
    counter = {"writes": 0}
    model = ScriptedModel(
        # A plain content argument: the trace sanitizer would mask a path or a
        # token anyway, so this is the value whose absence actually proves the
        # runner writes shapes, not values.
        [calls("write_tool", text="meine Notiz")],
        [answer("gemacht")],
    )
    runner = _runner(tmp_path, model, World(), _write_spec("write_tool", counter))

    def _on_event(event: RunnerEvent) -> None:
        if event.kind == "confirmation_requested":
            request = event.request
            assert request is not None
            runner.confirm(request.run_id, request.call_id, request.request_id)

    async def scenario():
        run = runner.begin(USER_TEXT)
        return run, await _consume(runner, run, on_event=_on_event)

    run, _events = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED
    records = TraceRecorder(tmp_path / "trace", run.id).read()
    blob = "".join(str(record) for record in records)
    assert "meine Notiz" not in blob
    assert "secret.txt" not in blob
    assert "sk-test-secret" not in blob
    started = [record for record in records if record["event"] == "run_started"][0]
    assert started["user_text_length"] == len(USER_TEXT)
    requested = [
        record
        for record in records
        if record["event"] == "tool_requested" and record.get("accepted")
    ][0]
    assert requested["arguments"] == {"text": len("meine Notiz")}


# -- the provider step adapter -------------------------------------------------


def test_the_provider_adapter_streams_deltas_and_collapses_one_call(tmp_path):
    from kiki.ai.provider import ToolCall as ProviderToolCall

    provider = ChunkProvider(
        StreamChunk(text="Ich sehe nach."),
        StreamChunk(
            tool_calls=(ProviderToolCall(id="1", name="status_tool", arguments={}),)
        ),
        StreamChunk(text=""),
    )
    adapter = ProviderStepAdapter(provider, model="m")

    async def scenario():
        events: list[StepEvent] = []
        async for event in adapter.next_action_stream(
            user_text="x",
            tool_schemas=[],
            observations=[],
            cancel_token=CancelToken(),
        ):
            events.append(event)
        return events

    events = asyncio.run(scenario())
    assert [e.text for e in events if e.kind == "delta"] == ["Ich sehe nach."]
    action = events[-1].action
    assert action is not None
    assert action.kind is ActionKind.TOOL_CALL
    assert action.tool_call is not None
    assert action.tool_call.name == "status_tool"


def test_the_provider_adapter_stops_streaming_when_cancelled(tmp_path):
    provider = ChunkProvider(StreamChunk(text="niemand liest das"))
    adapter = ProviderStepAdapter(provider, model="m")
    token = CancelToken()
    token.cancel()

    async def scenario():
        events: list[StepEvent] = []
        async for event in adapter.next_action_stream(
            user_text="x",
            tool_schemas=[],
            observations=[],
            cancel_token=token,
        ):
            events.append(event)
        return events

    events = asyncio.run(scenario())
    assert [e.kind for e in events] == ["action"]
    action = events[0].action
    assert action is not None and action.kind is ActionKind.FINAL


def test_the_provider_adapter_passes_observations_as_categories(tmp_path):
    seen_messages: list[list[Any]] = []

    class _Recording(ChunkProvider):
        async def stream_chat_tools(self, messages, **kwargs: Any) -> Any:
            seen_messages.append(list(messages))
            async for chunk in super().stream_chat_tools(messages, **kwargs):
                yield chunk

    provider = _Recording(StreamChunk(text="ok"))
    adapter = ProviderStepAdapter(provider, model="m")
    observations = [
        ToolResult(call_id="c1", name="t", ok=True, data={"a": 1}),
        ToolResult(call_id="c2", name="t", ok=False, error_code="tool_failed"),
    ]

    async def scenario():
        async for _event in adapter.next_action_stream(
            user_text="x",
            tool_schemas=[],
            observations=observations,
            cancel_token=CancelToken(),
        ):
            pass

    asyncio.run(scenario())
    tool_messages = [m for m in seen_messages[0] if getattr(m, "role", "") == "tool"]
    assert len(tool_messages) == 2
    assert '{"a": 1}' in tool_messages[0].content
    assert "tool_failed" in tool_messages[1].content
