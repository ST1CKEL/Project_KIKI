"""The /agent path on the unified stack: legacy wiring, current rules.

`/agent` is the developer path since the chat runs on the same runner. What
these tests pin down is the wiring the application builds: a step adapter
with the composed prompt and the configured window, one runner with the
configured limits, the run service on top -- and the semantics /agent always
had: preamble before a tool call is never an answer, the card settles only
through the request id, a cancel is silent.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from kiki.ai.provider import StreamChunk
from kiki.ai.provider import ToolCall as ProviderToolCall
from kiki.assistant import (
    AssistantRunner,
    ProviderStepAdapter,
    RunCallbacks,
    RunService,
)
from kiki.harness.confirmation import ConfirmationRequest
from kiki.harness.models import HarnessStatusEvent, RunBusyError, RunStatus
from kiki.storage.database import Database
from kiki.tools.audit import AuditLog
from kiki.tools.executor import ToolExecutor
from kiki.tools.gateway import ToolGateway
from kiki.tools.policy import RiskLevel, ToolPolicy
from kiki.tools.registry import ToolRegistry, ToolSpec

SYSTEM_PROMPT = "Du bist KIKI. Kernregeln folgen."
USER_TEXT = "erstelle eine notiz mit ghp_testtoken Inhalt"

READ_SCHEMA = {"type": "object", "properties": {}, "required": []}
TEXT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


class AgentPathProvider:
    """Records every request; plays scripted turns."""

    id = "agent-path"

    def __init__(self, *turns: list[StreamChunk]) -> None:
        self._turns = [list(turn) for turn in turns]
        self.calls = 0
        self.seen: list[dict[str, Any]] = []

    async def stream_chat(self, messages, *, model, temperature=0.7, num_ctx=None):
        yield "ohne werkzeuge"

    async def stream_chat_tools(
        self, messages, *, model, temperature=0.7, tools, num_ctx=None
    ):
        self.seen.append(
            {
                "messages": list(messages),
                "tools": list(tools),
                "num_ctx": num_ctx,
                "model": model,
            }
        )
        turn = self._turns[self.calls] if self.calls < len(self._turns) else []
        self.calls += 1
        for chunk in turn:
            yield chunk


class BlockingProvider:
    """Blocks inside the model step until cancelled."""

    id = "blocking"

    async def stream_chat(self, messages, *, model, temperature=0.7, num_ctx=None):
        yield "x"

    async def stream_chat_tools(
        self, messages, *, model, temperature=0.7, tools, num_ctx=None
    ):
        del messages, model, temperature, tools, num_ctx
        while True:
            await asyncio.sleep(0.005)
            yield StreamChunk(text="")


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

    @property
    def codes(self) -> list[str]:
        return [event.message_code for event in self.status]


def _spec(name: str, risk: RiskLevel, *, handler: Any = None, parameters=None) -> ToolSpec:
    return ToolSpec(
        name=name,
        title=name,
        description=f"{name} description",
        risk=risk,
        parameters=parameters if parameters is not None else READ_SCHEMA,
        handler=handler or (lambda _p: {"ran": name}),
        effect=f"{name} effect",
        auto_allow=True,
        requires_integration=False,
        model_callable=True,
    )


def _write_spec(counter: dict) -> ToolSpec:
    def _handler(params: dict[str, Any]) -> dict[str, Any]:
        counter["writes"] += 1
        return {"wrote": True}

    return ToolSpec(
        name="write_tool",
        title="Schreiben",
        description="Schreibt etwas.",
        risk=RiskLevel.WRITE,
        parameters=TEXT_SCHEMA,
        handler=_handler,
        effect="Schreibt den Text.",
        auto_allow=False,
        requires_integration=False,
        model_callable=True,
        sensitive_parameters=("text",),
    )


def _stack(
    tmp_path: Path,
    provider: Any,
    recorder: Recorder,
    *specs: ToolSpec,
    max_steps: int = 4,
    max_tool_calls: int = 6,
) -> RunService:
    """Mirrors what the application wires for /agent, without GTK."""
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    executor = ToolExecutor(registry, ToolPolicy("balanced"), AuditLog(Database(tmp_path / "kiki.db")))
    gateway = ToolGateway(
        executor,
        panic_check=lambda: False,
        integrations_check=lambda: True,
    )
    adapter = ProviderStepAdapter(
        provider,
        model="test-model",
        system_prompt=SYSTEM_PROMPT,
        num_ctx=1234,
    )
    runner = AssistantRunner(
        adapter,
        gateway,
        profile="observe",
        trace_dir=tmp_path / "traces",
        max_steps=max_steps,
        max_tool_calls=max_tool_calls,
    )
    return RunService(runner, recorder.callbacks())


def _text(text: str) -> StreamChunk:
    return StreamChunk(text=text)


def _call(call_id: str, name: str, arguments: dict | None = None) -> StreamChunk:
    return StreamChunk(tool_calls=(ProviderToolCall(id=call_id, name=name, arguments=arguments or {}),))


def _ask(service: RunService, text: str = USER_TEXT) -> Any:
    return asyncio.wait_for(service.ask(text), timeout=10)


# --- the wiring the application builds ---------------------------------------


def test_the_composed_prompt_and_window_reach_the_provider(tmp_path):
    recorder = Recorder()
    provider = AgentPathProvider([_text("Verstanden.")])
    service = _stack(tmp_path, provider, recorder, _spec("status_tool", RiskLevel.READ))

    async def scenario():
        return await _ask(service)

    run = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED
    request = provider.seen[0]
    assert request["messages"][0].role == "system"
    assert request["messages"][0].content == SYSTEM_PROMPT
    assert request["messages"][1].role == "user"
    assert request["messages"][1].content == USER_TEXT
    assert request["num_ctx"] == 1234
    assert request["model"] == "test-model"
    assert [s["name"] for s in (t["function"] for t in request["tools"])] == ["status_tool"]


def test_a_preamble_before_a_tool_call_is_never_the_answer(tmp_path):
    recorder = Recorder()
    provider = AgentPathProvider(
        [_text("Ich sehe kurz nach."), _call("c1", "status_tool")],
        [_text("Ergebnis da.")],
    )
    service = _stack(tmp_path, provider, recorder, _spec("status_tool", RiskLevel.READ))

    async def scenario():
        return await _ask(service)

    run = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED
    assert run.final_text == "Ergebnis da."
    # The preamble is not delivered as an answer and not spoken.
    assert recorder.answers == ["Ergebnis da."]
    assert recorder.spoken == ["Ergebnis da."]
    assert recorder.codes == ["working", "tool_running", "completed"]


def test_the_tool_exchange_reaches_the_provider_flat(tmp_path):
    recorder = Recorder()
    provider = AgentPathProvider(
        [_call("c1", "status_tool")],
        [_text("Fertig.")],
    )
    service = _stack(tmp_path, provider, recorder, _spec("status_tool", RiskLevel.READ))

    async def scenario():
        return await _ask(service)

    asyncio.run(scenario())
    # The /agent adapter stays flat, like the harness adapter before it:
    # [system, user, tool results]. Ollama -- the documented /agent provider --
    # accepts that shape. (The chat adapter accumulates the full protocol
    # exchange; /agent inherits that only if it ever needs OpenAI-compat.)
    second = provider.seen[1]["messages"]
    assert [m.role for m in second] == ["system", "user", "tool"]
    tool_msg = second[-1]
    assert tool_msg.tool_name == "status_tool"
    assert json.loads(tool_msg.content)["ran"] == "status_tool"


def test_the_configured_limits_govern_the_run(tmp_path):
    recorder = Recorder()
    provider = AgentPathProvider(
        *([_call(f"c{i}", "status_tool")] for i in range(20))
    )
    service = _stack(
        tmp_path,
        provider,
        recorder,
        _spec("status_tool", RiskLevel.READ),
        max_steps=2,
        max_tool_calls=6,
    )

    async def scenario():
        return await _ask(service)

    run = asyncio.run(scenario())
    assert run.status is RunStatus.LIMIT_REACHED
    assert run.error_code == "step_limit"
    # Two steps means two model requests -- not the runner's own default.
    assert provider.calls == 2
    assert recorder.codes[-1] == "limit_reached"


# --- confirmation through the card -------------------------------------------


def test_an_approved_card_writes_exactly_once(tmp_path):
    counter = {"writes": 0}
    recorder = Recorder()
    provider = AgentPathProvider(
        [_call("c1", "write_tool", {"text": "notiz"})],
        [_text("Angelegt.")],
    )
    service = _stack(tmp_path, provider, recorder, _write_spec(counter))

    async def scenario():
        task = asyncio.create_task(_ask(service))
        for _ in range(500):
            if service.pending_confirmation is not None:
                break
            await asyncio.sleep(0.01)
        request = service.pending_confirmation
        assert request is not None
        assert service.confirm(request.run_id, request.call_id, request.request_id) is True
        return await task

    run = asyncio.run(scenario())
    assert counter["writes"] == 1
    assert run.status is RunStatus.COMPLETED
    assert recorder.confirmations and recorder.confirmations[0].tool_name == "write_tool"
    assert recorder.answers == ["Angelegt."]


def test_a_rejected_card_runs_nothing_and_the_model_recovers(tmp_path):
    counter = {"writes": 0}
    recorder = Recorder()
    provider = AgentPathProvider(
        [_call("c1", "write_tool", {"text": "notiz"})],
        [_text("Dann nicht.")],
    )
    service = _stack(tmp_path, provider, recorder, _write_spec(counter))

    async def scenario():
        task = asyncio.create_task(_ask(service))
        for _ in range(500):
            if service.pending_confirmation is not None:
                break
            await asyncio.sleep(0.01)
        request = service.pending_confirmation
        assert request is not None
        assert service.reject(request.run_id, request.call_id) is True
        return await task

    run = asyncio.run(scenario())
    assert counter["writes"] == 0
    assert run.status is RunStatus.COMPLETED
    tool_msg = provider.seen[1]["messages"][-1]
    assert json.loads(tool_msg.content) == {"error": "confirmation_rejected"}
    assert recorder.answers == ["Dann nicht."]


# --- run shell ----------------------------------------------------------------


def test_a_second_run_is_refused_and_the_first_survives(tmp_path):
    recorder = Recorder()
    service = _stack(tmp_path, BlockingProvider(), recorder)

    async def scenario():
        first = asyncio.create_task(_ask(service))
        await asyncio.sleep(0.02)
        with pytest.raises(RunBusyError):
            await _ask(service)
        service.cancel()
        return await first

    run = asyncio.run(scenario())
    assert run.status is RunStatus.CANCELLED


def test_a_cancel_is_silent(tmp_path):
    recorder = Recorder()
    service = _stack(tmp_path, BlockingProvider(), recorder)

    async def scenario():
        task = asyncio.create_task(_ask(service))
        await asyncio.sleep(0.02)
        assert service.cancel() is True
        return await task

    run = asyncio.run(scenario())
    assert run.status is RunStatus.CANCELLED
    assert recorder.answers == []
    assert recorder.spoken == []


# --- the trace ----------------------------------------------------------------


def test_the_run_leaves_a_content_free_trace(tmp_path):
    recorder = Recorder()
    provider = AgentPathProvider(
        [_call("c1", "status_tool")],
        [_text("Fertig.")],
    )
    service = _stack(tmp_path, provider, recorder, _spec("status_tool", RiskLevel.READ))

    async def scenario():
        return await _ask(service)

    run = asyncio.run(scenario())
    trace = tmp_path / "traces" / f"{run.id}.jsonl"
    assert trace.is_file()
    blob = trace.read_text(encoding="utf-8")
    assert USER_TEXT not in blob
    assert "ghp_testtoken" not in blob
    assert "status_tool" in blob
