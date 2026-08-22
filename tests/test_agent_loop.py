"""The agent loop must stay inside the policy it was built on."""

from __future__ import annotations

import asyncio

import pytest

from kiki.ai.agent_loop import AgentLoop
from kiki.ai.provider import ChatMessage, ProviderError, StreamChunk, ToolCall
from kiki.tools.exposure import declarations, exposed_specs
from kiki.tools.policy import AutonomyLevel, Origin, RiskLevel, ToolPolicy
from kiki.tools.registry import ToolSpec


class ScriptedProvider:
    """Replays a fixed list of turns and records what it was asked."""

    id = "scripted"

    def __init__(self, turns: list[list[StreamChunk]]) -> None:
        self._turns = turns
        self.seen: list[list[ChatMessage]] = []
        self.tools_offered: list[list[dict]] = []

    async def stream_chat_tools(self, messages, *, model, temperature=0.7, tools, num_ctx=None):
        self.seen.append(list(messages))
        self.tools_offered.append(list(tools))
        index = min(len(self.seen) - 1, len(self._turns) - 1)
        for chunk in self._turns[index]:
            yield chunk


def _call(name: str, **arguments) -> ToolCall:
    return ToolCall(id=f"c-{name}", name=name, arguments=dict(arguments))


def _spec(**kwargs) -> ToolSpec:
    values = dict(
        name="status_disk",
        title="Speicher",
        description="Liest freien Speicher.",
        risk=RiskLevel.READ,
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda params: {"free_gb": 42},
        effect="Liest freien Speicher.",
        auto_allow=True,
        requires_integration=True,
        model_callable=True,
    )
    values.update(kwargs)
    return ToolSpec(**values)


def _drain(loop: AgentLoop, *, confirm=None, panic=False, integrations=True, profile="observe"):
    async def _go():
        events = []
        async for event in loop.run(
            [ChatMessage(role="user", content="wie voll ist die platte?")],
            model="test",
            temperature=0.0,
            tools=[{"type": "function", "function": {"name": "status_disk"}}],
            panic_check=lambda: panic,
            integrations_check=lambda: integrations,
            profile=profile,
            confirm=confirm,
        ):
            events.append(event)
        return events

    return asyncio.run(_go())


def test_tool_result_flows_back_into_the_answer(tools_env) -> None:
    registry, executor = tools_env
    registry.register(_spec())
    provider = ScriptedProvider(
        [
            [StreamChunk(tool_calls=(_call("status_disk"),))],
            [StreamChunk(text="Noch 42 GB frei.")],
        ]
    )
    events = _drain(AgentLoop(provider, executor))

    assert [e.kind for e in events] == ["tool_start", "tool_end", "delta", "done"]
    assert events[1].ok is True
    assert events[-1].text == "Noch 42 GB frei."
    # The second turn must actually carry the tool result back to the model.
    follow_up = provider.seen[1]
    assert follow_up[-1].role == "tool"
    assert "42" in follow_up[-1].content
    assert follow_up[-2].role == "assistant"
    assert follow_up[-2].tool_calls[0].name == "status_disk"


def test_hard_denied_tool_never_runs(tools_env) -> None:
    registry, executor = tools_env
    ran = {"n": 0}

    def handler(_params):
        ran["n"] += 1
        return {"done": True}

    registry.register(_spec(name="run_shell", handler=handler))
    provider = ScriptedProvider(
        [
            [StreamChunk(tool_calls=(_call("run_shell"),))],
            [StreamChunk(text="Das darf ich nicht.")],
        ]
    )
    events = _drain(AgentLoop(provider, executor))

    assert ran["n"] == 0
    end = [e for e in events if e.kind == "tool_end"][0]
    assert end.ok is False
    assert "verboten" in end.text


def test_tool_without_model_callable_is_denied(tools_env) -> None:
    registry, executor = tools_env
    ran = {"n": 0}

    def handler(_params):
        ran["n"] += 1
        return {"ok": True}

    registry.register(_spec(name="status_secret", model_callable=False, handler=handler))
    provider = ScriptedProvider(
        [
            [StreamChunk(tool_calls=(_call("status_secret"),))],
            [StreamChunk(text="Geht nicht.")],
        ]
    )
    events = _drain(AgentLoop(provider, executor))

    assert ran["n"] == 0
    end = [e for e in events if e.kind == "tool_end"][0]
    assert end.ok is False
    assert "nicht für Modellaufrufe" in end.text


def test_write_tool_from_model_reaches_the_approval_card(tools_env) -> None:
    registry, executor = tools_env
    ran = {"n": 0}
    asked: list[str] = []

    def handler(_params):
        ran["n"] += 1
        return {"ok": True}

    registry.register(
        _spec(name="write_note", risk=RiskLevel.WRITE, auto_allow=True, handler=handler)
    )

    async def deny(preview):
        asked.append(preview.tool)
        return False

    provider = ScriptedProvider(
        [
            [StreamChunk(tool_calls=(_call("write_note"),))],
            [StreamChunk(text="Abgebrochen.")],
        ]
    )
    events = _drain(AgentLoop(provider, executor), confirm=deny)

    assert asked == ["write_note"]
    assert ran["n"] == 0
    assert [e for e in events if e.kind == "tool_end"][0].ok is False


def test_control_tool_is_unattended_only_when_balanced(db) -> None:
    from kiki.tools.audit import AuditLog
    from kiki.tools.executor import ToolExecutor
    from kiki.tools.registry import ToolRegistry

    for level, expect_dialog in (
        (AutonomyLevel.STRICT, True),
        (AutonomyLevel.BALANCED, False),
    ):
        registry = ToolRegistry()
        executor = ToolExecutor(registry, ToolPolicy(level.value), AuditLog(db))
        registry.register(_spec(name="agent_stop", risk=RiskLevel.CONTROL))
        asked: list[str] = []

        async def confirm(preview, _asked=asked):
            _asked.append(preview.tool)
            return True

        provider = ScriptedProvider(
            [
                [StreamChunk(tool_calls=(_call("agent_stop"),))],
                [StreamChunk(text="Erledigt.")],
            ]
        )
        _drain(AgentLoop(provider, executor), confirm=confirm)
        assert bool(asked) is expect_dialog, f"{level} should ask: {expect_dialog}"


def test_panic_mid_turn_stops_further_tools(tools_env) -> None:
    registry, executor = tools_env
    ran = {"n": 0}

    def handler(_params):
        ran["n"] += 1
        return {"free_gb": 42}

    registry.register(_spec(handler=handler))
    provider = ScriptedProvider(
        [
            [StreamChunk(tool_calls=(_call("status_disk"),))],
            [StreamChunk(text="Geht gerade nicht.")],
        ]
    )
    events = _drain(AgentLoop(provider, executor), panic=True)

    assert ran["n"] == 0
    assert [e for e in events if e.kind == "tool_end"][0].ok is False


def test_repeated_identical_call_runs_once(tools_env) -> None:
    registry, executor = tools_env
    ran = {"n": 0}

    def handler(_params):
        ran["n"] += 1
        return {"free_gb": 42}

    registry.register(_spec(handler=handler))
    provider = ScriptedProvider(
        [
            [StreamChunk(tool_calls=(_call("status_disk"),))],
            [StreamChunk(tool_calls=(_call("status_disk"),))],
            [StreamChunk(text="Noch 42 GB frei.")],
        ]
    )
    events = _drain(AgentLoop(provider, executor))

    assert ran["n"] == 1
    assert events[-1].kind == "done"


def test_step_limit_reports_instead_of_claiming_success(tools_env) -> None:
    registry, executor = tools_env
    registry.register(_spec())
    provider = ScriptedProvider([[StreamChunk(tool_calls=(_call("status_disk", probe=1),))]])
    loop = AgentLoop(provider, executor, max_steps=2)

    async def _go():
        events = []
        async for event in loop.run(
            [ChatMessage(role="user", content="x")],
            model="test",
            temperature=0.0,
            tools=[],
            panic_check=lambda: False,
            integrations_check=lambda: True,
        ):
            events.append(event)
        return events

    events = asyncio.run(_go())
    assert events[-1].kind == "error"
    assert "Schrittlimit" in events[-1].text
    assert not any(e.kind == "done" for e in events)


def test_call_budget_stops_execution(tools_env) -> None:
    registry, executor = tools_env
    ran = {"n": 0}

    def handler(params):
        ran["n"] += 1
        return {"free_gb": params.get("probe", 0)}

    registry.register(
        _spec(
            handler=handler,
            parameters={
                "type": "object",
                "properties": {"probe": {"type": "integer"}},
                "additionalProperties": False,
            },
        )
    )
    provider = ScriptedProvider(
        [
            [
                StreamChunk(
                    tool_calls=(
                        _call("status_disk", probe=1),
                        _call("status_disk", probe=2),
                        _call("status_disk", probe=3),
                    )
                )
            ],
            [StreamChunk(text="Fertig.")],
        ]
    )
    _drain(AgentLoop(provider, executor, max_tool_calls=2))
    assert ran["n"] == 2


def test_malformed_arguments_are_reported_not_executed(tools_env) -> None:
    registry, executor = tools_env
    ran = {"n": 0}

    def handler(_params):
        ran["n"] += 1
        return {"ok": True}

    registry.register(_spec(handler=handler))
    broken = ToolCall(id="c1", name="status_disk", arguments={}, parse_error="kaputtes JSON")
    provider = ScriptedProvider(
        [
            [StreamChunk(tool_calls=(broken,))],
            [StreamChunk(text="Konnte ich nicht lesen.")],
        ]
    )
    events = _drain(AgentLoop(provider, executor))

    assert ran["n"] == 0
    end = [e for e in events if e.kind == "tool_end"][0]
    assert end.ok is False
    assert "kaputtes JSON" in end.text


def test_provider_error_becomes_an_error_event(tools_env) -> None:
    _registry, executor = tools_env

    class Failing:
        async def stream_chat_tools(self, messages, *, model, temperature=0.7, tools, num_ctx=None):
            raise ProviderError("Ollama ist nicht erreichbar.")
            yield  # pragma: no cover - generator marker

    events = _drain(AgentLoop(Failing(), executor))
    assert len(events) == 1
    assert events[0].kind == "error"
    assert events[0].ok is False
    assert "nicht erreichbar" in events[0].text


def test_exposure_follows_panic_and_model_callable(tools_env) -> None:
    registry, executor = tools_env
    registry.register(_spec(name="status_disk"))
    registry.register(_spec(name="status_hidden", model_callable=False))
    registry.register(_spec(name="status_panic_ok", allowed_in_panic=True))

    normal = {s.name for s in exposed_specs(registry, executor.policy, panic=False, integrations_enabled=True)}
    assert normal == {"status_disk", "status_panic_ok"}

    panicked = {s.name for s in exposed_specs(registry, executor.policy, panic=True, integrations_enabled=True)}
    assert panicked == {"status_panic_ok"}

    schema = declarations(registry, executor.policy, panic=False, integrations_enabled=True)
    assert schema[0]["function"]["name"] == "status_disk"
    assert schema[0]["function"]["parameters"]["additionalProperties"] is False


def test_user_origin_behaviour_is_unchanged(tools_env) -> None:
    """A tool the model may not touch is still reachable from a user click."""
    registry, executor = tools_env
    registry.register(_spec(name="status_hidden", model_callable=False))

    result = asyncio.run(
        executor.run("status_hidden", {}, panic=False, integrations_enabled=True)
    )
    assert result.ok is True
    assert result.data == {"free_gb": 42}

    denied = asyncio.run(
        executor.run(
            "status_hidden", {}, panic=False, integrations_enabled=True, origin=Origin.MODEL
        )
    )
    assert denied.ok is False


@pytest.mark.parametrize("bad", ["", "operator", "yolo", None])
def test_unreadable_autonomy_falls_back_to_strict(bad) -> None:
    assert ToolPolicy(bad).autonomy is AutonomyLevel.STRICT
