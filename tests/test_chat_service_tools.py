"""ChatService wiring: settings gate the loop, events reach the bus, tools are recorded."""

from __future__ import annotations

import asyncio

from kiki.ai.chat_service import ChatService
from kiki.ai.provider import StreamChunk, ToolCall
from kiki.runtime.event_bus import EventBus
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec


class FakeProvider:
    """Answers with a tool call first, then with text."""

    id = "fake"

    def __init__(self) -> None:
        self.tool_turns = 0
        self.plain_turns = 0
        self.tools_seen: list[list[dict]] = []

    async def stream_chat(self, messages, *, model, temperature=0.7, num_ctx=None):
        self.plain_turns += 1
        yield "Ich kann nichts nachsehen."

    async def stream_chat_tools(self, messages, *, model, temperature=0.7, tools, num_ctx=None):
        self.tool_turns += 1
        self.tools_seen.append(list(tools))
        if self.tool_turns == 1:
            yield StreamChunk(tool_calls=(ToolCall(id="c1", name="status_disk", arguments={}),))
        else:
            yield StreamChunk(text="Noch 42 GB frei.")


def _register_disk(registry) -> None:
    registry.register(
        ToolSpec(
            name="status_disk",
            title="Speicher",
            description="Liest freien Speicher.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            risk=RiskLevel.READ,
            handler=lambda _p: {"free_gb": 42},
            effect="Liest freien Speicher.",
            auto_allow=True,
            requires_integration=True,
            model_callable=True,
        )
    )


def _service(settings, chats, secrets, tools_env) -> tuple[ChatService, FakeProvider, EventBus]:
    registry, executor = tools_env
    _register_disk(registry)
    bus = EventBus()
    service = ChatService(settings, chats, secrets, bus, executor)
    provider = FakeProvider()
    service._provider = provider  # bypass the keyring/HTTP factory
    return service, provider, bus


def _send(service: ChatService, conversation_id: str) -> list:
    async def _go():
        return [event async for event in service.send(conversation_id, "wie voll ist die platte?")]

    return asyncio.run(_go())


def test_loop_runs_and_records_the_tool(settings, chats, secrets, tools_env) -> None:
    settings.tools.model_tool_use = True
    service, provider, bus = _service(settings, chats, secrets, tools_env)
    seen: list[str] = []
    bus.subscribe("chat.stream.tool_end", lambda **p: seen.append(str(p.get("tool"))))

    conv = service.ensure_conversation(None)
    events = _send(service, conv.id)

    assert provider.tool_turns == 2
    assert provider.plain_turns == 0
    assert [e.kind for e in events] == ["tool_start", "tool_end", "delta", "done"]
    assert events[-1].text == "Noch 42 GB frei."
    assert seen == ["status_disk"]

    stored = chats.history(conv.id)
    assert stored[-1].role == "assistant"
    assert "Noch 42 GB frei." in stored[-1].content
    # The transcript must say what produced the number.
    assert "status_disk" in stored[-1].content


def test_disabled_setting_keeps_the_plain_stream(settings, chats, secrets, tools_env) -> None:
    settings.tools.model_tool_use = False
    service, provider, _bus = _service(settings, chats, secrets, tools_env)

    conv = service.ensure_conversation(None)
    events = _send(service, conv.id)

    assert provider.tool_turns == 0
    assert provider.plain_turns == 1
    assert [e.kind for e in events] == ["delta", "done"]
    assert "status_disk" not in chats.history(conv.id)[-1].content


def test_panic_disables_the_loop_entirely(settings, chats, secrets, tools_env) -> None:
    settings.tools.model_tool_use = True
    settings.app.privacy_panic = True
    service, provider, _bus = _service(settings, chats, secrets, tools_env)

    conv = service.ensure_conversation(None)
    _send(service, conv.id)

    assert service.tools_active() is False
    assert provider.tool_turns == 0
    assert provider.plain_turns == 1


def test_provider_without_tool_support_falls_back(settings, chats, secrets, tools_env) -> None:
    class TextOnly:
        id = "text-only"

        def __init__(self) -> None:
            self.calls = 0

        async def stream_chat(self, messages, *, model, temperature=0.7, num_ctx=None):
            self.calls += 1
            yield "Nur Text."

    settings.tools.model_tool_use = True
    registry, executor = tools_env
    _register_disk(registry)
    service = ChatService(settings, chats, secrets, EventBus(), executor)
    provider = TextOnly()
    service._provider = provider

    conv = service.ensure_conversation(None)
    events = _send(service, conv.id)

    assert service.tools_active() is False
    assert provider.calls == 1
    assert [e.kind for e in events] == ["delta", "done"]


def test_autonomy_setting_reaches_the_policy(settings, chats, secrets, tools_env) -> None:
    _registry, executor = tools_env
    service = ChatService(settings, chats, secrets, EventBus(), executor)

    settings.tools.autonomy = "strict"
    service.update_settings(settings)
    assert executor.policy.autonomy.value == "strict"

    settings.tools.autonomy = "balanced"
    service.update_settings(settings)
    assert executor.policy.autonomy.value == "balanced"


def test_declared_tools_shrink_when_integrations_are_off(
    settings, chats, secrets, tools_env
) -> None:
    settings.tools.model_tool_use = True
    settings.integrations.enabled = False
    service, provider, _bus = _service(settings, chats, secrets, tools_env)

    conv = service.ensure_conversation(None)
    _send(service, conv.id)

    # status_disk requires integrations, so nothing is left to offer and the
    # service falls back to a plain answer instead of a crippled loop.
    assert provider.tool_turns == 0
    assert provider.plain_turns == 1
