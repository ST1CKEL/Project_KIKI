"""Normal chat on the unified runner: same door, same rules, one engine.

These tests prove the promises the chat path gains by moving onto
`AssistantRunner`: the conversation keeps the protocol shape providers
expect, refusals arrive as categories, a limit ends the turn visibly, the
provider's own error sentence survives (it is user-facing data the plain
path has always shown), and every turn leaves a content-free trace.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from kiki.ai.chat_service import ChatService
from kiki.ai.provider import ProviderError, StreamChunk, ToolCall
from kiki.assistant.run_service import failure_text
from kiki.harness.confirmation import ConfirmationRequest
from kiki.runtime.event_bus import EventBus
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec

USER_TEXT = "wie voll ist die platte? sk-test-secret bleibt hier außen vor."
SECRET_NOTE = "meine Notiz mit ghp_testtoken"


class ScriptedProvider:
    """Records every message list it is handed; plays scripted turns."""

    id = "scripted"

    def __init__(self, *turns: list[StreamChunk]) -> None:
        self._turns = [list(turn) for turn in turns]
        self.calls = 0
        self.seen_messages: list[list[Any]] = []
        self.seen_tools: list[list[dict]] = []

    async def stream_chat(self, messages, *, model, temperature=0.7, num_ctx=None):
        yield "Keine Werkzeuge."

    async def stream_chat_tools(self, messages, *, model, temperature=0.7, tools, num_ctx=None):
        self.seen_messages.append(list(messages))
        self.seen_tools.append(list(tools))
        turn = self._turns[self.calls] if self.calls < len(self._turns) else []
        self.calls += 1
        for chunk in turn:
            yield chunk


def _chunk_text(text: str) -> StreamChunk:
    return StreamChunk(text=text)


def _chunk_call(call_id: str, name: str, arguments: dict | None = None) -> StreamChunk:
    return StreamChunk(tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments or {}),))


def _read_spec(name: str = "status_disk", *, integration: bool = True) -> ToolSpec:
    return ToolSpec(
        name=name,
        title="Speicher",
        description="Liest freien Speicher.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        risk=RiskLevel.READ,
        handler=lambda _p: {"free_gb": 42},
        effect="Liest freien Speicher.",
        auto_allow=True,
        requires_integration=integration,
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
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        risk=RiskLevel.WRITE,
        handler=_handler,
        effect="Schreibt den Text.",
        auto_allow=False,
        requires_integration=False,
        model_callable=True,
        sensitive_parameters=("text",),
    )


def _service(
    settings,
    chats,
    secrets,
    tools_env,
    provider,
    *,
    confirm=None,
    trace_dir: Path | None = None,
) -> ChatService:
    registry, executor = tools_env
    service = ChatService(
        settings,
        chats,
        secrets,
        EventBus(),
        executor,
        confirm=confirm,
        trace_dir=trace_dir,
    )
    service._provider = provider  # noqa: SLF001 - bypass the provider factory
    return service


def _send(service: ChatService, conversation_id: str, text: str = USER_TEXT) -> list[Any]:
    async def _go():
        return [event async for event in service.send(conversation_id, text)]

    return asyncio.run(asyncio.wait_for(_go(), timeout=10))


def _turn(provider: ScriptedProvider) -> list[Any]:
    return provider.seen_messages[-1]


# --- conversation shape ------------------------------------------------------


def test_the_tool_exchange_keeps_the_protocol_shape(settings, chats, secrets, tools_env):
    settings.tools.model_tool_use = True
    provider = ScriptedProvider(
        [_chunk_text("Ich sehe nach."), _chunk_call("c1", "status_disk")],
        [_chunk_text("Noch 42 GB frei.")],
    )
    registry, _executor = tools_env
    registry.register(_read_spec())
    service = _service(settings, chats, secrets, tools_env, provider)

    conv = service.ensure_conversation(None)
    events = _send(service, conv.id)

    # Preamble streams, the tool runs, the answer streams.
    assert [e.kind for e in events] == [
        "delta", "tool_start", "tool_end", "delta", "done",
    ]
    assert "".join(e.text for e in events if e.kind == "delta").startswith("Ich sehe nach.")

    # The second request carried the exchange the protocol expects: the call
    # as an assistant message, its result as a tool message with the same id.
    second = _turn(provider)
    roles = [m.role for m in second]
    assert roles[-3:] == ["user", "assistant", "tool"] or roles[-2:] == ["assistant", "tool"]
    assistant = second[-2]
    tool_msg = second[-1]
    assert assistant.tool_calls and assistant.tool_calls[0].name == "status_disk"
    assert tool_msg.tool_call_id == assistant.tool_calls[0].id
    assert tool_msg.tool_name == "status_disk"
    assert json.loads(tool_msg.content)["free_gb"] == 42
    # The preamble travelled with the call, not into the answer.
    assert assistant.content == "Ich sehe nach."


def test_the_first_turn_carries_the_planned_context(settings, chats, secrets, tools_env):
    settings.tools.model_tool_use = True
    provider = ScriptedProvider([_chunk_text("Danke.")])
    registry, _executor = tools_env
    registry.register(_read_spec())
    service = _service(settings, chats, secrets, tools_env, provider)

    conv = service.ensure_conversation(None)
    chats.add_message(conv.id, "user", "Vorhin: Hallo!")
    chats.add_message(conv.id, "assistant", "Hallo!")
    _send(service, conv.id, "und jetzt?")

    first = _turn(provider)
    roles = [m.role for m in first]
    assert roles == ["system", "user", "assistant", "user"]
    assert "und jetzt?" == first[-1].content


# --- refusals and limits -----------------------------------------------------


def test_an_unknown_tool_becomes_a_category_not_a_crash(settings, chats, secrets, tools_env):
    settings.tools.model_tool_use = True
    provider = ScriptedProvider(
        [_chunk_call("c1", "no_such_tool")],
        [_chunk_text("Dann eben ohne.")],
    )
    # A real tool must exist, or the turn takes the plain path and the model
    # never gets to name something wrong.
    registry, _executor = tools_env
    registry.register(_read_spec())
    service = _service(settings, chats, secrets, tools_env, provider)

    conv = service.ensure_conversation(None)
    events = _send(service, conv.id)

    assert [e.kind for e in events] == ["tool_end", "delta", "done"]
    assert events[0].ok is False
    assert events[0].text == "unknown_tool"
    assert events[-1].text == "Dann eben ohne."
    # The model was told, in the category vocabulary, and could recover.
    tool_msg = _turn(provider)[-1]
    assert json.loads(tool_msg.content) == {"error": "unknown_tool"}


def test_a_step_limit_ends_the_turn_visibly(settings, chats, secrets, tools_env):
    settings.tools.max_steps = 2
    settings.tools.model_tool_use = True
    provider = ScriptedProvider(
        *([[_chunk_call(f"c{i}", "status_disk")] for i in range(10)])
    )
    registry, _executor = tools_env
    registry.register(_read_spec())
    service = _service(settings, chats, secrets, tools_env, provider)

    conv = service.ensure_conversation(None)
    events = _send(service, conv.id)

    error = [e for e in events if e.kind == "error"]
    assert len(error) == 1
    assert error[0].text == failure_text("step_limit")
    stored = chats.history(conv.id)[-1]
    assert stored.role == "assistant"
    assert failure_text("step_limit") in stored.content


def test_the_providers_own_error_sentence_survives(settings, chats, secrets, tools_env):
    settings.tools.model_tool_use = True
    message = (
        "Ollama unter http://127.0.0.1:11434 ist nicht erreichbar. "
        "Läuft der Dienst?"
    )

    class _Broken:
        id = "broken"

        async def stream_chat(self, messages, *, model, temperature=0.7, num_ctx=None):
            raise ProviderError(message, code="network")

        async def stream_chat_tools(self, messages, *, model, temperature=0.7, tools, num_ctx=None):
            raise ProviderError(message, code="network")
            yield  # pragma: no cover

    registry, _executor = tools_env
    registry.register(_read_spec())
    service = _service(settings, chats, secrets, tools_env, _Broken())

    conv = service.ensure_conversation(None)
    events = _send(service, conv.id)

    error = [e for e in events if e.kind == "error"]
    assert len(error) == 1
    assert error[0].text == message
    assert message in chats.history(conv.id)[-1].content


def test_a_non_provider_exception_stays_a_category(settings, chats, secrets, tools_env):
    settings.tools.model_tool_use = True
    trace: list[str] = []

    class _Exploding:
        id = "exploding"

        async def stream_chat(self, messages, *, model, temperature=0.7, num_ctx=None):
            yield "x"

        async def stream_chat_tools(self, messages, *, model, temperature=0.7, tools, num_ctx=None):
            raise RuntimeError(f"pfad geladen: {SECRET_NOTE}")
            yield  # pragma: no cover

    registry, _executor = tools_env
    registry.register(_read_spec())
    service = _service(settings, chats, secrets, tools_env, _Exploding())

    conv = service.ensure_conversation(None)
    events = _send(service, conv.id)
    del trace

    error = [e for e in events if e.kind == "error"]
    assert len(error) == 1
    # The category only: an exception quotes whatever it choked on, and the
    # chat transcript is not the place for that.
    assert error[0].text == failure_text("provider_error")
    assert SECRET_NOTE not in chats.history(conv.id)[-1].content


# --- confirmation ------------------------------------------------------------


def test_an_approved_write_runs_exactly_once(settings, chats, secrets, tools_env):
    settings.tools.model_tool_use = True
    counter = {"writes": 0}
    provider = ScriptedProvider(
        [_chunk_call("c1", "write_tool", {"text": SECRET_NOTE})],
        [_chunk_text("Angelegt.")],
    )
    registry, _executor = tools_env
    registry.register(_write_spec(counter))
    requests: list[ConfirmationRequest] = []

    async def _confirm(request: ConfirmationRequest) -> bool:
        requests.append(request)
        return True

    service = _service(settings, chats, secrets, tools_env, provider, confirm=_confirm)

    conv = service.ensure_conversation(None)
    events = _send(service, conv.id)

    assert counter["writes"] == 1
    assert len(requests) == 1
    assert requests[0].tool_name == "write_tool"
    assert requests[0].request_id
    assert [e.kind for e in events][-1] == "done"
    assert events[-1].text == "Angelegt."


def test_a_denied_write_runs_never_and_the_turn_recovers(settings, chats, secrets, tools_env):
    settings.tools.model_tool_use = True
    counter = {"writes": 0}
    provider = ScriptedProvider(
        [_chunk_call("c1", "write_tool", {"text": SECRET_NOTE})],
        [_chunk_text("Dann eben nicht.")],
    )
    registry, _executor = tools_env
    registry.register(_write_spec(counter))

    async def _confirm(_request: ConfirmationRequest) -> bool:
        return False

    service = _service(settings, chats, secrets, tools_env, provider, confirm=_confirm)

    conv = service.ensure_conversation(None)
    events = _send(service, conv.id)

    assert counter["writes"] == 0
    assert events[-1].kind == "done"
    assert events[-1].text == "Dann eben nicht."
    # The model saw the refusal as a category and answered around it.
    tool_msg = _turn(provider)[-1]
    assert json.loads(tool_msg.content) == {"error": "confirmation_rejected"}


def test_no_confirm_callback_refuses_the_write(settings, chats, secrets, tools_env):
    settings.tools.model_tool_use = True
    counter = {"writes": 0}
    provider = ScriptedProvider(
        [_chunk_call("c1", "write_tool", {"text": SECRET_NOTE})],
        [_chunk_text("Ohne Karte.")],
    )
    registry, _executor = tools_env
    registry.register(_write_spec(counter))
    service = _service(settings, chats, secrets, tools_env, provider)

    conv = service.ensure_conversation(None)
    events = _send(service, conv.id)

    # Nobody could be asked, so nobody approved: the write never runs, and
    # the turn still ends in an answer instead of a hang.
    assert counter["writes"] == 0
    assert events[-1].text == "Ohne Karte."


# --- panic mid-turn ----------------------------------------------------------


def test_panic_mid_turn_removes_the_tools_from_the_next_request(
    settings, chats, secrets, tools_env
):
    settings.tools.model_tool_use = True
    provider = ScriptedProvider(
        [_chunk_call("c1", "status_disk")],
        [_chunk_text("Ohne Werkzeuge.")],
    )
    registry, executor = tools_env
    registry.register(_read_spec())
    service = _service(settings, chats, secrets, tools_env, provider)

    # Flip privacy panic during the FIRST provider stream: the runner builds
    # the next request's tool list before the model is asked, so the flip has
    # to land before that to prove the list shrinks.
    original = provider.stream_chat_tools

    async def _wrapped(messages, **kwargs):
        if provider.calls == 0:
            settings.app.privacy_panic = True
        async for chunk in original(messages, **kwargs):
            yield chunk

    provider.stream_chat_tools = _wrapped  # type: ignore[method-assign]

    conv = service.ensure_conversation(None)
    events = _send(service, conv.id)

    assert provider.seen_tools[0], "first request offered the tool"
    assert provider.seen_tools[1] == [], "second request offered nothing"
    # The turn still completed: the model answered without its tools.
    assert events[-1].kind == "done"
    assert events[-1].text == "Ohne Werkzeuge."


# --- traces ------------------------------------------------------------------


def test_every_turn_leaves_a_content_free_trace(settings, chats, secrets, tools_env, tmp_path):
    settings.tools.model_tool_use = True
    trace_dir = tmp_path / "assistant"
    provider = ScriptedProvider(
        [_chunk_call("c1", "status_disk")],
        [_chunk_text("Noch 42 GB frei.")],
    )
    registry, _executor = tools_env
    registry.register(_read_spec())
    service = _service(settings, chats, secrets, tools_env, provider, trace_dir=trace_dir)

    conv = service.ensure_conversation(None)
    _send(service, conv.id)

    files = list(trace_dir.glob("*.jsonl"))
    assert len(files) == 1
    blob = files[0].read_text(encoding="utf-8")
    assert USER_TEXT not in blob
    assert "sk-test-secret" not in blob
    assert "status_disk" in blob
    for line in blob.splitlines():
        record = json.loads(line)
        assert "user_text" not in record
        assert record.get("user_text_length") is None or isinstance(
            record.get("user_text_length"), int
        )
