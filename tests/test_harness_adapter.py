"""Reducing a real provider stream to exactly one decision."""

from __future__ import annotations

import asyncio

import pytest

from kiki.ai.provider import StreamChunk
from kiki.ai.provider import ToolCall as ProviderCall
from kiki.harness.adapter import ProviderError, ProviderModelAdapter
from kiki.harness.models import ActionKind, CancelToken, validate_action


class FakeProvider:
    """Replays chunks, records what it was asked, and notes if it was closed."""

    def __init__(self, chunks, *, raises: BaseException | None = None) -> None:
        self._chunks = list(chunks)
        self._raises = raises
        self.seen: list[dict] = []
        self.closed = False
        self.delivered = 0

    def stream_chat_tools(self, messages, *, model, temperature=0.7, tools, num_ctx=None):
        self.seen.append({"messages": list(messages), "model": model, "tools": list(tools)})
        return self._stream()

    async def _stream(self):
        try:
            if self._raises is not None:
                raise self._raises
            for chunk in self._chunks:
                self.delivered += 1
                yield chunk
                await asyncio.sleep(0)
        finally:
            self.closed = True


def _act(provider, *, token=None, observations=None):
    adapter = ProviderModelAdapter(provider, model="test", system_prompt="sei knapp")
    return asyncio.run(
        adapter.next_action(
            user_text="Wie ist dein Status?",
            tool_schemas=[{"name": "system_status", "description": "d", "input_schema": {}}],
            observations=observations or [],
            cancel_token=token or CancelToken(),
        )
    )


def test_text_only_becomes_a_final_answer() -> None:
    action = _act(FakeProvider([StreamChunk(text="Mir geht "), StreamChunk(text="es gut.")]))
    assert action.kind is ActionKind.FINAL
    assert action.final_text == "Mir geht es gut."
    assert validate_action(action) == ""


def test_one_tool_call_becomes_a_tool_action() -> None:
    provider = FakeProvider(
        [StreamChunk(tool_calls=(ProviderCall(id="a", name="system_status", arguments={}),))]
    )
    action = _act(provider)
    assert action.kind is ActionKind.TOOL_CALL
    assert action.tool_call.name == "system_status"
    assert validate_action(action) == ""


def test_a_tool_call_wins_over_the_text_that_came_with_it() -> None:
    """The preamble must never be shown as the answer: the work is not done."""
    provider = FakeProvider([
        StreamChunk(text="Ich schaue mal nach."),
        StreamChunk(tool_calls=(ProviderCall(id="a", name="system_status", arguments={}),)),
    ])
    action = _act(provider)

    assert action.kind is ActionKind.TOOL_CALL
    assert action.final_text is None


def test_two_tool_calls_are_a_protocol_error() -> None:
    provider = FakeProvider([
        StreamChunk(tool_calls=(
            ProviderCall(id="a", name="system_status", arguments={}),
            ProviderCall(id="b", name="system_status", arguments={}),
        ))
    ])
    assert validate_action(_act(provider)) == "model_protocol_error"


def test_an_unparsable_tool_call_is_a_protocol_error() -> None:
    provider = FakeProvider([
        StreamChunk(tool_calls=(
            ProviderCall(id="a", name="system_status", arguments={}, parse_error="kaputtes JSON"),
        ))
    ])
    assert validate_action(_act(provider)) == "model_protocol_error"


def test_a_stream_with_nothing_in_it_is_a_protocol_error() -> None:
    assert validate_action(_act(FakeProvider([]))) == "model_protocol_error"
    assert validate_action(_act(FakeProvider([StreamChunk(text="   ")]))) == "model_protocol_error"


def test_a_provider_failure_becomes_a_category() -> None:
    provider = FakeProvider([], raises=RuntimeError("https://intern:8080 refused"))
    with pytest.raises(ProviderError) as excinfo:
        _act(provider)
    assert excinfo.value.code == "provider_error"
    assert "intern" not in str(excinfo.value)


def test_a_cancel_closes_the_provider_stream() -> None:
    token = CancelToken()
    token.cancel()
    provider = FakeProvider([StreamChunk(text="zu spät")] * 50)

    _act(provider, token=token)
    assert provider.closed is True
    assert provider.delivered <= 1


def test_the_stream_is_closed_even_on_success() -> None:
    provider = FakeProvider([StreamChunk(text="fertig")])
    _act(provider)
    assert provider.closed is True


def test_the_tools_reach_the_provider_as_declarations() -> None:
    provider = FakeProvider([StreamChunk(text="fertig")])
    _act(provider)
    declaration = provider.seen[0]["tools"][0]
    assert declaration["type"] == "function"
    assert declaration["function"]["name"] == "system_status"


def test_observations_come_back_as_tool_messages() -> None:
    from kiki.harness.models import ToolResult

    provider = FakeProvider([StreamChunk(text="fertig")])
    _act(provider, observations=[
        ToolResult(call_id="c1", name="system_status", ok=True, data={"ok": True})
    ])
    roles = [message.role for message in provider.seen[0]["messages"]]
    assert roles == ["system", "user", "tool"]


def test_a_failed_observation_is_reduced_to_its_category() -> None:
    from kiki.harness.models import ToolResult

    provider = FakeProvider([StreamChunk(text="fertig")])
    _act(provider, observations=[
        ToolResult(call_id="c1", name="create_note", ok=False, error_code="note_exists")
    ])
    tool_message = provider.seen[0]["messages"][-1]
    assert "note_exists" in tool_message.content
