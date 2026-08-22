"""Provider-side tool-call decoding. Both wire formats, including the broken ones."""

from __future__ import annotations

import json

from kiki.ai.ollama import parse_ollama_tool_calls
from kiki.ai.openai_compatible import ToolCallAccumulator
from kiki.ai.provider import ChatMessage, ToolCall
from kiki.ai.vision import ollama_message_payload, openai_message_payload


def test_ollama_decodes_object_arguments() -> None:
    calls = parse_ollama_tool_calls(
        {"tool_calls": [{"function": {"name": "status_disk", "arguments": {"path": "/"}}}]}
    )
    assert len(calls) == 1
    assert calls[0].name == "status_disk"
    assert calls[0].arguments == {"path": "/"}
    assert calls[0].parse_error == ""
    assert calls[0].id  # a synthetic id is generated when Ollama omits one


def test_ollama_decodes_string_arguments() -> None:
    calls = parse_ollama_tool_calls(
        {"tool_calls": [{"function": {"name": "status_disk", "arguments": '{"path": "/home"}'}}]}
    )
    assert calls[0].arguments == {"path": "/home"}
    assert calls[0].parse_error == ""


def test_ollama_flags_broken_arguments_instead_of_guessing() -> None:
    calls = parse_ollama_tool_calls(
        {"tool_calls": [{"function": {"name": "status_disk", "arguments": "{oops"}}]}
    )
    assert calls[0].arguments == {}
    assert "ungültiges Argument-JSON" in calls[0].parse_error


def test_ollama_skips_nameless_and_malformed_entries() -> None:
    calls = parse_ollama_tool_calls(
        {
            "tool_calls": [
                {"function": {"name": "", "arguments": {}}},
                "not-a-dict",
                {"function": {"name": "status_disk", "arguments": {}}},
            ]
        }
    )
    assert [c.name for c in calls] == ["status_disk"]


def test_ollama_ignores_messages_without_tool_calls() -> None:
    assert parse_ollama_tool_calls({"content": "hallo"}) == ()
    assert parse_ollama_tool_calls({"tool_calls": "nope"}) == ()


def test_openai_accumulator_reassembles_split_arguments() -> None:
    acc = ToolCallAccumulator()
    acc.feed({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "status_disk"}}]})
    acc.feed({"tool_calls": [{"index": 0, "function": {"arguments": '{"pa'}}]})
    acc.feed({"tool_calls": [{"index": 0, "function": {"arguments": 'th": "/"}'}}]})
    calls = acc.finish()

    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].name == "status_disk"
    assert calls[0].arguments == {"path": "/"}


def test_openai_accumulator_keeps_parallel_calls_apart() -> None:
    acc = ToolCallAccumulator()
    acc.feed(
        {
            "tool_calls": [
                {"index": 0, "id": "a", "function": {"name": "status_disk", "arguments": "{}"}},
                {"index": 1, "id": "b", "function": {"name": "status_upower", "arguments": "{}"}},
            ]
        }
    )
    calls = acc.finish()
    assert [c.name for c in calls] == ["status_disk", "status_upower"]
    assert [c.id for c in calls] == ["a", "b"]


def test_openai_accumulator_reports_unparsable_arguments() -> None:
    acc = ToolCallAccumulator()
    acc.feed({"tool_calls": [{"index": 0, "function": {"name": "x", "arguments": "{broken"}}]})
    call = acc.finish()[0]
    assert call.arguments == {}
    assert "ungültiges Argument-JSON" in call.parse_error


def test_openai_accumulator_rejects_non_object_arguments() -> None:
    acc = ToolCallAccumulator()
    acc.feed({"tool_calls": [{"index": 0, "function": {"name": "x", "arguments": "[1,2]"}}]})
    call = acc.finish()[0]
    assert call.arguments == {}
    assert "kein Objekt" in call.parse_error


def test_openai_accumulator_is_empty_without_tool_calls() -> None:
    acc = ToolCallAccumulator()
    acc.feed({"content": "hallo"})
    assert acc.finish() == ()


def test_assistant_tool_calls_survive_both_wire_formats() -> None:
    message = ChatMessage(
        role="assistant",
        content="",
        tool_calls=(ToolCall(id="call_1", name="status_disk", arguments={"path": "/"}),),
    )

    ollama = ollama_message_payload(message)
    assert ollama["tool_calls"][0]["function"]["arguments"] == {"path": "/"}

    openai = openai_message_payload(message)
    fragment = openai["tool_calls"][0]
    assert fragment["id"] == "call_1"
    assert fragment["type"] == "function"
    # OpenAI wants arguments as a JSON string, Ollama as an object.
    assert json.loads(fragment["function"]["arguments"]) == {"path": "/"}


def test_tool_results_link_by_name_for_ollama_and_by_id_for_openai() -> None:
    message = ChatMessage(
        role="tool",
        content='{"free_gb": 42}',
        tool_call_id="call_1",
        tool_name="status_disk",
    )
    assert ollama_message_payload(message)["tool_name"] == "status_disk"
    assert openai_message_payload(message)["tool_call_id"] == "call_1"


def test_plain_messages_are_untouched_by_the_tool_fields() -> None:
    message = ChatMessage(role="user", content="hallo")
    assert ollama_message_payload(message) == {"role": "user", "content": "hallo"}
    assert openai_message_payload(message) == {"role": "user", "content": "hallo"}
