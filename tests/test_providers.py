from __future__ import annotations

import asyncio
import json

import httpx

from kiki.ai.ollama import OllamaProvider, parse_ollama_ndjson_line
from kiki.ai.openai_compatible import OpenAICompatibleProvider, parse_openai_sse_data
from kiki.ai.prompts import attach_status_block, build_messages
from kiki.ai.provider import ChatMessage, ProviderError


def test_parse_ollama_ndjson() -> None:
    delta, done, err = parse_ollama_ndjson_line('{"message":{"content":"He"},"done":false}')
    assert delta == "He" and done is False and err is None
    delta, done, err = parse_ollama_ndjson_line('{"message":{"content":""},"done":true}')
    assert done is True
    delta, done, err = parse_ollama_ndjson_line('{"error":"model not found"}')
    assert err == "model not found"


def test_parse_openai_sse() -> None:
    delta, done = parse_openai_sse_data(
        json.dumps({"choices": [{"delta": {"content": "Hi"}}]})
    )
    assert delta == "Hi" and done is False
    delta, done = parse_openai_sse_data("[DONE]")
    assert done is True


def test_ollama_stream() -> None:
    chunks = [
        {"message": {"role": "assistant", "content": "Hallo"}, "done": False},
        {"message": {"role": "assistant", "content": " KIKI"}, "done": True},
    ]
    payload = "\n".join(json.dumps(c) for c in chunks) + "\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/chat")
        return httpx.Response(200, content=payload.encode())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider("http://127.0.0.1:11434", client=client)

    async def _run() -> str:
        parts: list[str] = []
        async for delta in provider.stream_chat(
            [ChatMessage("user", "hi")], model="llama3.2"
        ):
            parts.append(delta)
        return "".join(parts)

    assert asyncio.run(_run()) == "Hallo KIKI"


def test_ollama_health_recommends_the_selected_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/tags")
        return httpx.Response(200, json={"models": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider("http://127.0.0.1:11434", client=client)
    health = asyncio.run(provider.ping("qwen3-vl:8b"))

    assert health.ok is True
    assert health.selected_model_present is False
    assert "ollama pull qwen3-vl:8b" in health.detail
    assert "qwen3-vl:4b" not in health.detail


def test_openai_stream_and_auth() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret"
        body = (
            'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body.encode())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider("https://api.x.ai/v1", api_key="secret", client=client)

    async def _run() -> str:
        parts: list[str] = []
        async for delta in provider.stream_chat([ChatMessage("user", "hi")], model="grok-4.5"):
            parts.append(delta)
        return "".join(parts)

    assert asyncio.run(_run()) == "Hi"

    empty = OpenAICompatibleProvider("https://api.x.ai/v1", api_key=None)

    async def _fail() -> None:
        async for _ in empty.stream_chat([ChatMessage("user", "x")], model="grok-4.5"):
            pass

    try:
        asyncio.run(_fail())
        raise AssertionError("expected ProviderError")
    except ProviderError as exc:
        assert "Keyring" in str(exc) or "API-Key" in str(exc)


def test_build_messages_no_hidden_status() -> None:
    messages = build_messages(
        system_prompt="sys",
        history=[ChatMessage("user", "a"), ChatMessage("assistant", "b")],
        user_text="c",
        history_limit=10,
    )
    assert messages[0].role == "system"
    assert messages[-1].content == "c"
    assert all("kiki-status" not in m.content for m in messages)


def test_status_block_is_visible() -> None:
    text = attach_status_block("Wie spät?", {"time": "12:00"})
    assert "```kiki-status" in text
    assert "Wie spät?" in text


def test_truncated_note_only_fires_on_a_silent_length_stop() -> None:
    from kiki.ai.ollama import truncated_note

    # The case that produced a blank chat bubble: the model spent the whole
    # context window thinking and emitted no answer.
    assert "Kontextfenster" in truncated_note("length", produced_text=False)
    # Any answer at all means the run was fine, however it ended.
    assert truncated_note("length", produced_text=True) == ""
    assert truncated_note("stop", produced_text=False) == ""
    assert truncated_note("", produced_text=False) == ""


def test_num_ctx_is_sent_only_when_configured() -> None:
    from kiki.ai.ollama import OllamaProvider

    assert "num_ctx" not in OllamaProvider("http://x")._options(0.5)
    options = OllamaProvider("http://x", num_ctx=8192)._options(0.5)
    assert options["num_ctx"] == 8192
    assert options["temperature"] == 0.5


def test_thinking_prefill_is_appended_only_when_suppressing() -> None:
    from kiki.ai.ollama import THINK_PREFILL, with_thinking_suppressed
    from kiki.ai.provider import ChatMessage

    base = [ChatMessage(role="user", content="Hallo")]

    off = with_thinking_suppressed(base, suppress=False)
    assert off == base

    on = with_thinking_suppressed(base, suppress=True)
    assert len(on) == 2
    assert on[-1].role == "assistant"
    assert on[-1].content == THINK_PREFILL
    # A closed, empty block: the model cannot open a new one.
    assert THINK_PREFILL.strip() == "<think>\n\n</think>"
    # The original list is not mutated.
    assert len(base) == 1


def test_deliberating_on_purpose_disables_the_prefill() -> None:
    """think=true means the user wants reasoning; the prefill would defeat it."""
    from kiki.ai.ollama import OllamaProvider

    assert OllamaProvider("http://x", think=False, suppress_thinking=True)._suppress is True
    assert OllamaProvider("http://x", think=True, suppress_thinking=True)._suppress is False
    assert OllamaProvider("http://x", think=False, suppress_thinking=False)._suppress is False
