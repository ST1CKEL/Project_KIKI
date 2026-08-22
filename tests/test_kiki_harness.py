"""KIKI's own LLM harness: the service protocol and the provider talking to it."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

from kiki.ai.kiki_harness import KikiHarnessProvider
from kiki.ai.provider import ChatMessage, ProviderError

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "services" / "kiki-llm" / "kiki_llm_server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("kiki_llm_server", SERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def echo_server():
    """The service with its echo engine: no torch, no GPU, no download."""
    mod = _load_server()
    from http.server import ThreadingHTTPServer

    mod.LlmHandler.engine = mod.EchoEngine()
    mod.LlmHandler.budget = mod.VramBudget(total_bytes=1)
    mod.LlmHandler.slots = mod.Slots(2)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), mod.LlmHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield mod, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def _post(port, path, payload):
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
    response = conn.getresponse()
    body = response.read().decode("utf-8")
    conn.close()
    return response.status, body


# --- service ----------------------------------------------------------------


def test_health_reports_model_slots_and_vram(echo_server) -> None:
    _mod, port = echo_server
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/health")
    payload = json.loads(conn.getresponse().read())
    conn.close()
    assert payload["ok"] is True and payload["ready"] is True
    assert payload["model"] == "echo"
    assert payload["slots"] == 2
    assert "vram_free" in payload


def test_generate_streams_ndjson_and_terminates(echo_server) -> None:
    _mod, port = echo_server
    status, body = _post(port, "/v1/generate", {"messages": [{"role": "user", "content": "Hallo"}]})
    assert status == 200
    lines = [json.loads(x) for x in body.strip().splitlines()]
    assert lines[-1] == {"done": True}
    assert "".join(x.get("delta", "") for x in lines).startswith("Echo: Hallo")


def test_malformed_requests_are_refused(echo_server) -> None:
    _mod, port = echo_server
    assert _post(port, "/v1/generate", {})[0] == 400
    assert _post(port, "/v1/generate", {"messages": []})[0] == 400
    assert _post(port, "/nope", {"messages": [{"role": "user", "content": "x"}]})[0] == 404


def test_the_service_refuses_to_leave_loopback() -> None:
    mod = _load_server()
    assert mod._bind_host("127.0.0.1") == "127.0.0.1"
    assert mod._bind_host("localhost") == "127.0.0.1"
    for host in ("0.0.0.0", "192.168.1.10", "::"):
        with pytest.raises(SystemExit):
            mod._bind_host(host)


def test_reasoning_openers_are_banned_by_token_not_by_prose() -> None:
    """The prefill trick works, but banning the opener is the honest version."""
    mod = _load_server()
    assert "<think>" in mod.THINK_OPENERS


# --- slots ------------------------------------------------------------------


def test_slots_admit_up_to_capacity_then_refuse() -> None:
    mod = _load_server()
    slots = mod.Slots(2)
    assert slots.acquire("a", mod.Priority.HIGH, timeout=0.1) is True
    assert slots.acquire("b", mod.Priority.LOW, timeout=0.1) is True
    assert slots.acquire("c", mod.Priority.HIGH, timeout=0.1) is False

    slots.release("a")
    assert slots.acquire("c", mod.Priority.HIGH, timeout=0.1) is True
    assert set(slots.active()) == {"b", "c"}


def test_releasing_an_unknown_slot_does_not_free_capacity() -> None:
    mod = _load_server()
    slots = mod.Slots(1)
    slots.release("never-acquired")
    assert slots.acquire("a", mod.Priority.HIGH, timeout=0.1) is True
    assert slots.acquire("b", mod.Priority.HIGH, timeout=0.1) is False


# --- provider ---------------------------------------------------------------


def test_provider_streams_from_the_service(echo_server) -> None:
    _mod, port = echo_server
    provider = KikiHarnessProvider(f"http://127.0.0.1:{port}")

    async def go():
        return "".join(
            [
                d
                async for d in provider.stream_chat(
                    [ChatMessage(role="user", content="Hallo")], model="egal"
                )
            ]
        )

    assert asyncio.run(go()).startswith("Echo: Hallo")


def test_provider_reports_an_unreachable_service_clearly() -> None:
    provider = KikiHarnessProvider("http://127.0.0.1:1")

    async def go():
        async for _ in provider.stream_chat([ChatMessage(role="user", content="x")], model="e"):
            pass

    with pytest.raises(ProviderError, match="nicht erreichbar"):
        asyncio.run(go())


def test_ping_describes_the_loaded_model(echo_server) -> None:
    _mod, port = echo_server
    health = asyncio.run(KikiHarnessProvider(f"http://127.0.0.1:{port}").ping("egal"))
    assert health.ok is True
    assert "echo" in health.detail


def test_the_harness_is_tool_capable() -> None:
    from kiki.ai.provider import LLMProvider, ToolCapableProvider

    provider = KikiHarnessProvider("http://127.0.0.1:1")
    assert isinstance(provider, ToolCapableProvider)
    assert isinstance(provider, LLMProvider)


def test_tool_calls_arrive_in_the_same_stream_as_text(echo_server) -> None:
    """The echo engine emits a call when the prompt mentions a Werkzeug."""
    _mod, port = echo_server
    provider = KikiHarnessProvider(f"http://127.0.0.1:{port}")
    tools = [{"type": "function", "function": {"name": "status_disk", "parameters": {}}}]

    async def go():
        return [
            c
            async for c in provider.stream_chat_tools(
                [ChatMessage(role="user", content="Nutze ein Werkzeug")],
                model="egal",
                tools=tools,
            )
        ]

    chunks = asyncio.run(go())
    calls = [c for chunk in chunks for c in chunk.tool_calls]
    assert len(calls) == 1
    assert calls[0].name == "status_disk"
    assert calls[0].parse_error == ""


def test_tool_results_and_calls_survive_the_round_trip(echo_server) -> None:
    """A follow-up turn carries the assistant's call and the tool's answer."""
    from kiki.ai.kiki_harness import _payload
    from kiki.ai.provider import ToolCall

    assistant = ChatMessage(
        role="assistant",
        content="",
        tool_calls=(ToolCall(id="c1", name="status_disk", arguments={"path": "/"}),),
    )
    rendered = _payload(assistant)
    assert rendered["tool_calls"][0]["function"]["name"] == "status_disk"
    assert rendered["tool_calls"][0]["function"]["arguments"] == {"path": "/"}

    result = ChatMessage(
        role="tool", content='{"free_gb": 42}', tool_call_id="c1", tool_name="status_disk"
    )
    assert _payload(result)["name"] == "status_disk"


def test_a_broken_call_reaches_the_loop_as_an_error_not_a_guess() -> None:
    from kiki.ai.kiki_harness import _tool_call

    call = _tool_call({"id": "c1", "name": "", "parse_error": "ungültiges JSON"})
    assert call.parse_error == "ungültiges JSON"
    assert call.arguments == {}
