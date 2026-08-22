"""Ollama HTTP provider (`/api/chat` NDJSON stream)."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from kiki.ai.provider import ChatMessage, ProviderError, ProviderHealth, StreamChunk, ToolCall
from kiki.ai.vision import ollama_message_payload

log = logging.getLogger(__name__)


def parse_ollama_ndjson_line(line: str) -> tuple[str, bool, str | None]:
    """Return (delta, done, error). Empty delta is valid."""
    text = line.strip()
    if not text:
        return "", False, None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"invalid Ollama stream chunk: {exc}") from exc
    if "error" in payload and payload["error"]:
        return "", True, str(payload["error"])
    message = payload.get("message") or {}
    delta = str(message.get("content") or "")
    done = bool(payload.get("done"))
    return delta, done, None


def truncated_note(done_reason: str, *, produced_text: bool) -> str:
    """Explain a stream that ended without an answer.

    Thinking-capable models (qwen3-vl) can spend the whole context on internal
    deliberation and finish with `done_reason: length` and empty content. Left
    unexplained that reaches the user as a blank bubble.
    """
    if produced_text or done_reason != "length":
        return ""
    return (
        "Das Modell hat das Kontextfenster aufgebraucht, bevor eine Antwort "
        "entstand — bei denkenden Modellen meist zu langes internes Überlegen. "
        "Erhöhe `ai.ollama.num_ctx`, stelle eine kürzere Frage oder leere den Chat."
    )


# A closed, empty reasoning block placed in the assistant's turn. The model can
# no longer *open* one, so it starts answering immediately.
#
# Measured on qwen3-vl:4b with a real question, cache-busted:
#   think=false alone      43.0 s to first token, 15 726 characters of thinking
#   with this prefill       0.4 s to first token,      0 characters of thinking
# The answer stayed complete (3 794 characters), and a model without a thinking
# mode returns byte-identical output with or without it.
THINK_PREFILL = "<think>\n\n</think>\n\n"


def with_thinking_suppressed(
    messages: list[ChatMessage], *, suppress: bool
) -> list[ChatMessage]:
    """Append the prefill turn unless the caller wants the model to deliberate."""
    if not suppress:
        return list(messages)
    return [*messages, ChatMessage(role="assistant", content=THINK_PREFILL)]


def parse_ollama_tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    """Read `message.tool_calls`. Ollama sends arguments already decoded."""
    raw = message.get("tool_calls") or []
    if not isinstance(raw, list):
        return ()
    calls: list[ToolCall] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function") or {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        arguments = function.get("arguments")
        parse_error = ""
        if isinstance(arguments, str):
            # Some Ollama builds forward a JSON string instead of an object.
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                arguments, parse_error = {}, f"ungültiges Argument-JSON: {exc}"
        if not isinstance(arguments, dict):
            arguments, parse_error = {}, parse_error or "Argumente sind kein Objekt"
        calls.append(
            ToolCall(
                id=str(entry.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                name=name,
                arguments=arguments,
                parse_error=parse_error,
            )
        )
    return tuple(calls)


class OllamaProvider:
    id = "ollama"

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
        num_ctx: int = 0,
        think: bool = False,
        suppress_thinking: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._timeout = timeout
        self._num_ctx = int(num_ctx)
        self._think = bool(think)
        # Only meaningful while think is off; deliberating on purpose wins.
        self._suppress = bool(suppress_thinking) and not bool(think)

    def _options(self, temperature: float, num_ctx: int | None = None) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature}
        # A per-turn window from the context planner wins over the setting:
        # a greeting must not pay the prefill of a code review.
        window = int(num_ctx) if num_ctx else self._num_ctx
        if window > 0:
            # Ollama defaults to 4096, which the system prompt, memories and a
            # thinking model's deliberation can exhaust before any answer.
            options["num_ctx"] = window
        return options

    def _http(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        return httpx.AsyncClient(timeout=httpx.Timeout(self._timeout, connect=5.0))

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.7,
        num_ctx: int | None = None,
    ) -> AsyncIterator[str]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                ollama_message_payload(m)
                for m in with_thinking_suppressed(messages, suppress=self._suppress)
            ],
            "stream": True,
            "options": self._options(temperature, num_ctx),
            "think": self._think,
        }
        owns = self._client is None
        client = self._http()
        produced = False
        done_reason = ""
        try:
            try:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=body) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode("utf-8", "replace")[:400]
                        raise ProviderError(
                            f"Ollama HTTP {response.status_code}: {detail or response.reason_phrase}",
                            code="http",
                        )
                    async for line in response.aiter_lines():
                        text = line.strip()
                        if not text:
                            continue
                        delta, _done, error = parse_ollama_ndjson_line(text)
                        if error:
                            raise ProviderError(error, code="ollama")
                        if delta:
                            produced = True
                            yield delta
                        else:
                            done_reason = _done_reason(text) or done_reason
            except httpx.HTTPError as exc:
                raise ProviderError(_ollama_connect_hint(self.base_url, exc), code="network") from exc
        finally:
            if owns:
                await client.aclose()
        note = truncated_note(done_reason, produced_text=produced)
        if note:
            raise ProviderError(note, code="context")

    async def stream_chat_tools(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.7,
        tools: list[dict[str, Any]],
        num_ctx: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                ollama_message_payload(m)
                for m in with_thinking_suppressed(messages, suppress=self._suppress)
            ],
            "stream": True,
            "options": self._options(temperature, num_ctx),
            "think": self._think,
        }
        if tools:
            body["tools"] = tools
        owns = self._client is None
        client = self._http()
        produced = False
        done_reason = ""
        try:
            try:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=body) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode("utf-8", "replace")[:400]
                        raise ProviderError(
                            f"Ollama HTTP {response.status_code}: {detail or response.reason_phrase}",
                            code="http",
                        )
                    async for line in response.aiter_lines():
                        text = line.strip()
                        if not text:
                            continue
                        try:
                            payload = json.loads(text)
                        except json.JSONDecodeError as exc:
                            raise ProviderError(f"invalid Ollama stream chunk: {exc}") from exc
                        if payload.get("error"):
                            raise ProviderError(str(payload["error"]), code="ollama")
                        message = payload.get("message") or {}
                        delta = str(message.get("content") or "")
                        calls = parse_ollama_tool_calls(message)
                        if payload.get("done_reason"):
                            done_reason = str(payload["done_reason"])
                        if delta or calls:
                            produced = True
                            yield StreamChunk(text=delta, tool_calls=calls)
            except httpx.HTTPError as exc:
                raise ProviderError(_ollama_connect_hint(self.base_url, exc), code="network") from exc
        finally:
            if owns:
                await client.aclose()
        note = truncated_note(done_reason, produced_text=produced)
        if note:
            raise ProviderError(note, code="context")

    async def list_models(self) -> list[str]:
        owns = self._client is None
        client = self._http()
        try:
            try:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ProviderError(_ollama_connect_hint(self.base_url, exc), code="network") from exc
            payload = response.json()
            names = [str(item.get("name")) for item in payload.get("models", []) if item.get("name")]
            return names
        finally:
            if owns:
                await client.aclose()

    async def ping(self, model: str) -> ProviderHealth:
        try:
            models = await self.list_models()
        except ProviderError as exc:
            return ProviderHealth(ok=False, detail=str(exc), models=(), selected_model_present=False)
        present = any(name == model or name.startswith(f"{model}:") for name in models)
        if not models:
            detail = f"Ollama antwortet, aber es ist kein Modell installiert. `ollama pull {model}`"
        elif not present:
            detail = (
                f"Ollama ist erreichbar. Modell „{model}“ fehlt. "
                f"Installieren: `ollama pull {model}`. "
                f"Installiert: {', '.join(models[:8]) or '—'}"
            )
        else:
            detail = f"Ollama ist erreichbar. Modell {model} ist verfügbar."
        return ProviderHealth(ok=True, detail=detail, models=tuple(models), selected_model_present=present)


def _done_reason(line: str) -> str:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return ""
    return str(payload.get("done_reason") or "") if isinstance(payload, dict) else ""


def _ollama_connect_hint(base_url: str, exc: Exception) -> str:
    return (
        f"Ollama unter {base_url} ist nicht erreichbar ({exc}). "
        "Ist der Dienst gestartet? `systemctl --user status ollama` oder `ollama serve`."
    )
