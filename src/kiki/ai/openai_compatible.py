"""OpenAI-compatible Chat Completions (SpaceXAI, llama.cpp, vLLM, …)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from kiki.ai.provider import ChatMessage, ProviderError, ProviderHealth, StreamChunk, ToolCall
from kiki.ai.vision import openai_message_payload

log = logging.getLogger(__name__)


class ToolCallAccumulator:
    """Reassemble `delta.tool_calls` fragments, which arrive split by index.

    Name and id come once, arguments stream in as string pieces that only parse
    as JSON when complete, so nothing is decoded until `finish()`.
    """

    def __init__(self) -> None:
        self._slots: dict[int, dict[str, str]] = {}

    def feed(self, delta: dict[str, Any]) -> None:
        fragments = delta.get("tool_calls") or []
        if not isinstance(fragments, list):
            return
        for fragment in fragments:
            if not isinstance(fragment, dict):
                continue
            index = int(fragment.get("index", 0) or 0)
            slot = self._slots.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if fragment.get("id"):
                slot["id"] = str(fragment["id"])
            function = fragment.get("function") or {}
            if function.get("name"):
                slot["name"] = str(function["name"])
            piece = function.get("arguments")
            if piece:
                slot["arguments"] += str(piece)

    def finish(self) -> tuple[ToolCall, ...]:
        calls: list[ToolCall] = []
        for index in sorted(self._slots):
            slot = self._slots[index]
            name = slot["name"].strip()
            if not name:
                continue
            raw = slot["arguments"].strip()
            arguments: dict[str, Any] = {}
            parse_error = ""
            if raw:
                try:
                    decoded = json.loads(raw)
                except json.JSONDecodeError as exc:
                    parse_error = f"ungültiges Argument-JSON: {exc}"
                else:
                    if isinstance(decoded, dict):
                        arguments = decoded
                    else:
                        parse_error = "Argumente sind kein Objekt"
            calls.append(
                ToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=name,
                    arguments=arguments,
                    parse_error=parse_error,
                )
            )
        return tuple(calls)


def parse_openai_sse_data(data: str) -> tuple[str, bool]:
    """Parse one SSE `data:` payload. Returns (delta, done)."""
    text = data.strip()
    if not text:
        return "", False
    if text == "[DONE]":
        return "", True
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"invalid OpenAI stream chunk: {exc}") from exc
    if payload.get("error"):
        err = payload["error"]
        message = err.get("message", err) if isinstance(err, dict) else str(err)
        raise ProviderError(str(message), code="openai")
    choices = payload.get("choices") or []
    if not choices:
        return "", bool(payload.get("done"))
    choice = choices[0]
    if choice.get("finish_reason"):
        delta = ((choice.get("delta") or {}).get("content")) or ""
        return str(delta or ""), True
    delta = (choice.get("delta") or {}).get("content")
    if delta is None:
        # Non-stream fallback chunk
        message = choice.get("message") or {}
        delta = message.get("content")
    return str(delta or ""), False


class OpenAICompatibleProvider:
    id = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

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
        # num_ctx is an Ollama option; a hosted endpoint sizes its own window.
        del num_ctx
        if not self._api_key:
            raise ProviderError(
                "Kein API-Key im GNOME Keyring. Öffne die Einstellungen und speichere den Schlüssel.",
                code="auth",
            )
        body: dict[str, Any] = {
            "model": model,
            "messages": [openai_message_payload(m) for m in messages],
            "stream": True,
            "temperature": temperature,
        }
        owns = self._client is None
        client = self._http()
        try:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=body,
                ) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode("utf-8", "replace")[:400]
                        raise ProviderError(
                            f"API HTTP {response.status_code}: {detail or response.reason_phrase}",
                            code="http",
                        )
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].lstrip()
                        delta, done = parse_openai_sse_data(data)
                        if delta:
                            yield delta
                        if done:
                            break
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"OpenAI-kompatibler Endpunkt {self.base_url} ist nicht erreichbar ({exc}).",
                    code="network",
                ) from exc
        finally:
            if owns:
                await client.aclose()

    async def stream_chat_tools(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.7,
        tools: list[dict[str, Any]],
        num_ctx: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        del num_ctx
        if not self._api_key:
            raise ProviderError(
                "Kein API-Key im GNOME Keyring. Öffne die Einstellungen und speichere den Schlüssel.",
                code="auth",
            )
        body: dict[str, Any] = {
            "model": model,
            "messages": [openai_message_payload(m) for m in messages],
            "stream": True,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        owns = self._client is None
        client = self._http()
        accumulator = ToolCallAccumulator()
        try:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=body,
                ) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode("utf-8", "replace")[:400]
                        raise ProviderError(
                            f"API HTTP {response.status_code}: {detail or response.reason_phrase}",
                            code="http",
                        )
                    async for line in response.aiter_lines():
                        if not line or line.startswith(":") or not line.startswith("data:"):
                            continue
                        data = line[5:].lstrip().strip()
                        if not data:
                            continue
                        if data == "[DONE]":
                            break
                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise ProviderError(f"invalid OpenAI stream chunk: {exc}") from exc
                        if payload.get("error"):
                            err = payload["error"]
                            message = err.get("message", err) if isinstance(err, dict) else str(err)
                            raise ProviderError(str(message), code="openai")
                        choices = payload.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta") or choice.get("message") or {}
                        accumulator.feed(delta)
                        text = delta.get("content")
                        if text:
                            yield StreamChunk(text=str(text))
                        if choice.get("finish_reason"):
                            break
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"OpenAI-kompatibler Endpunkt {self.base_url} ist nicht erreichbar ({exc}).",
                    code="network",
                ) from exc
        finally:
            if owns:
                await client.aclose()
        calls = accumulator.finish()
        if calls:
            yield StreamChunk(tool_calls=calls)

    async def list_models(self) -> list[str]:
        if not self._api_key:
            raise ProviderError("Kein API-Key im GNOME Keyring.", code="auth")
        owns = self._client is None
        client = self._http()
        try:
            try:
                response = await client.get(f"{self.base_url}/models", headers=self._headers())
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"OpenAI-kompatibler Endpunkt {self.base_url} ist nicht erreichbar ({exc}).",
                    code="network",
                ) from exc
            payload = response.json()
            data = payload.get("data") or payload.get("models") or []
            names: list[str] = []
            for item in data:
                if isinstance(item, dict) and item.get("id"):
                    names.append(str(item["id"]))
                elif isinstance(item, str):
                    names.append(item)
            return names
        finally:
            if owns:
                await client.aclose()

    async def ping(self, model: str) -> ProviderHealth:
        try:
            models = await self.list_models()
        except ProviderError as exc:
            return ProviderHealth(ok=False, detail=str(exc), models=(), selected_model_present=False)
        present = model in models if models else True
        if models and not present:
            detail = f"Endpunkt erreichbar, Modell „{model}“ nicht in der Liste. Beispiele: {', '.join(models[:8])}"
        else:
            detail = f"Endpunkt {self.base_url} ist erreichbar."
        return ProviderHealth(ok=True, detail=detail, models=tuple(models), selected_model_present=present)
