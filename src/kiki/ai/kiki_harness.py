"""Provider for KIKI's own LLM harness (`kiki-llm.service`).

Speaks the harness's NDJSON protocol. Deliberately thin: the interesting parts —
reasoning suppression at the token level, slots, the VRAM budget — live in the
service, because that is where the model is.

Tool calls travel in the same NDJSON stream as text, so the agent loop sees them
the moment the model finishes writing one. Declarations are rendered by the
model's own chat template inside the service — no prose description of tools, no
guessing at a convention.

Tools themselves stay ordinary KIKI tools: a handler may perfectly well call a
cloud API. What is local here is the *model*, not what the tools are allowed to
reach.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from kiki.ai.provider import (
    ChatMessage,
    ProviderError,
    ProviderHealth,
    StreamChunk,
    ToolCall,
)

log = logging.getLogger(__name__)


def _payload(message: ChatMessage) -> dict:
    """One message in the shape the chat template expects."""
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in message.tool_calls
        ]
    if message.role == "tool" and message.tool_name:
        out["name"] = message.tool_name
    return out


def _tool_call(raw: dict) -> ToolCall:
    return ToolCall(
        id=str(raw.get("id") or ""),
        name=str(raw.get("name") or ""),
        arguments=raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {},
        parse_error=str(raw.get("parse_error") or ""),
    )


class KikiHarnessProvider:
    id = "kiki_harness"

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 300.0,
        client: httpx.AsyncClient | None = None,
        priority: str = "high",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._timeout = timeout
        self._priority = priority

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
        async for chunk in self._stream(messages, temperature=temperature, tools=None):
            if chunk.text:
                yield chunk.text

    async def stream_chat_tools(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.7,
        tools: list[dict[str, Any]],
        num_ctx: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        async for chunk in self._stream(messages, temperature=temperature, tools=tools):
            yield chunk

    async def _stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        tools: list[dict[str, Any]] | None,
    ) -> AsyncIterator[StreamChunk]:
        # The harness sizes its own window; the planner's budget still governs
        # how much we put in, which is what actually costs time.
        body: dict[str, Any] = {
            "messages": [_payload(m) for m in messages],
            "temperature": temperature,
            "priority": self._priority,
            "suppress_reasoning": True,
        }
        if tools:
            body["tools"] = tools
        owns = self._client is None
        client = self._http()
        try:
            try:
                async with client.stream(
                    "POST", f"{self.base_url}/v1/generate", json=body
                ) as response:
                    if response.status_code == 503:
                        detail = (await response.aread()).decode("utf-8", "replace")[:200]
                        raise ProviderError(
                            f"KIKI-Harness ist beschäftigt oder lädt noch: {detail}",
                            code="busy",
                        )
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode("utf-8", "replace")[:400]
                        raise ProviderError(
                            f"Harness HTTP {response.status_code}: {detail}", code="http"
                        )
                    async for line in response.aiter_lines():
                        text = line.strip()
                        if not text:
                            continue
                        try:
                            chunk = json.loads(text)
                        except json.JSONDecodeError as exc:
                            raise ProviderError(f"ungültige Harness-Antwort: {exc}") from exc
                        if chunk.get("error"):
                            raise ProviderError(str(chunk["error"]), code="harness")
                        delta = chunk.get("delta")
                        if delta:
                            yield StreamChunk(text=str(delta))
                        raw_call = chunk.get("tool_call")
                        if isinstance(raw_call, dict):
                            yield StreamChunk(tool_calls=(_tool_call(raw_call),))
                        if chunk.get("done"):
                            break
            except httpx.HTTPError as exc:
                raise ProviderError(_hint(self.base_url, exc), code="network") from exc
        finally:
            if owns:
                await client.aclose()

    async def list_models(self) -> list[str]:
        health = await self._health()
        name = str(health.get("model") or "")
        return [name] if name else []

    async def ping(self, model: str) -> ProviderHealth:
        try:
            health = await self._health()
        except ProviderError as exc:
            return ProviderHealth(ok=False, detail=str(exc))
        name = str(health.get("model") or "unbekannt")
        ready = bool(health.get("ready"))
        vram = float(health.get("vram_bytes") or 0) / 1e9
        detail = (
            f"KIKI-Harness bereit. Modell {name}, {vram:.1f} GB VRAM, "
            f"{health.get('slots', '?')} Slots."
            if ready
            else f"KIKI-Harness erreichbar, Modell {name} lädt noch."
        )
        return ProviderHealth(
            ok=True, detail=detail, models=(name,), selected_model_present=ready
        )

    async def _health(self) -> dict:
        owns = self._client is None
        client = self._http()
        try:
            try:
                response = await client.get(f"{self.base_url}/health", timeout=10.0)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ProviderError(_hint(self.base_url, exc), code="network") from exc
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
        finally:
            if owns:
                await client.aclose()


def _hint(base_url: str, exc: Exception) -> str:
    return (
        f"KIKI-Harness unter {base_url} ist nicht erreichbar ({exc}). "
        "Läuft der Dienst? `systemctl --user status kiki-llm.service`"
    )
