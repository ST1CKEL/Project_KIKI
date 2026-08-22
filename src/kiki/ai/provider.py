"""Provider-agnostic LLM types. Implementations live in sibling modules."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class ProviderError(Exception):
    """Network, auth, or protocol failure talking to a model server."""

    def __init__(self, message: str, *, code: str = "provider") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ToolCall:
    """One model-requested tool invocation.

    `arguments` is already decoded. A model that emits malformed JSON yields a
    call with empty arguments and a non-empty `parse_error`; the loop turns that
    into a tool error message instead of guessing what the model meant.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    parse_error: str = ""


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    # Raw base64 image payloads (no data: prefix). Empty for text-only turns.
    images: tuple[str, ...] = ()
    # Set on assistant turns that requested tools.
    tool_calls: tuple[ToolCall, ...] = ()
    # Set on role="tool" turns. OpenAI links results by id, Ollama by name, so
    # both are carried and each provider picks the one it understands.
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True)
class StreamChunk:
    """One step of a tool-capable stream: text, tool requests, or both."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ProviderHealth:
    ok: bool
    detail: str
    models: tuple[str, ...] = ()
    selected_model_present: bool = False


@runtime_checkable
class LLMProvider(Protocol):
    id: str

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.7,
        num_ctx: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas. Must not raise on empty streams; use ProviderError.

        `num_ctx` is the context window the planner sized for this turn. A
        provider that has no such knob ignores it.
        """
        ...

    async def list_models(self) -> list[str]:
        ...

    async def ping(self, model: str) -> ProviderHealth:
        ...


@runtime_checkable
class ToolCapableProvider(Protocol):
    """A provider that can be handed tool declarations and return tool calls.

    Kept separate from `LLMProvider` so a provider without function calling stays
    valid and the agent loop can detect the difference instead of assuming it.
    """

    async def stream_chat_tools(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.7,
        tools: list[dict[str, Any]],
        num_ctx: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Yield text deltas and any tool calls the model requested."""
        ...
