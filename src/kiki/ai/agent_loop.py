"""Model → tool → model orchestration. No GTK, no direct HTTP, no shell.

Every call still goes through `ToolExecutor`, so the policy, the approval card
and the audit log stay on the path exactly as they are for a user-clicked
action. What changes here is only *who* may ask: `Origin.MODEL` instead of
`Origin.USER`.

The loop is bounded on four axes — steps, total calls, repeated calls and result
size — because a model that misreads a tool result will otherwise retry it
forever.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from kiki.ai.provider import ChatMessage, ProviderError, ToolCall, ToolCapableProvider
from kiki.tools.executor import ConfirmFn, ToolExecutor
from kiki.tools.policy import Origin

log = logging.getLogger(__name__)

MAX_STEPS = 6
MAX_TOOL_CALLS = 12
RESULT_LIMIT = 8192


@dataclass(frozen=True)
class LoopEvent:
    """One observable step. `tool_end` carries the outcome, never the raw data."""

    kind: str  # delta | tool_start | tool_end | error | done
    text: str = ""
    tool: str = ""
    title: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    ok: bool = True


def _signature(call: ToolCall) -> str:
    return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"


def _serialize(payload: Any, *, limit: int = RESULT_LIMIT) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(payload)
    if len(text) > limit:
        return text[:limit] + f"… [gekürzt, {len(text)} Zeichen gesamt]"
    return text


class AgentLoop:
    def __init__(
        self,
        provider: ToolCapableProvider,
        executor: ToolExecutor,
        *,
        max_steps: int = MAX_STEPS,
        max_tool_calls: int = MAX_TOOL_CALLS,
        result_limit: int = RESULT_LIMIT,
    ) -> None:
        self._provider = provider
        self._executor = executor
        self._max_steps = max(1, max_steps)
        self._max_tool_calls = max(0, max_tool_calls)
        self._result_limit = result_limit

    async def run(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float,
        tools: list[dict[str, Any]],
        panic_check: Callable[[], bool],
        integrations_check: Callable[[], bool],
        profile: str = "observe",
        confirm: ConfirmFn | None = None,
        num_ctx: int | None = None,
    ) -> AsyncIterator[LoopEvent]:
        """Stream the conversation until the model answers without asking for tools."""
        working = list(messages)
        used_calls = 0
        results: dict[str, str] = {}

        for step in range(self._max_steps):
            text_parts: list[str] = []
            calls: list[ToolCall] = []
            try:
                async for chunk in self._provider.stream_chat_tools(
                    working,
                    model=model,
                    temperature=temperature,
                    tools=tools,
                    num_ctx=num_ctx,
                ):
                    if chunk.text:
                        text_parts.append(chunk.text)
                        yield LoopEvent(kind="delta", text=chunk.text)
                    if chunk.tool_calls:
                        calls.extend(chunk.tool_calls)
            except ProviderError as exc:
                yield LoopEvent(kind="error", text=str(exc), ok=False)
                return

            answer = "".join(text_parts)
            if not calls:
                yield LoopEvent(kind="done", text=answer.strip())
                return

            working.append(
                ChatMessage(role="assistant", content=answer, tool_calls=tuple(calls))
            )

            for call in calls:
                if used_calls >= self._max_tool_calls:
                    working.append(self._tool_message(call, "Aufrufbudget erschöpft."))
                    yield LoopEvent(
                        kind="tool_end", tool=call.name, params=call.arguments, ok=False,
                        text="Aufrufbudget erschöpft.",
                    )
                    continue
                used_calls += 1

                if call.parse_error:
                    detail = f"Aufruf verworfen: {call.parse_error}"
                    working.append(self._tool_message(call, detail))
                    yield LoopEvent(
                        kind="tool_end", tool=call.name, params={}, ok=False, text=detail
                    )
                    continue

                signature = _signature(call)
                if signature in results:
                    # Answering from the earlier result breaks the retry cycle
                    # without pretending the tool ran twice.
                    working.append(
                        self._tool_message(
                            call, f"Bereits in diesem Zug ausgeführt. Ergebnis: {results[signature]}"
                        )
                    )
                    continue

                spec = self._executor.registry.get(call.name)
                title = spec.title if spec is not None else call.name
                yield LoopEvent(
                    kind="tool_start", tool=call.name, title=title, params=call.arguments
                )

                result = await self._executor.run(
                    call.name,
                    call.arguments,
                    panic=panic_check(),
                    integrations_enabled=integrations_check(),
                    confirm=confirm,
                    profile=profile,
                    origin=Origin.MODEL,
                )
                if result.ok:
                    payload = _serialize(result.data, limit=self._result_limit)
                else:
                    payload = _serialize(
                        {"error": result.error or "unbekannter Fehler"}, limit=self._result_limit
                    )
                results[signature] = payload
                working.append(self._tool_message(call, payload))
                yield LoopEvent(
                    kind="tool_end",
                    tool=call.name,
                    title=title,
                    params=call.arguments,
                    ok=result.ok,
                    text="" if result.ok else (result.error or ""),
                )

            if step == self._max_steps - 1:
                # Out of steps with tool results pending: report what we have
                # rather than presenting a half-finished turn as complete.
                yield LoopEvent(
                    kind="error",
                    text="Schrittlimit erreicht — KIKI hat die Aufgabe nicht abgeschlossen.",
                    ok=False,
                )
                return

    @staticmethod
    def _tool_message(call: ToolCall, content: str) -> ChatMessage:
        return ChatMessage(
            role="tool",
            content=content,
            tool_call_id=call.id or call.name,
            tool_name=call.name,
        )
