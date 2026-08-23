"""Reduce a streaming provider to one decision per harness step.

`ToolCapableProvider` yields `StreamChunk`s: text deltas and tool calls, in any
order and any number. The harness wants exactly one answer to "what now" — a
tool call or a final text — so the whole stream for one step is consumed here
and collapsed.

The rules are deliberately strict, because a loose reading is how an agent ends
up doing something nobody asked for:

* one complete tool call wins over any text that came with it — the text is a
  preamble, not an answer, and showing it would tell the user something is
  finished when it is not;
* more than one tool call in a step is a protocol error, not a queue;
* a step that produces neither text nor a call is a protocol error;
* a provider or transport failure becomes a category, never a message.

Nothing is executed here. The adapter proposes; the runner decides.
"""

from __future__ import annotations

import logging
from typing import Any

from kiki.harness.models import ActionKind, CancelToken, ModelAction, ToolCall, ToolResult

log = logging.getLogger(__name__)


class ProviderError(Exception):
    """The provider could not be reached or spoke nonsense. Category only."""

    def __init__(self, code: str = "provider_error") -> None:
        super().__init__(code)
        self.code = code


class ProviderModelAdapter:
    """Wraps a `ToolCapableProvider` for one harness step at a time."""

    def __init__(
        self,
        provider: Any,
        *,
        model: str,
        temperature: float = 0.2,
        system_prompt: str = "",
    ) -> None:
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._system_prompt = system_prompt

    async def next_action(
        self,
        *,
        user_text: str,
        tool_schemas: list[dict[str, Any]],
        observations: list[ToolResult],
        cancel_token: CancelToken,
    ) -> ModelAction:
        messages = self._messages(user_text, observations)
        tools = [_declaration(schema) for schema in tool_schemas]
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        stream = self._provider.stream_chat_tools(
            messages, model=self._model, temperature=self._temperature, tools=tools
        )
        try:
            async for chunk in stream:
                if cancel_token.cancelled:
                    # Leaving the loop closes the generator, which closes the
                    # provider's HTTP response with it.
                    break
                text = getattr(chunk, "text", "")
                if isinstance(text, str) and text:
                    text_parts.append(text)
                for raw in getattr(chunk, "tool_calls", ()) or ():
                    calls.append(_to_call(raw))
        except Exception as exc:
            log.warning("harness provider step failed: %s", type(exc).__name__)
            raise ProviderError from exc
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    log.debug("closing the provider stream failed", exc_info=True)

        if cancel_token.cancelled:
            # The runner turns this into CANCELLED at its next checkpoint; the
            # action is never used.
            return ModelAction.answer("abgebrochen")
        if len(calls) > 1:
            return ModelAction(ActionKind.TOOL_CALL)  # refused by validate_action
        if calls:
            call = calls[0]
            if not call.name or not isinstance(call.arguments, dict):
                return ModelAction(ActionKind.TOOL_CALL)
            # A tool call wins: whatever text came with it is a preamble, and
            # showing it as the answer would claim the work is done.
            return ModelAction(ActionKind.TOOL_CALL, tool_call=call)
        answer = "".join(text_parts).strip()
        if not answer:
            return ModelAction(ActionKind.FINAL)  # refused by validate_action
        return ModelAction.answer(answer)

    def _messages(self, user_text: str, observations: list[ToolResult]) -> list[Any]:
        from kiki.ai.provider import ChatMessage

        messages: list[Any] = []
        if self._system_prompt:
            messages.append(ChatMessage(role="system", content=self._system_prompt))
        messages.append(ChatMessage(role="user", content=user_text))
        for result in observations:
            messages.append(
                ChatMessage(
                    role="tool",
                    content=_observation_text(result),
                    tool_name=result.name,
                    tool_call_id=result.call_id,
                )
            )
        return messages


def _observation_text(result: ToolResult) -> str:
    """What a tool result looks like to the model. Data or a category."""
    import json

    if result.ok:
        return json.dumps(result.data or {}, ensure_ascii=False, sort_keys=True)
    return json.dumps({"error": result.error_code or "tool_failed"}, ensure_ascii=False)


def _declaration(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema.get("input_schema", {}),
        },
    }


def _to_call(raw: Any) -> ToolCall:
    """Translate a provider tool call, keeping a malformed one malformed.

    A call the provider could not parse arrives with `parse_error` set; it is
    given an empty name so `validate_action` refuses it, rather than being
    quietly repaired into something plausible.
    """
    if getattr(raw, "parse_error", ""):
        return ToolCall(name="", arguments={})
    name = getattr(raw, "name", "")
    arguments = getattr(raw, "arguments", {})
    call_id = getattr(raw, "id", "") or ""
    call = ToolCall(name=name if isinstance(name, str) else "", arguments={})
    if isinstance(arguments, dict):
        object.__setattr__(call, "arguments", dict(arguments))
    else:
        object.__setattr__(call, "name", "")
    if call_id:
        object.__setattr__(call, "id", f"call-{call_id}"[:32])
    return call
