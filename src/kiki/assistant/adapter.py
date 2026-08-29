"""The model side of one assistant step: a stream that ends in one decision.

`ToolCapableProvider` yields text deltas and tool calls mixed together. The
runner wants both at once -- deltas for the person watching, and exactly one
`ModelAction` for itself. A step adapter supplies both: every `delta` it yields
is shown immediately, the final `action` is what the runner acts on.

The collapse rules are the harness's, deliberately strict:

* one complete tool call wins over any text that came with it -- the text was a
  preamble, and the run is not over;
* more than one tool call in a stream is a protocol error, not a queue: the
  runner rejects the action instead of executing an unordered pile;
* a stream with neither text nor a call is a protocol error too;
* a call the provider could not parse keeps its parse error as an empty name,
  so validation refuses it rather than repairing it into something plausible.

Nothing here runs tools. The adapter proposes; the runner decides.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from kiki.harness.adapter import ProviderError
from kiki.harness.models import ActionKind, CancelToken, ModelAction, ToolCall, ToolResult

log = logging.getLogger(__name__)

# Tool results go back to the model in the next step's messages. Beyond this
# bound a result is truncated, not dropped: the model keeps the beginning of
# what it asked for.
RESULT_LIMIT = 8192


@dataclass(frozen=True)
class StepEvent:
    """One piece of a model step. Exactly one `action` event, at the end."""

    kind: str  # delta | action
    text: str = ""
    action: ModelAction | None = None


class StepAdapter(Protocol):
    async def next_action_stream(
        self,
        *,
        user_text: str,
        tool_schemas: list[dict[str, Any]],
        observations: list[ToolResult],
        cancel_token: CancelToken,
    ) -> AsyncIterator[StepEvent]: ...


def as_step_adapter(adapter: Any) -> Any:
    """Accept the older `next_action` protocol as a step adapter.

    Existing adapters (the provider adapter, the test fakes) answer one step
    without streaming. They still work: their single decision becomes the
    step's only event, with no deltas. New adapters stream and win deltas for
    the person watching; the runner cannot tell the difference.
    """
    if callable(getattr(adapter, "next_action_stream", None)):
        return adapter
    return _NextActionStepAdapter(adapter)


class _NextActionStepAdapter:
    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    async def next_action_stream(
        self,
        *,
        user_text: str,
        tool_schemas: list[dict[str, Any]],
        observations: list[ToolResult],
        cancel_token: CancelToken,
    ) -> AsyncIterator[StepEvent]:
        action = await self._adapter.next_action(
            user_text=user_text,
            tool_schemas=tool_schemas,
            observations=list(observations),
            cancel_token=cancel_token,
        )
        yield StepEvent(kind="action", action=action)


class ProviderStepAdapter:
    """Wraps a `ToolCapableProvider`: deltas out as they arrive, one decision."""

    def __init__(
        self,
        provider: Any,
        *,
        model: str,
        temperature: float = 0.2,
        system_prompt: str = "",
        num_ctx: int | None = None,
        result_limit: int = RESULT_LIMIT,
    ) -> None:
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._num_ctx = num_ctx
        self._result_limit = result_limit

    async def next_action_stream(
        self,
        *,
        user_text: str,
        tool_schemas: list[dict[str, Any]],
        observations: list[ToolResult],
        cancel_token: CancelToken,
    ) -> AsyncIterator[StepEvent]:
        messages = self._messages(user_text, observations)
        tools = [_declaration(schema) for schema in tool_schemas]
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        stream = self._provider.stream_chat_tools(
            messages,
            model=self._model,
            temperature=self._temperature,
            tools=tools,
            num_ctx=self._num_ctx,
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
                    yield StepEvent(kind="delta", text=text)
                for raw in getattr(chunk, "tool_calls", ()) or ():
                    calls.append(_to_call(raw))
        except Exception as exc:
            log.warning("assistant provider step failed: %s", type(exc).__name__)
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
            yield StepEvent(kind="action", action=ModelAction.answer("abgebrochen"))
            return
        yield StepEvent(kind="action", action=_collapse(text_parts, calls))

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
                    content=_observation_text(result, limit=self._result_limit),
                    tool_name=result.name,
                    tool_call_id=result.call_id,
                )
            )
        return messages


class ChatStepAdapter:
    """A provider step inside one chat turn: the conversation is given, the
    tool exchange is accumulated here.

    `ProviderStepAdapter` builds a flat [system, user, observations] exchange
    -- right for a one-question run. A chat turn brings its own planned
    context (system prompt, history, memories) and must grow it the way the
    provider protocols expect: an assistant message carrying the tool call,
    then one tool message per result. The adapter owns that working list for
    the turn; the runner never sees it.

    `last_provider_message` holds the provider's own error sentence after a
    failed step. The plain chat path has always shown that sentence, so the
    agent path keeps it -- the runner still gets a category only, the prose
    never passes through it.
    """

    def __init__(
        self,
        provider: Any,
        *,
        model: str,
        temperature: float = 0.2,
        messages: list[Any] | None = None,
        num_ctx: int | None = None,
        result_limit: int = RESULT_LIMIT,
    ) -> None:
        from kiki.ai.provider import ProviderError as AiProviderError

        self._ai_provider_error = AiProviderError
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._working = list(messages or [])
        self._num_ctx = num_ctx
        self._result_limit = result_limit
        self._folded = 0
        self.last_provider_message: str = ""

    async def next_action_stream(
        self,
        *,
        user_text: str,
        tool_schemas: list[dict[str, Any]],
        observations: list[ToolResult],
        cancel_token: CancelToken,
    ) -> AsyncIterator[StepEvent]:
        del user_text  # the conversation is complete already; this echo adds nothing
        for result in observations[self._folded :]:
            self._working.append(
                _tool_message(result, limit=self._result_limit)
            )
        self._folded = len(observations)
        tools = [_declaration(schema) for schema in tool_schemas]
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        stream = self._provider.stream_chat_tools(
            self._working,
            model=self._model,
            temperature=self._temperature,
            tools=tools,
            num_ctx=self._num_ctx,
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
                    yield StepEvent(kind="delta", text=text)
                for raw in getattr(chunk, "tool_calls", ()) or ():
                    calls.append(_to_call(raw))
        except Exception as exc:
            if isinstance(exc, self._ai_provider_error):
                self.last_provider_message = str(exc)
            log.warning("chat provider step failed: %s", type(exc).__name__)
            raise ProviderError from exc
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    log.debug("closing the provider stream failed", exc_info=True)

        if cancel_token.cancelled:
            yield StepEvent(kind="action", action=ModelAction.answer("abgebrochen"))
            return
        action = _collapse(text_parts, calls)
        call = action.tool_call
        if action.kind is ActionKind.TOOL_CALL and call is not None and call.name:
            # The protocol shape the next request needs: the call as an
            # assistant message, its answer as the tool messages appended on
            # the next entry. Preamble text travels with the call, exactly as
            # the loop before it did.
            self._working.append(
                _assistant_tool_message("".join(text_parts).strip(), call)
            )
        yield StepEvent(kind="action", action=action)


def _tool_message(result: ToolResult, *, limit: int) -> Any:
    from kiki.ai.provider import ChatMessage

    return ChatMessage(
        role="tool",
        content=_observation_text(result, limit=limit),
        tool_name=result.name,
        tool_call_id=result.call_id,
    )


def _assistant_tool_message(preamble: str, call: ToolCall) -> Any:
    from kiki.ai.provider import ChatMessage
    from kiki.ai.provider import ToolCall as ProviderToolCall

    return ChatMessage(
        role="assistant",
        content=preamble,
        tool_calls=(
            ProviderToolCall(id=call.id, name=call.name, arguments=dict(call.arguments)),
        ),
    )


def _observation_text(result: ToolResult, *, limit: int) -> str:
    """What a tool result looks like to the model. Data, or a bare category."""
    if result.ok:
        return _serialize(result.data or {}, limit=limit)
    return _serialize({"error": result.error_code or "tool_failed"}, limit=limit)


def _serialize(payload: Any, *, limit: int) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(payload)
    if len(text) > limit:
        return text[:limit] + f"… [gekürzt, {len(text)} Zeichen gesamt]"
    return text


def _collapse(text_parts: list[str], calls: list[ToolCall]) -> ModelAction:
    """The stream's one decision. A malformed one stays malformed on purpose."""
    if len(calls) > 1:
        return ModelAction(ActionKind.TOOL_CALL)  # refused by validate_action
    if calls:
        call = calls[0]
        if not call.name or not isinstance(call.arguments, dict):
            return ModelAction(ActionKind.TOOL_CALL)
        # A tool call wins: whatever text came with it is a preamble, and
        # showing it as the answer would claim the work is done.
        return ModelAction(ActionKind.TOOL_CALL, tool_call=call)
    return ModelAction.answer("".join(text_parts).strip())


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
