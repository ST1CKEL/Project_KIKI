"""Chat orchestration: history, streaming, character states. No GTK."""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiki.ai.factory import active_model, create_provider
from kiki.ai.prompts import (
    DEFAULT_IMAGE_PROMPT,
    MEMORY_LIMIT,
    attach_status_block,
    build_messages,
)
from kiki.ai.provider import ChatMessage, LLMProvider, ProviderError, ToolCapableProvider
from kiki.assistant.adapter import ChatStepAdapter
from kiki.assistant.run_service import failure_text
from kiki.assistant.runner import AssistantRunner
from kiki.config.settings import Settings
from kiki.context.planner import ContextPlanner, PlannedContext
from kiki.harness.confirmation import ConfirmationRequest
from kiki.harness.models import RunStatus
from kiki.paths import state_dir
from kiki.runtime.event_bus import EventBus
from kiki.storage.chat_repository import ChatRepository, Conversation
from kiki.storage.memory_repository import MemoryRepository
from kiki.storage.secrets import SecretStore
from kiki.tools.direct_actions import DirectActionService, DirectLaunchRequest
from kiki.tools.executor import ToolExecutor, ToolResult
from kiki.tools.exposure import declarations
from kiki.tools.gateway import ToolGateway

log = logging.getLogger(__name__)

# The dialog callback for the agent path: it receives the card the runner
# armed -- the display record with the request id -- and answers with a
# verdict. It never computes an authorisation of its own.
AssistantConfirm = Callable[[ConfirmationRequest], Awaitable[bool] | bool]

CANCEL_TEXT = "Der Durchlauf wurde abgebrochen."


@dataclass(frozen=True)
class StreamEvent:
    kind: str  # delta, tool_start, tool_end, error, done
    text: str = ""
    tool: str = ""
    ok: bool = True


def _with_tool_note(answer: str, used_tools: list[str]) -> str:
    """Record which tools produced an answer, so the transcript stays auditable."""
    if not used_tools:
        return answer
    seen: list[str] = []
    for name in used_tools:
        if name not in seen:
            seen.append(name)
    return f"{answer}\n\n_[KIKI hat benutzt: {', '.join(seen)}]_"


class ChatService:
    def __init__(
        self,
        settings: Settings,
        chats: ChatRepository,
        secrets: SecretStore,
        bus: EventBus,
        executor: ToolExecutor | None = None,
        confirm: AssistantConfirm | None = None,
        memories: MemoryRepository | None = None,
        trace_dir: Path | None = None,
        direct_actions: DirectActionService | None = None,
    ) -> None:
        self._settings = settings
        self._chats = chats
        self._secrets = secrets
        self._bus = bus
        self._executor = executor
        self._confirm = confirm
        self._memories = memories
        self._trace_dir = trace_dir
        self._direct_actions = direct_actions
        self._planner = ContextPlanner()
        # Diagnostics only. The plan a request is built from is passed down the
        # call chain; reading this back after concurrent turns reports whichever
        # finished last, so it must never be used to build a request.
        self._last_plan: PlannedContext | None = None
        self._provider: LLMProvider | None = None

    def update_settings(self, settings: Settings) -> None:
        self._settings = settings
        self._provider = None
        if self._executor is not None:
            self._executor.policy.set_autonomy(self._settings.tools.autonomy)

    def set_confirm(self, confirm: AssistantConfirm | None) -> None:
        self._confirm = confirm

    def _active_memories(self) -> list[Any]:
        """Memories for this turn, or nothing while panic or integrations are off.

        Under panic the user wants nothing local reaching a model, and with a
        cloud provider configured the memory block would be exactly that.
        """
        if self._memories is None or not self._settings.integrations_active():
            return []
        try:
            return list(self._memories.list(limit=MEMORY_LIMIT))
        except Exception:
            log.exception("could not read memories")
            return []

    def tools_active(self) -> bool:
        """True when the model may request tools on this turn."""
        return (
            self._settings.tools.model_tool_use
            and self._executor is not None
            and not self._settings.app.privacy_panic
            and isinstance(self.provider(), ToolCapableProvider)
        )

    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = create_provider(self._settings, self._secrets)
        return self._provider

    def ensure_conversation(self, conversation_id: str | None) -> Conversation:
        if conversation_id:
            found = self._chats.get_conversation(conversation_id)
            if found:
                return found
        return self._chats.create_conversation()

    def direct_action(self, text: str) -> DirectLaunchRequest | None:
        """Return a bounded local action only when the input is an exact command."""
        if self._direct_actions is None:
            return None
        return self._direct_actions.parse(text)

    async def send(
        self,
        conversation_id: str,
        user_text: str,
        *,
        status_snapshot: dict[str, Any] | None = None,
        images: tuple[str, ...] = (),
        image_names: tuple[str, ...] = (),
    ) -> AsyncIterator[StreamEvent]:
        text = user_text.strip()
        if not text and images:
            text = DEFAULT_IMAGE_PROMPT
        if not text:
            yield StreamEvent(kind="error", text="Leere Nachricht.")
            return
        if status_snapshot:
            text = attach_status_block(text, status_snapshot)
        stored = text
        if image_names:
            listed = ", ".join(image_names)
            stored = f"{text}\n\n_[Bild angehängt: {listed}]_"
        history = self._chats.history(conversation_id)
        self._chats.add_message(conversation_id, "user", stored)
        if history == []:
            title = user_text.strip().split("\n", 1)[0][:48] if user_text.strip() else "Bildfrage"
            self._chats.rename_conversation(conversation_id, title or "Chat")
        direct = self.direct_action(user_text) if not images and not status_snapshot else None
        if direct is not None:
            async for event in self._run_direct(conversation_id, direct):
                yield event
            return
        # Size the context to what this turn actually needs. A greeting must
        # not pay the prefill of a code review: measured on qwen3-vl:4b, 8.6k
        # prompt tokens cost 2.75 s before the first word, 3.0k cost 1.10 s.
        plan = self._planner.plan(
            user_text=text,
            system_prompt=self._settings.compose_prompt(),
            history=history,
            memories=self._active_memories(),
        )
        self._last_plan = plan
        log.info("context plan %s", plan.summary())
        self._bus.emit(
            "chat.context.planned",
            conversation_id=conversation_id,
            intent=plan.intent.value,
            tokens=plan.used_tokens,
            num_ctx=plan.num_ctx,
        )
        messages = build_messages(
            system_prompt=plan.system_prompt,
            history=plan.history,
            user_text=text,
            history_limit=len(plan.history) or 1,
            images=images,
            memories=plan.memories,
        )
        self._bus.emit("chat.stream.start", conversation_id=conversation_id)
        chunks: list[str] = []
        speaking = False
        used_tools: list[str] = []
        try:
            model = active_model(self._settings)
            stream = (
                self._run_agent(messages, model=model, plan=plan, user_text=text)
                if self.tools_active()
                else self._run_plain(messages, model=model, plan=plan)
            )
            async for event in stream:
                if event.kind == "delta":
                    if not speaking:
                        speaking = True
                        self._bus.emit("chat.stream.speaking", conversation_id=conversation_id)
                    chunks.append(event.text)
                    self._bus.emit(
                        "chat.stream.delta", conversation_id=conversation_id, text=event.text
                    )
                    yield event
                elif event.kind in {"tool_start", "tool_end"}:
                    if event.kind == "tool_end" and event.ok and event.tool:
                        used_tools.append(event.tool)
                    self._bus.emit(
                        f"chat.stream.{event.kind}",
                        conversation_id=conversation_id,
                        tool=event.tool,
                        text=event.text,
                        ok=event.ok,
                    )
                    yield event
                elif event.kind == "error":
                    raise ProviderError(event.text)
        except ProviderError as exc:
            self._bus.emit("chat.stream.error", conversation_id=conversation_id, error=str(exc))
            yield StreamEvent(kind="error", text=str(exc))
            self._chats.add_message(conversation_id, "assistant", f"**Fehler:** {exc}")
            self._bus.emit("chat.stream.done", conversation_id=conversation_id, ok=False)
            return
        except Exception as exc:
            log.exception("chat stream crashed")
            self._bus.emit("chat.stream.error", conversation_id=conversation_id, error=str(exc))
            yield StreamEvent(kind="error", text=f"Unerwarteter Fehler: {exc}")
            self._bus.emit("chat.stream.done", conversation_id=conversation_id, ok=False)
            return
        answer = "".join(chunks).strip()
        if answer:
            self._chats.add_message(conversation_id, "assistant", _with_tool_note(answer, used_tools))
        else:
            self._chats.add_message(conversation_id, "assistant", "_Leere Antwort vom Modell._")
        self._bus.emit("chat.stream.done", conversation_id=conversation_id, ok=True, text=answer)
        yield StreamEvent(kind="done", text=answer)

    async def _run_direct(
        self,
        conversation_id: str,
        request: DirectLaunchRequest,
    ) -> AsyncIterator[StreamEvent]:
        """Execute a locally resolved user command without consulting a model."""
        assert self._direct_actions is not None
        self._bus.emit("chat.stream.start", conversation_id=conversation_id)
        try:
            result = await self._direct_actions.execute(request)
        except Exception as exc:
            log.warning("direct local action failed: %s", type(exc).__name__)
            answer = "Der lokale Start ist unerwartet fehlgeschlagen."
            self._chats.add_message(conversation_id, "assistant", answer)
            self._bus.emit("chat.stream.error", conversation_id=conversation_id, error=answer)
            self._bus.emit("chat.stream.done", conversation_id=conversation_id, ok=False)
            yield StreamEvent(kind="error", text=answer)
            return
        self._bus.emit("chat.stream.speaking", conversation_id=conversation_id)
        self._bus.emit("chat.stream.delta", conversation_id=conversation_id, text=result.answer)
        yield StreamEvent(kind="delta", text=result.answer)
        stored = _with_tool_note(result.answer, [result.tool] if result.ok and result.tool else [])
        self._chats.add_message(conversation_id, "assistant", stored)
        self._bus.emit(
            "chat.stream.done",
            conversation_id=conversation_id,
            ok=True,
            text=result.answer,
        )
        yield StreamEvent(kind="done", text=result.answer)

    async def _run_plain(
        self, messages: list[ChatMessage], *, model: str, plan: PlannedContext | None = None
    ) -> AsyncIterator[StreamEvent]:
        window = plan.num_ctx if plan else None
        async for delta in self.provider().stream_chat(
            messages,
            model=model,
            temperature=self._settings.ai.temperature,
            num_ctx=window,
        ):
            yield StreamEvent(kind="delta", text=delta)

    async def _run_agent(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        plan: PlannedContext | None = None,
        user_text: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """One chat turn on the unified runner.

        The same `AssistantRunner` the agent path uses, with a step adapter
        that owns this turn's planned conversation. What the turn loses is the
        old loop's private ideas: refusals now come back as categories, limits
        end the turn visibly instead of fading into a half answer, and the
        run is traced like any other run -- length and shapes, never content.
        """
        assert self._executor is not None
        executor = self._executor
        tools = declarations(
            executor.registry,
            executor.policy,
            panic=self._settings.app.privacy_panic,
            integrations_enabled=self._settings.integrations_active(),
        )
        if not tools:
            async for event in self._run_plain(messages, model=model, plan=plan):
                yield event
            return
        # Built per turn from live sources: the panic switch must take effect
        # mid-turn, including after an approval card was already answered.
        gateway = ToolGateway(
            executor,
            panic_check=lambda: self._settings.app.privacy_panic,
            integrations_check=self._settings.integrations_active,
        )
        adapter = ChatStepAdapter(
            self.provider(),
            model=model,
            temperature=self._settings.ai.temperature,
            messages=messages,
            num_ctx=plan.num_ctx if plan else None,
        )
        runner = AssistantRunner(
            adapter,
            gateway,
            trace_dir=self._trace_dir or (state_dir() / "assistant"),
            max_steps=self._settings.tools.max_steps,
            max_tool_calls=self._settings.tools.max_tool_calls,
        )
        run = runner.begin(user_text)
        async for event in runner.drive(run):
            if event.kind == "delta":
                yield StreamEvent(kind="delta", text=event.text)
            elif event.kind == "tool_start":
                yield StreamEvent(kind="tool_start", tool=event.tool, text=event.text)
            elif event.kind == "tool_end":
                yield StreamEvent(kind="tool_end", tool=event.tool, text=event.text, ok=event.ok)
            elif event.kind == "confirmation_requested":
                await self._settle_confirmation(runner, event.request)
            elif event.kind == "finished":
                # The deltas already streamed; a completed turn needs nothing
                # more. Everything else is a turn that did not end in an
                # answer, and the chat must say so instead of storing silence.
                if run.status is not RunStatus.COMPLETED:
                    yield StreamEvent(kind="error", text=self._failure_line(run, adapter))
                return

    async def _settle_confirmation(
        self,
        runner: AssistantRunner,
        request: ConfirmationRequest | None,
    ) -> None:
        """Take the card to the dialog while the runner holds the question.

        The answer travels back the only way it may: the request id the card
        was armed with, spent through the runner, so the broker behind the
        gateway mints the grant -- or refuses it if anything moved.
        """
        if request is None:
            return
        allowed: object = False
        if self._confirm is not None:
            allowed = self._confirm(request)
            if inspect.isawaitable(allowed):
                allowed = await allowed
        if allowed:
            runner.confirm(request.run_id, request.call_id, request.request_id)
        else:
            runner.reject(request.run_id, request.call_id)

    def _failure_line(self, run: Any, adapter: ChatStepAdapter) -> str:
        if run.error_code == "provider_error" and adapter.last_provider_message:
            # The provider's own sentence, shown exactly as the plain path
            # shows it. It never passed through the runner.
            return adapter.last_provider_message
        if run.status is RunStatus.CANCELLED:
            return CANCEL_TEXT
        return failure_text(run.error_code)

    async def collect_status(self, names: list[str] | None = None) -> dict[str, Any]:
        """Run read-only status tools. Never invoked implicitly by the model."""
        if self._executor is None:
            return {"error": "Kein Tool-Executor."}
        if not self._settings.integrations_active():
            return {"disabled": True, "reason": "privacy_panic_or_integrations_off"}
        wanted = names or [
            "status_datetime",
            "status_upower",
            "status_networkmanager",
            "status_disk",
        ]
        out: dict[str, Any] = {}
        for name in wanted:
            result: ToolResult = await self._executor.run(
                name,
                {},
                panic=self._settings.app.privacy_panic,
                integrations_enabled=self._settings.integrations_active(),
            )
            out[name] = result.data if result.ok else {"error": result.error}
        return out
