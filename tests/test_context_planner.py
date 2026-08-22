"""Dynamic context budget: intent classification and budget-bounded assembly."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from kiki.context.budget import Intent, Priority, budget_for, estimate_tokens, num_ctx_for
from kiki.context.planner import ContextPlanner, classify


@dataclass
class Msg:
    role: str
    content: str


@dataclass
class Mem:
    content: str
    kind: str = "fact"


# --- token estimation -------------------------------------------------------


def test_estimate_is_calibrated_and_conservative() -> None:
    """Measured German text runs 2.83–3.95 chars/token; we assume 3.0.

    Assuming fewer characters per token over-counts, which keeps the assembler
    inside its budget instead of overshooting it.
    """
    german = "Der Nutzer arbeitet an einem Linux-Projekt und mag praezise Antworten. " * 8
    assert estimate_tokens(german) >= len(german) / 3.95
    assert estimate_tokens("") == 0


# --- classification ---------------------------------------------------------


@pytest.mark.parametrize("text", ["Hallo", "Hi KIKI", "Guten Morgen", "Danke!", "Moin"])
def test_greetings_are_smalltalk(text) -> None:
    assert classify(text) is Intent.SMALLTALK


@pytest.mark.parametrize(
    "text",
    [
        "Wie richte ich einen Reverse Proxy mit nginx ein und worauf muss ich achten?",
        "Erklär mir bitte den Unterschied zwischen systemd targets und runlevels.",
    ],
)
def test_substantial_questions_are_advice(text) -> None:
    assert classify(text) is Intent.ADVICE


def test_coding_words_raise_the_budget() -> None:
    assert classify("Refactor bitte das Modul für die Workspaces") is Intent.CODING_PLAN


def test_an_active_session_outranks_the_wording() -> None:
    """Inside a coding session even a greeting gets the coding budget."""
    assert classify("Hallo", coding_session_active=True) is Intent.CODING_PLAN
    assert classify("Schau dir den Diff an", coding_session_active=True) is Intent.CODING_REVIEW


def test_background_work_is_always_cheap_and_low_priority() -> None:
    assert classify("Fasse das zusammen", background=True) is Intent.BACKGROUND
    assert budget_for(Intent.BACKGROUND).priority is Priority.LOW
    assert budget_for(Intent.BACKGROUND).max_tokens <= 2000


def test_review_claims_the_runtime_exclusively() -> None:
    assert budget_for(Intent.CODING_REVIEW).priority is Priority.EXCLUSIVE


def test_budgets_grow_with_the_task() -> None:
    order = [Intent.SMALLTALK, Intent.ADVICE, Intent.CODING_PLAN, Intent.CODING_REVIEW]
    sizes = [budget_for(i).max_tokens for i in order]
    assert sizes == sorted(sizes)
    assert sizes[0] <= 2000 and sizes[-1] >= 16000


def test_window_leaves_room_for_the_answer() -> None:
    for intent in Intent:
        b = budget_for(intent)
        assert num_ctx_for(b) > b.max_tokens
    assert num_ctx_for(budget_for(Intent.SMALLTALK)) == 4096
    assert num_ctx_for(budget_for(Intent.CODING_REVIEW)) == 32768


# --- assembly ---------------------------------------------------------------


def _planner() -> ContextPlanner:
    return ContextPlanner(reserve_for_answer=0)


def test_smalltalk_stays_tiny_even_with_a_long_history() -> None:
    history = [Msg("user" if i % 2 == 0 else "assistant", f"Nachricht {i} " * 40) for i in range(200)]
    memories = [Mem(f"Fakt {i}") for i in range(50)]

    plan = _planner().plan(
        user_text="Hallo",
        system_prompt="Du bist KIKI.",
        history=history,
        memories=memories,
    )
    assert plan.intent is Intent.SMALLTALK
    assert plan.used_tokens <= plan.budget.max_tokens
    assert len(plan.memories) <= 1
    # 200 messages down to at most the smalltalk window of 8.
    assert len(plan.history) <= 8
    assert plan.dropped_history >= 192
    assert plan.dropped_memories >= 49


def test_a_long_question_gets_more_room_than_a_greeting() -> None:
    history = [Msg("user" if i % 2 == 0 else "assistant", f"Zeile {i} " * 30) for i in range(100)]
    memories = [Mem(f"Fakt {i}") for i in range(20)]
    p = _planner()

    small = p.plan(user_text="Hallo", system_prompt="S", history=history, memories=memories)
    advice = p.plan(
        user_text="Wie konfiguriere ich nginx als Reverse Proxy vor mehreren Diensten?",
        system_prompt="S",
        history=history,
        memories=memories,
    )
    assert advice.used_tokens > small.used_tokens
    assert len(advice.history) > len(small.history)
    assert len(advice.memories) > len(small.memories)


def test_memories_are_kept_when_history_must_go() -> None:
    """Relevance beats recency: history is what gets truncated."""
    history = [Msg("user", "x" * 4000) for _ in range(20)]
    memories = [Mem("Nutzt Fedora 44."), Mem("Mag kurze Antworten.")]

    plan = _planner().plan(
        user_text="Wie konfiguriere ich meinen Reverse Proxy richtig?",
        system_prompt="S",
        history=history,
        memories=memories,
    )
    assert len(plan.memories) == 2
    assert plan.dropped_history > 0
    assert plan.used_tokens <= plan.budget.max_tokens


def test_the_budget_is_never_exceeded() -> None:
    huge = [Msg("user", "y" * 100_000)]
    huge_mem = [Mem("z" * 100_000)]
    for text, active in (("Hallo", False), ("Erklär mir Kubernetes im Detail", False), ("Diff", True)):
        plan = _planner().plan(
            user_text=text, system_prompt="S" * 500, history=huge, memories=huge_mem
        )
        assert plan.used_tokens <= plan.budget.max_tokens, plan.intent


def test_nothing_to_work_with_is_fine() -> None:
    plan = _planner().plan(user_text="Hallo", system_prompt="S", history=[], memories=[])
    assert plan.history == [] and plan.memories == []
    assert plan.used_tokens > 0


def test_reserve_keeps_headroom_for_the_reply() -> None:
    history = [Msg("user", "w" * 2000) for _ in range(30)]
    plan = ContextPlanner(reserve_for_answer=512).plan(
        user_text="Erklär mir bitte ausführlich, wie DNS funktioniert.",
        system_prompt="S",
        history=history,
        memories=[],
    )
    assert plan.used_tokens <= plan.budget.max_tokens - 512


def test_summary_reports_what_was_spent() -> None:
    plan = _planner().plan(
        user_text="Hallo", system_prompt="S", history=[Msg("user", "a" * 9000)], memories=[]
    )
    text = plan.summary()
    assert "smalltalk" in text and "Tokens" in text


def test_concurrent_turns_do_not_share_a_plan() -> None:
    """Three workers in flight each get their own context window.

    The plan is passed down the call chain rather than kept on the service, so
    it cannot be shared between turns. Reading it back from the instance
    afterwards reports only the last turn — which is why `_last_plan` is for
    diagnostics and never for building a request.
    """
    import asyncio

    from kiki.ai.chat_service import ChatService
    from kiki.config.settings import load_settings
    from kiki.runtime.event_bus import EventBus
    from kiki.storage.chat_repository import ChatRepository
    from kiki.storage.database import Database
    from kiki.storage.secrets import MemorySecretStore

    class SlowProvider:
        id = "slow"

        def __init__(self) -> None:
            self.windows: list[int | None] = []

        async def stream_chat(self, messages, *, model, temperature=0.7, num_ctx=None):
            # Yield control so the other turns interleave before this one reads.
            await asyncio.sleep(0.01)
            self.windows.append(num_ctx)
            yield "ok"

    import tempfile
    from pathlib import Path

    db = Database(Path(tempfile.mkdtemp()) / "k.sqlite3")
    settings = load_settings(Path("/nonexistent.toml"))
    provider = SlowProvider()
    service = ChatService(settings, ChatRepository(db), MemorySecretStore(), EventBus())
    service._provider = provider

    async def turn(text):
        conv = service.ensure_conversation(None)
        async for _ in service.send(conv.id, text):
            pass

    async def all_three():
        # gather() must be built inside the loop, not before asyncio.run().
        await asyncio.wait_for(
            asyncio.gather(
                turn("Hallo"),
                turn("Erklär mir bitte ausführlich, wie ein Reverse Proxy funktioniert."),
                turn("Hi"),
            ),
            timeout=20,
        )

    asyncio.run(all_three())

    # Two greetings at the smalltalk window, one question at the advice window.
    assert sorted(provider.windows) == [4096, 4096, 8192], provider.windows
