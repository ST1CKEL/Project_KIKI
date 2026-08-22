"""Pick an intent, then fill its budget from the memory layers.

The layering follows the four levels KIKI keeps:

  L1  the active turn      — user input, session state; always present
  L2  short-term           — recent messages and a rolling summary
  L3  long-term            — explicitly stored facts, retrieved by relevance
  L4  artefacts            — diffs, logs, files; referenced, never prompted whole

Filling order is L1, then L3, then L2, and that order is deliberate: a handful
of relevant facts is worth more than four more lines of small talk, so history
is what gets truncated when the budget runs out.

No LLM is involved. Classification is a keyword heuristic and retrieval is
SQLite work, matching the rule that memory retrieval costs zero model tokens.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from kiki.context.budget import Budget, Intent, budget_for, estimate_tokens, num_ctx_for

log = logging.getLogger(__name__)

_GREETING = re.compile(
    r"\b(hallo|hi|hey|moin|servus|guten (morgen|tag|abend)|wie geht|na\b|danke|tschüss|bis später)\b",
    re.IGNORECASE,
)
_REVIEW = re.compile(
    r"\b(review|diff|prüf|teste?n?\b|test(lauf|s)?\b|commit|pull request|abnahme)\b",
    re.IGNORECASE,
)
_CODING = re.compile(
    r"\b(code|coding|repo|repository|workspace|branch|refactor|implementier|"
    r"bug|fehler im code|funktion|klasse|modul|plan)\b",
    re.IGNORECASE,
)

# Below this many characters and with no substance markers, a turn is chatter.
_SMALLTALK_CHARS = 60


def classify(
    user_text: str,
    *,
    coding_session_active: bool = False,
    background: bool = False,
) -> Intent:
    """Cheap intent guess. Session state outranks the wording."""
    if background:
        return Intent.BACKGROUND
    text = (user_text or "").strip()
    if coding_session_active:
        return Intent.CODING_REVIEW if _REVIEW.search(text) else Intent.CODING_PLAN
    if _REVIEW.search(text) or _CODING.search(text):
        return Intent.CODING_PLAN
    if len(text) <= _SMALLTALK_CHARS and (_GREETING.search(text) or "?" not in text):
        return Intent.SMALLTALK
    return Intent.ADVICE


@dataclass
class PlannedContext:
    """What the assembler decided to spend this turn on."""

    intent: Intent
    budget: Budget
    system_prompt: str
    memories: list[Any] = field(default_factory=list)
    history: list[Any] = field(default_factory=list)
    used_tokens: int = 0
    # Counted against everything that was *offered*, not against the already
    # capped window — otherwise the number reads as "nothing was left out" on a
    # turn that dropped a 200-message history down to eight.
    dropped_history: int = 0
    dropped_memories: int = 0

    @property
    def num_ctx(self) -> int:
        return num_ctx_for(self.budget)

    def summary(self) -> str:
        return (
            f"{self.intent.value}: {self.used_tokens}/{self.budget.max_tokens} Tokens, "
            f"{len(self.history)} Nachrichten, {len(self.memories)} Erinnerungen"
            + (f", {self.dropped_history} Nachrichten gekürzt" if self.dropped_history else "")
        )


class ContextPlanner:
    """Assembles one turn's context inside a budget."""

    def __init__(self, *, reserve_for_answer: int = 512) -> None:
        self._reserve = max(0, reserve_for_answer)

    def plan(
        self,
        *,
        user_text: str,
        system_prompt: str,
        history: list[Any],
        memories: list[Any],
        coding_session_active: bool = False,
        background: bool = False,
    ) -> PlannedContext:
        intent = classify(
            user_text,
            coding_session_active=coding_session_active,
            background=background,
        )
        budget = budget_for(intent)
        limit = max(0, budget.max_tokens - self._reserve)

        # L1: the turn itself and the rules are not negotiable.
        used = estimate_tokens(system_prompt) + estimate_tokens(user_text)

        # L3: relevance beats recency, so memories are placed before history.
        kept_memories: list[Any] = []
        offered = list(memories)[: budget.max_memories]
        for item in offered:
            cost = estimate_tokens(str(getattr(item, "content", "")))
            if used + cost > limit:
                break
            kept_memories.append(item)
            used += cost

        # L2: recent messages, newest first, until the budget is gone.
        eligible = [m for m in history if getattr(m, "role", "") in {"user", "assistant"}]
        recent = eligible[-budget.history_messages :] if budget.history_messages else []
        kept_reversed: list[Any] = []
        for message in reversed(recent):
            cost = estimate_tokens(str(getattr(message, "content", "")))
            if used + cost > limit:
                break
            kept_reversed.append(message)
            used += cost
        kept_history = list(reversed(kept_reversed))

        return PlannedContext(
            intent=intent,
            budget=budget,
            system_prompt=system_prompt,
            memories=kept_memories,
            history=kept_history,
            used_tokens=used,
            dropped_history=len(eligible) - len(kept_history),
            dropped_memories=max(0, len(memories) - len(kept_memories)),
        )
