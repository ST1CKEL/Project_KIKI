"""How many tokens a turn may spend, and on what.

Context is the dominant latency cost. Measured on an RTX 5060 Ti with
qwen3-vl:4b, cache-busted so no prefix is reused:

    prompt tokens      prefill    time to first token
               61       0.02 s                 0.44 s
            3 009       0.59 s                 1.10 s
            8 587       1.95 s                 2.75 s
           23 652       8.94 s                10.41 s
           32 758      15.92 s                28.88 s

That is roughly 4 400 tokens/s of prefill, so every extra 1 000 tokens costs
about a quarter second before KIKI says a word. A fixed large window therefore
makes every greeting pay for a context only a code review needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# Calibrated against Ollama's own prompt_eval_count for German text, code and
# mixed content: 2.83–3.95 characters per token. Three is deliberately below
# the observed range so the estimate errs towards *over*-counting and the
# assembler stays inside its budget rather than overshooting it.
CHARS_PER_TOKEN = 3.0


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN) + 1


class Intent(StrEnum):
    SMALLTALK = "smalltalk"
    ADVICE = "advice"
    CODING_PLAN = "coding_plan"
    CODING_REVIEW = "coding_review"
    BACKGROUND = "background"


class Priority(StrEnum):
    """Who gets the runtime when several workers want it at once."""

    HIGH = "high"
    EXCLUSIVE = "exclusive"
    LOW = "low"


@dataclass(frozen=True)
class Budget:
    intent: Intent
    max_tokens: int
    # L2: how many recent messages may be replayed.
    history_messages: int
    # L3: how many stored memories may be retrieved.
    max_memories: int
    priority: Priority
    # L4 artefacts (diffs, logs, files) are referenced, never prompted whole.
    allow_artifacts: bool = False


BUDGETS: dict[Intent, Budget] = {
    Intent.SMALLTALK: Budget(
        intent=Intent.SMALLTALK,
        max_tokens=2_000,
        history_messages=8,
        max_memories=1,
        priority=Priority.HIGH,
    ),
    Intent.ADVICE: Budget(
        intent=Intent.ADVICE,
        max_tokens=6_000,
        history_messages=20,
        max_memories=6,
        priority=Priority.HIGH,
    ),
    Intent.CODING_PLAN: Budget(
        intent=Intent.CODING_PLAN,
        max_tokens=12_000,
        history_messages=20,
        max_memories=6,
        priority=Priority.HIGH,
        allow_artifacts=True,
    ),
    Intent.CODING_REVIEW: Budget(
        intent=Intent.CODING_REVIEW,
        max_tokens=16_000,
        history_messages=12,
        max_memories=4,
        priority=Priority.EXCLUSIVE,
        allow_artifacts=True,
    ),
    Intent.BACKGROUND: Budget(
        intent=Intent.BACKGROUND,
        max_tokens=2_000,
        history_messages=0,
        max_memories=0,
        priority=Priority.LOW,
    ),
}


def budget_for(intent: Intent) -> Budget:
    return BUDGETS.get(intent, BUDGETS[Intent.ADVICE])


def num_ctx_for(budget: Budget, *, floor: int = 2048, ceiling: int = 32768) -> int:
    """Round the budget up to a sane window, with room for the answer.

    The window has to hold the prompt *and* what KIKI generates, so this adds
    headroom rather than handing Ollama exactly the prompt size.
    """
    wanted = budget.max_tokens + 1024
    size = floor
    while size < wanted and size < ceiling:
        size *= 2
    return min(max(size, floor), ceiling)
