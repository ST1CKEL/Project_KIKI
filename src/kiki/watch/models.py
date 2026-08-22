"""What a watcher may say. Watchers report; they never act.

A `Notice` carries text and nothing else — no callback, no tool, no parameters.
That is the whole safety story for proactivity: KIKI gains the ability to speak
up on her own, but not the ability to *do* anything on her own. Everything that
changes the system still goes through the approval card.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    URGENT = "urgent"


# Only these reach the speakers. Info is quiet by design: a spoken sentence for
# every minor observation turns the assistant into a nuisance.
SPOKEN_SEVERITIES: frozenset[Severity] = frozenset({Severity.WARNING, Severity.URGENT})


@dataclass(frozen=True)
class Notice:
    """One thing worth telling the user about."""

    # Stable across polls for the same condition, so repeats can be suppressed.
    key: str
    watcher: str
    title: str
    # A short, plain sentence. No markdown, no numbers-with-units salad.
    spoken: str
    detail: str = ""
    severity: Severity = Severity.INFO

    def __post_init__(self) -> None:
        if not self.key or not self.title or not self.spoken:
            raise ValueError("notice needs key, title and spoken text")
