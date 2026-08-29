"""Lifecycle for one wake-word conversation turn.

The application receives completion, answer-delivery and TTS-idle signals on
different paths.  This tiny gate makes their order explicit: follow-up may open
only after a wake-word turn reached a terminal answer and that answer was
actually delivered.  Intermediate confirmation prompts never satisfy the
gate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FollowUpTurn:
    active: bool = False
    requested: bool = False
    terminal: bool = False
    delivered: bool = False

    def begin(self, *, enabled: bool) -> None:
        self.active = True
        self.requested = bool(enabled)
        self.terminal = False
        self.delivered = False

    def cancel(self) -> None:
        self.active = False
        self.requested = False
        self.terminal = False
        self.delivered = False

    def mark_terminal(self, *, cancelled: bool = False) -> None:
        if cancelled:
            self.cancel()
        elif self.active:
            self.terminal = True

    def mark_response_delivered(self) -> bool:
        """Record the final response, not an intermediate spoken prompt."""
        if not self.active or not self.terminal:
            return False
        self.delivered = True
        return True

    def consume_ready(self) -> bool:
        settled = self.active and self.terminal and self.delivered
        ready = settled and self.requested
        if settled:
            self.cancel()
        return ready
