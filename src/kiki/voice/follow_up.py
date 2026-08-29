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
    requested: bool = False
    terminal: bool = False
    delivered: bool = False

    def begin(self, *, enabled: bool) -> None:
        self.requested = bool(enabled)
        self.terminal = False
        self.delivered = False

    def cancel(self) -> None:
        self.requested = False
        self.terminal = False
        self.delivered = False

    def mark_terminal(self, *, cancelled: bool = False) -> None:
        if cancelled:
            self.cancel()
        elif self.requested:
            self.terminal = True

    def mark_response_delivered(self) -> bool:
        """Record the final response, not an intermediate spoken prompt."""
        if not self.requested or not self.terminal:
            return False
        self.delivered = True
        return True

    def consume_ready(self) -> bool:
        ready = self.requested and self.terminal and self.delivered
        if ready:
            self.cancel()
        return ready
