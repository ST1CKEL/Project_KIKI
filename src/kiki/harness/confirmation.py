"""What a person is shown before a confirmable action, and nothing more.

This module used to mint its own authorisation: a sixteen-character fingerprint
over the run, the call and the proposed content, held in a one-shot slot, with
the UI handing the fingerprint back to spend it. That was a second confirmation
system beside `kiki.tools.confirmation`, with a weaker binding and a dialog that
computed its own authorisation value.

It mints nothing now. `ConfirmationBroker` in `kiki.tools.confirmation` is the
only thing that issues and redeems a grant, over the full SHA-256 binding that
also covers the validated arguments and the tool spec. What is left here is the
display record the harness hands to the UI, and it carries a `request_id` --
the *name* of the question, which the UI answers with. A name is not an
authorisation: presenting it proves nothing, and the broker still checks the run,
the call, the arguments and the preview before anything runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from kiki.tools.confirmation import ConfirmationError
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ActionPreview

__all__ = ["ConfirmationError", "ConfirmationRequest"]


@dataclass(frozen=True)
class ConfirmationRequest:
    """What the UI must show, and nothing the UI may change.

    `content` is the human-readable body. It holds the workspace-relative target
    and the full content, because a user cannot approve what they are not shown
    -- but never an absolute path, so nothing about the machine leaks into a
    dialog or a screenshot of one.
    """

    run_id: str
    call_id: str
    tool_name: str
    title: str
    target: str
    content: str
    request_id: str = ""
    risk: RiskLevel = RiskLevel.WRITE

    @classmethod
    def from_preview(
        cls, run_id: str, call_id: str, preview: ActionPreview
    ) -> ConfirmationRequest:
        """Built from the card the registry produced, not from anything the model said."""
        return cls(
            run_id=run_id,
            call_id=call_id,
            tool_name=preview.tool,
            title=preview.title,
            target=preview.target,
            content=preview.effect,
            request_id=preview.request_id,
            risk=preview.risk,
        )
