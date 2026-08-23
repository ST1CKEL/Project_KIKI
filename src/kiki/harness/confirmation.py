"""The binding between what the user saw and what may then run.

The existing `ActionPreview` and `present_confirmation` show an action and
return a bool. That is a *display*, not a binding: it says "the user pressed
yes", not "the user pressed yes to exactly this call, with exactly these
arguments, in exactly this run". A bool cannot tell those apart, and a write
tool has to.

So a pending write is pinned to a fingerprint over the run, the call id, the
tool name and the canonical form of the arguments. Approving means presenting
that fingerprint back. Anything that changed in between — a different run, a
different call, one edited character in the note — produces a different
fingerprint and is refused, and a fingerprint can only be spent once.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from kiki.harness.models import ToolCall


def canonical_arguments(arguments: dict[str, Any]) -> str:
    """One stable text for one set of arguments.

    Sorted keys and no whitespace, so a re-ordered dictionary is the same
    arguments and a changed value is not.
    """
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(
    run_id: str, call: ToolCall, *, target: str = "", content: str = ""
) -> str:
    """What the user is agreeing to, reduced to sixteen hex characters.

    Covers the *displayed* target and content as well as the call, so the
    binding is to what was actually on screen. Without them, a proposal whose
    display was built differently from its arguments would still redeem — the
    dialog and the write could disagree and nothing would notice.
    """
    payload = "\x1f".join(
        (run_id, call.id, call.name, canonical_arguments(call.arguments), target, content)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ConfirmationError(Exception):
    """The approval did not match a pending request. Carries a category only."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ConfirmationRequest:
    """What the UI must show, and nothing the UI may change.

    `preview_lines` is the human-readable body. It holds the workspace-relative
    target and the full content, because a user cannot approve what they are not
    shown — but never an absolute path, so nothing about the machine leaks into
    a dialog or a screenshot of one.
    """

    run_id: str
    call: ToolCall
    tool_name: str
    title: str
    target: str
    content: str
    fingerprint: str = field(default="")

    @classmethod
    def build(
        cls, run_id: str, call: ToolCall, *, title: str, target: str, content: str
    ) -> ConfirmationRequest:
        return cls(
            run_id=run_id,
            call=call,
            tool_name=call.name,
            title=title,
            target=target,
            content=content,
            fingerprint=fingerprint(run_id, call, target=target, content=content),
        )

    @property
    def call_id(self) -> str:
        return self.call.id


class PendingConfirmation:
    """At most one pending write, spendable exactly once.

    Deliberately not a queue: a second proposal replaces the first, and a
    replaced proposal can never be approved afterwards.
    """

    def __init__(self) -> None:
        self._request: ConfirmationRequest | None = None
        self._spent: set[str] = set()

    @property
    def pending(self) -> ConfirmationRequest | None:
        return self._request

    def arm(self, request: ConfirmationRequest) -> None:
        self._request = request

    def clear(self) -> None:
        """Invalidate whatever was waiting. Idempotent."""
        self._request = None

    def approve(self, run_id: str, call_id: str, print_: str) -> ConfirmationRequest:
        """Redeem an approval, or say precisely why it does not count.

        The fingerprint is checked *and* consumed here, so a second press of the
        same button, a replayed dialog or a retried call cannot write twice.
        """
        request = self._request
        if request is None:
            raise ConfirmationError("no_pending_confirmation")
        if print_ in self._spent:
            raise ConfirmationError("confirmation_already_used")
        if request.run_id != run_id or request.call_id != call_id:
            raise ConfirmationError("confirmation_mismatch")
        if request.fingerprint != print_:
            # Arguments, tool or run changed after the user looked at them.
            raise ConfirmationError("confirmation_mismatch")
        self._spent.add(print_)
        self._request = None
        return request

    def reject(self, run_id: str, call_id: str) -> ConfirmationRequest:
        request = self._request
        if request is None:
            raise ConfirmationError("no_pending_confirmation")
        if request.run_id != run_id or request.call_id != call_id:
            raise ConfirmationError("confirmation_mismatch")
        self._request = None
        return request
