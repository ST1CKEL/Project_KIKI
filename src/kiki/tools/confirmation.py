"""Binding a human "yes" to exactly one action.

`ConfirmFn` returns a bool. A bool cannot tell "yes to this call with these
arguments in this run" apart from "yes to something", so nothing stops a
confirmed action from being executed with different arguments, twice, or after
the run it belonged to is gone. This module supplies that binding.

How it is split
---------------
The UI is handed a `ConfirmationRequest` and knows one thing about it: an opaque
`id`. It never computes, holds or passes an authorisation value — it says "the
person approved request X". Only then does the broker mint a
`ConfirmationGrant`, and only the caller that is about to execute redeems it.

Redeeming re-derives the binding from what is *actually* about to run and
compares it against what was displayed. Arguments edited in between, a different
call, a different run, or a `ToolSpec` whose security-relevant fields changed all
produce a different binding and are refused. A grant is spendable once, expires
on its own, and dies with its run.

What this module deliberately does not do: decide whether an action is allowed.
A grant is a human authorisation, never a policy override — a later check can
still refuse an action the user approved, and must.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from kiki.tools.registry import ActionPreview, ToolSpec

# Long enough that guessing is hopeless, short enough to stay readable in a
# debugger. Never logged.
_TOKEN_BYTES = 32
_ID_BYTES = 12
# A dialog a person left open for two minutes is a dialog they have forgotten.
DEFAULT_TTL_S = 120.0

REDACTED = "[grant]"


class ConfirmationError(Exception):
    """A category, never a message: these strings reach logs and the model."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def canonical_arguments(arguments: dict[str, Any]) -> str:
    """One stable text for one set of arguments.

    Sorted keys and no whitespace, so a re-ordered dictionary is the same
    arguments and one changed character is not.
    """
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def preview_digest(preview: ActionPreview) -> str:
    """What the person actually saw, reduced to one stable text.

    The preview is part of the binding because the dialog is the whole point: if
    what is displayed and what is executed can drift apart, the confirmation
    means nothing.
    """
    return canonical_arguments(
        {
            "tool": preview.tool,
            "title": preview.title,
            "params": preview.params,
            "target": preview.target,
            "effect": preview.effect,
            "risk": preview.risk.value,
            "reason": preview.reason,
        }
    )


def spec_digest(spec: ToolSpec) -> str:
    """The security-relevant shape of a tool.

    Only the fields that decide whether an action is safe. A title change must
    not invalidate a dialog; a risk level change must.
    """
    return canonical_arguments(
        {
            "name": spec.name,
            "risk": spec.risk.value,
            "target": spec.target,
            "auto_allow": spec.auto_allow,
            "requires_integration": spec.requires_integration,
            "allowed_profiles": sorted(spec.allowed_profiles),
            "allowed_in_panic": spec.allowed_in_panic,
            "model_callable": spec.model_callable,
        }
    )


def binding(
    *,
    run_id: str,
    call_id: str,
    spec: ToolSpec,
    arguments: dict[str, Any],
    preview: ActionPreview,
) -> str:
    """The full SHA-256 of everything the approval is tied to.

    Not truncated: a shortened hash saves nothing here and only invites a
    collision argument nobody wants to have about an authorisation.
    """
    payload = "\x1f".join(
        (
            run_id,
            call_id,
            spec.name,
            canonical_arguments(arguments),
            preview_digest(preview),
            spec_digest(spec),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConfirmationRequest:
    """What the UI is given. `id` is the only part it needs to hand back."""

    id: str
    run_id: str
    call_id: str
    tool_name: str
    preview: ActionPreview
    binding: str = field(repr=False)
    created_at: float = field(default_factory=time.monotonic)

    def expired(self, ttl: float, *, now: float | None = None) -> bool:
        return (now if now is not None else time.monotonic()) - self.created_at > ttl


@dataclass(frozen=True)
class ConfirmationGrant:
    """A one-shot authorisation. The token never appears in a log or a trace."""

    token: str = field(repr=False)
    request_id: str
    run_id: str
    call_id: str
    tool_name: str
    binding: str = field(repr=False)
    issued_at: float = field(default_factory=time.monotonic)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"ConfirmationGrant({self.tool_name} {REDACTED})"


class ConfirmationBroker:
    """Holds pending requests and mints grants. One request per call id."""

    def __init__(self, *, ttl_s: float = DEFAULT_TTL_S) -> None:
        self._ttl = float(ttl_s)
        self._pending: dict[str, ConfirmationRequest] = {}
        self._grants: dict[str, ConfirmationGrant] = {}

    # --- proposing ---------------------------------------------------------

    def request(
        self,
        *,
        run_id: str,
        call_id: str,
        spec: ToolSpec,
        arguments: dict[str, Any],
        preview: ActionPreview,
    ) -> ConfirmationRequest:
        """Register one proposal and return what the UI should display.

        A second proposal for the same call replaces the first, and the replaced
        one can no longer be approved.
        """
        self.expire()
        for existing in [r for r in self._pending.values() if r.call_id == call_id]:
            self._pending.pop(existing.id, None)
        request = ConfirmationRequest(
            id=secrets.token_urlsafe(_ID_BYTES),
            run_id=run_id,
            call_id=call_id,
            tool_name=spec.name,
            preview=preview,
            binding=binding(
                run_id=run_id,
                call_id=call_id,
                spec=spec,
                arguments=arguments,
                preview=preview,
            ),
        )
        self._pending[request.id] = request
        return request

    def pending(self, request_id: str) -> ConfirmationRequest | None:
        self.expire()
        return self._pending.get(request_id)

    @property
    def open_requests(self) -> tuple[str, ...]:
        return tuple(sorted(self._pending))

    # --- deciding ----------------------------------------------------------

    def approve(self, request_id: str) -> ConfirmationGrant:
        """The person said yes to this request. Mints the grant.

        The UI reaches exactly this method with exactly one string. It computes
        nothing, so it cannot be talked into authorising something else.
        """
        self.expire()
        request = self._pending.pop(request_id, None)
        if request is None:
            raise ConfirmationError("no_pending_confirmation")
        grant = ConfirmationGrant(
            token=secrets.token_urlsafe(_TOKEN_BYTES),
            request_id=request.id,
            run_id=request.run_id,
            call_id=request.call_id,
            tool_name=request.tool_name,
            binding=request.binding,
        )
        self._grants[grant.token] = grant
        return grant

    def reject(self, request_id: str) -> ConfirmationRequest:
        request = self._pending.pop(request_id, None)
        if request is None:
            raise ConfirmationError("no_pending_confirmation")
        return request

    # --- redeeming ---------------------------------------------------------

    def redeem(
        self,
        grant: ConfirmationGrant | None,
        *,
        run_id: str,
        call_id: str,
        spec: ToolSpec,
        arguments: dict[str, Any],
        preview: ActionPreview,
    ) -> None:
        """Spend one grant for exactly the action it was issued for.

        Called immediately before the side effect, with the values that are
        really about to be used — that is what makes an edit in between
        impossible rather than merely unlikely.
        """
        if grant is None:
            raise ConfirmationError("no_pending_confirmation")
        held = self._grants.pop(grant.token, None)
        if held is None:
            # Either never issued here, or already spent. Both are the same
            # answer to the caller.
            raise ConfirmationError("confirmation_already_used")
        if held.issued_at + self._ttl < time.monotonic():
            raise ConfirmationError("authorization_expired")
        if held.run_id != run_id or held.call_id != call_id or held.tool_name != spec.name:
            raise ConfirmationError("confirmation_mismatch")
        expected = binding(
            run_id=run_id,
            call_id=call_id,
            spec=spec,
            arguments=arguments,
            preview=preview,
        )
        if not secrets.compare_digest(held.binding, expected):
            raise ConfirmationError("confirmation_mismatch")

    # --- ending ------------------------------------------------------------

    def cancel_run(self, run_id: str) -> int:
        """A cancelled or finished run takes its authorisations with it."""
        dropped = [key for key, item in self._pending.items() if item.run_id == run_id]
        for key in dropped:
            self._pending.pop(key, None)
        spent = [key for key, item in self._grants.items() if item.run_id == run_id]
        for key in spent:
            self._grants.pop(key, None)
        return len(dropped) + len(spent)

    def clear(self) -> None:
        """Shutdown: nothing outstanding may survive the process."""
        self._pending.clear()
        self._grants.clear()

    def expire(self, *, now: float | None = None) -> int:
        moment = now if now is not None else time.monotonic()
        stale = [key for key, item in self._pending.items() if item.expired(self._ttl, now=moment)]
        for key in stale:
            self._pending.pop(key, None)
        old = [key for key, item in self._grants.items() if item.issued_at + self._ttl < moment]
        for key in old:
            self._grants.pop(key, None)
        return len(stale) + len(old)
