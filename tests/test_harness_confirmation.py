"""The binding: what the user saw is the only thing that may then run."""

from __future__ import annotations

import pytest

from kiki.harness.confirmation import (
    ConfirmationError,
    ConfirmationRequest,
    PendingConfirmation,
    canonical_arguments,
    fingerprint,
)
from kiki.harness.models import ToolCall


def _request(run_id="run-1", **arguments) -> ConfirmationRequest:
    call = ToolCall("create_note", arguments or {"title": "Milch", "content": "kaufen"})
    return ConfirmationRequest.build(
        run_id, call, title="create_note", target="milch.md", content="kaufen"
    )


def test_the_same_arguments_give_the_same_fingerprint() -> None:
    call = ToolCall("create_note", {"title": "a", "content": "b"}, id="call-1")
    reordered = ToolCall("create_note", {"content": "b", "title": "a"}, id="call-1")
    assert fingerprint("run-1", call) == fingerprint("run-1", reordered)


@pytest.mark.parametrize(
    ("run_id", "call"),
    [
        ("run-2", ToolCall("create_note", {"title": "a", "content": "b"}, id="call-1")),
        ("run-1", ToolCall("create_note", {"title": "a", "content": "b"}, id="call-2")),
        ("run-1", ToolCall("other_tool", {"title": "a", "content": "b"}, id="call-1")),
        ("run-1", ToolCall("create_note", {"title": "a", "content": "B"}, id="call-1")),
        ("run-1", ToolCall("create_note", {"title": "a"}, id="call-1")),
    ],
)
def test_anything_that_changed_changes_the_fingerprint(run_id, call) -> None:
    """Run, call, tool or one edited character — each must produce a new one."""
    original = ToolCall("create_note", {"title": "a", "content": "b"}, id="call-1")
    assert fingerprint(run_id, call) != fingerprint("run-1", original)


def test_the_canonical_form_is_order_independent() -> None:
    assert canonical_arguments({"b": 1, "a": 2}) == canonical_arguments({"a": 2, "b": 1})


def test_an_approval_matching_the_proposal_is_redeemed() -> None:
    pending = PendingConfirmation()
    request = _request()
    pending.arm(request)

    redeemed = pending.approve(request.run_id, request.call_id, request.fingerprint)
    assert redeemed is request
    assert pending.pending is None


def test_the_same_approval_cannot_be_spent_twice() -> None:
    """A second press of the same button must not write a second note."""
    pending = PendingConfirmation()
    request = _request()
    pending.arm(request)
    pending.approve(request.run_id, request.call_id, request.fingerprint)
    pending.arm(request)

    with pytest.raises(ConfirmationError) as excinfo:
        pending.approve(request.run_id, request.call_id, request.fingerprint)
    assert excinfo.value.code == "confirmation_already_used"


def test_an_approval_for_a_changed_proposal_is_refused() -> None:
    pending = PendingConfirmation()
    request = _request()
    pending.arm(request)
    tampered = _request(title="Milch", content="ETWAS ANDERES")

    with pytest.raises(ConfirmationError) as excinfo:
        pending.approve(request.run_id, request.call_id, tampered.fingerprint)
    assert excinfo.value.code == "confirmation_mismatch"
    assert pending.pending is request, "der Vorschlag bleibt bestehen"


@pytest.mark.parametrize(("run_id", "call_id"), [("run-anders", None), (None, "call-anders")])
def test_an_approval_for_another_run_or_call_is_refused(run_id, call_id) -> None:
    pending = PendingConfirmation()
    request = _request()
    pending.arm(request)

    with pytest.raises(ConfirmationError) as excinfo:
        pending.approve(
            run_id or request.run_id, call_id or request.call_id, request.fingerprint
        )
    assert excinfo.value.code == "confirmation_mismatch"


def test_an_approval_without_a_proposal_is_refused() -> None:
    with pytest.raises(ConfirmationError) as excinfo:
        PendingConfirmation().approve("run-1", "call-1", "abc")
    assert excinfo.value.code == "no_pending_confirmation"


def test_a_cleared_proposal_can_never_be_approved() -> None:
    """Shutdown, cancel or a window closing must make it unredeemable."""
    pending = PendingConfirmation()
    request = _request()
    pending.arm(request)
    pending.clear()

    with pytest.raises(ConfirmationError):
        pending.approve(request.run_id, request.call_id, request.fingerprint)


def test_a_replacing_proposal_invalidates_the_first() -> None:
    pending = PendingConfirmation()
    first = _request(title="Erste", content="a")
    second = _request(title="Zweite", content="b")
    pending.arm(first)
    pending.arm(second)

    with pytest.raises(ConfirmationError):
        pending.approve(first.run_id, first.call_id, first.fingerprint)


def test_rejecting_clears_the_proposal() -> None:
    pending = PendingConfirmation()
    request = _request()
    pending.arm(request)
    pending.reject(request.run_id, request.call_id)

    assert pending.pending is None
    with pytest.raises(ConfirmationError):
        pending.approve(request.run_id, request.call_id, request.fingerprint)


def test_the_request_shows_the_target_and_content_but_no_path() -> None:
    request = _request()
    assert request.target == "milch.md"
    assert request.content == "kaufen"
    assert "/" not in request.target
    assert "/home" not in repr(request)
