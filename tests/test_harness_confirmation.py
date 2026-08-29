"""The binding: what the user saw is the only thing that may then run.

These twelve guarantees used to be proved against a second confirmation system
that lived in `kiki.harness.confirmation` -- a sixteen-character fingerprint in
a one-shot slot, with the dialog handing the fingerprint back to spend it.

That module mints nothing now. Every guarantee below is the same guarantee,
proved against the one broker in `kiki.tools.confirmation` that the gateway
uses for both agent paths. The binding is stronger than what it replaced: full
SHA-256, and it covers the validated arguments and the tool spec, not just the
displayed text. Nothing was dropped in the move; `test_confirmation_broker.py`
carries the rest.
"""

from __future__ import annotations

import pytest

from kiki.harness.confirmation import ConfirmationError, ConfirmationRequest
from kiki.tools.confirmation import (
    ConfirmationBroker,
    binding,
    canonical_arguments,
)
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ActionPreview, ToolRegistry, ToolSpec

ARGS = {"title": "a", "content": "b"}


def _spec(name="create_note"):
    return ToolSpec(
        name=name,
        title="Notiz anlegen",
        description="d",
        risk=RiskLevel.WRITE,
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda params: {"created": True},
        effect="Legt eine Notiz an.",
        auto_allow=True,
        requires_integration=False,
        model_callable=True,
    )


def _preview(spec, arguments=None, reason=""):
    return ToolRegistry().preview(spec, dict(arguments or ARGS), reason)


def _armed(broker, *, run_id="run-1", call_id="call-1", spec=None, arguments=None):
    spec = spec or _spec()
    arguments = dict(arguments or ARGS)
    preview = _preview(spec, arguments)
    request = broker.request(
        run_id=run_id, call_id=call_id, spec=spec, arguments=arguments, preview=preview
    )
    return request, spec, arguments, preview


# -- the binding --------------------------------------------------------------


def test_the_same_arguments_give_the_same_binding() -> None:
    spec = _spec()
    one = binding(
        run_id="run-1", call_id="call-1", spec=spec, arguments=ARGS, preview=_preview(spec)
    )
    reordered = {"content": "b", "title": "a"}
    two = binding(
        run_id="run-1",
        call_id="call-1",
        spec=spec,
        arguments=reordered,
        preview=_preview(spec, reordered),
    )
    assert one == two


@pytest.mark.parametrize(
    ("run_id", "call_id", "tool", "arguments"),
    [
        ("run-2", "call-1", "create_note", ARGS),
        ("run-1", "call-2", "create_note", ARGS),
        ("run-1", "call-1", "other_tool", ARGS),
        ("run-1", "call-1", "create_note", {"title": "a", "content": "B"}),
        ("run-1", "call-1", "create_note", {"title": "a"}),
    ],
)
def test_anything_that_changed_changes_the_binding(run_id, call_id, tool, arguments) -> None:
    """Run, call, tool or one edited character -- each must produce a new one."""
    original = binding(
        run_id="run-1",
        call_id="call-1",
        spec=_spec(),
        arguments=ARGS,
        preview=_preview(_spec()),
    )
    spec = _spec(tool)
    changed = binding(
        run_id=run_id,
        call_id=call_id,
        spec=spec,
        arguments=arguments,
        preview=_preview(spec, arguments),
    )
    assert changed != original


def test_the_canonical_form_is_order_independent() -> None:
    assert canonical_arguments({"b": 1, "a": 2}) == canonical_arguments({"a": 2, "b": 1})


def test_the_binding_is_not_truncated() -> None:
    """The old fingerprint was sixteen hex characters. This one is the whole hash."""
    value = binding(
        run_id="run-1", call_id="call-1", spec=_spec(), arguments=ARGS, preview=_preview(_spec())
    )
    assert len(value) == 64


# -- redeeming ----------------------------------------------------------------


def test_an_approval_matching_the_proposal_is_redeemed() -> None:
    broker = ConfirmationBroker()
    request, spec, arguments, preview = _armed(broker)
    grant = broker.approve(request.id)
    broker.redeem(
        grant, run_id="run-1", call_id="call-1", spec=spec, arguments=arguments, preview=preview
    )


def test_the_same_approval_cannot_be_spent_twice() -> None:
    broker = ConfirmationBroker()
    request, spec, arguments, preview = _armed(broker)
    grant = broker.approve(request.id)
    kwargs = dict(
        run_id="run-1", call_id="call-1", spec=spec, arguments=arguments, preview=preview
    )
    broker.redeem(grant, **kwargs)
    with pytest.raises(ConfirmationError) as caught:
        broker.redeem(grant, **kwargs)
    assert caught.value.code == "confirmation_already_used"


def test_an_approval_for_a_changed_proposal_is_refused() -> None:
    broker = ConfirmationBroker()
    request, spec, _arguments, preview = _armed(broker)
    grant = broker.approve(request.id)
    with pytest.raises(ConfirmationError) as caught:
        broker.redeem(
            grant,
            run_id="run-1",
            call_id="call-1",
            spec=spec,
            arguments={"title": "a", "content": "ETWAS ANDERES"},
            preview=preview,
        )
    assert caught.value.code == "confirmation_mismatch"


@pytest.mark.parametrize(
    ("run_id", "call_id"), [("run-gibtsnicht", "call-1"), ("run-1", "call-gibtsnicht")]
)
def test_an_approval_for_another_run_or_call_is_refused(run_id, call_id) -> None:
    broker = ConfirmationBroker()
    request, spec, arguments, preview = _armed(broker)
    grant = broker.approve(request.id)
    with pytest.raises(ConfirmationError):
        broker.redeem(
            grant,
            run_id=run_id,
            call_id=call_id,
            spec=spec,
            arguments=arguments,
            preview=preview,
        )


def test_an_approval_without_a_proposal_is_refused() -> None:
    broker = ConfirmationBroker()
    with pytest.raises(ConfirmationError) as caught:
        broker.approve("gibtsnicht")
    assert caught.value.code == "no_pending_confirmation"


def test_a_cleared_proposal_can_never_be_approved() -> None:
    broker = ConfirmationBroker()
    request, _spec_, _arguments, _card = _armed(broker)
    broker.clear()
    with pytest.raises(ConfirmationError):
        broker.approve(request.id)


def test_a_replacing_proposal_invalidates_the_first() -> None:
    """A second card for the same call means the first is no longer the question."""
    broker = ConfirmationBroker()
    first, spec, _arguments, _card = _armed(broker)
    second_args = {"title": "a", "content": "anders"}
    second, spec2, arguments2, preview2 = _armed(broker, arguments=second_args)
    assert first.id != second.id
    grant = broker.approve(second.id)
    with pytest.raises(ConfirmationError):
        broker.redeem(
            grant,
            run_id="run-1",
            call_id="call-1",
            spec=spec,
            arguments=ARGS,
            preview=_preview(spec),
        )


def test_rejecting_clears_the_proposal() -> None:
    broker = ConfirmationBroker()
    request, _spec_, _arguments, _card = _armed(broker)
    broker.reject(request.id)
    with pytest.raises(ConfirmationError):
        broker.approve(request.id)


def test_cancelling_the_run_voids_what_was_waiting() -> None:
    broker = ConfirmationBroker()
    request, _spec_, _arguments, _card = _armed(broker)
    assert broker.cancel_run("run-1") == 1
    with pytest.raises(ConfirmationError):
        broker.approve(request.id)


# -- what the dialog is handed -------------------------------------------------


def test_the_request_shows_the_target_and_content_but_no_path() -> None:
    preview = ActionPreview(
        tool="create_note",
        title="Notiz anlegen",
        params={"Datei": "milch.md"},
        target="milch.md",
        effect="Legt eine Notiz an.",
        risk=RiskLevel.WRITE,
        reason="",
        request_id="abc123",
    )
    request = ConfirmationRequest.from_preview("run-1", "call-1", preview)
    assert request.tool_name == "create_note"
    assert request.target == "milch.md"
    assert request.request_id == "abc123"
    assert "/home/" not in repr(request)


def test_the_display_record_mints_nothing() -> None:
    """It names the question. It cannot authorise an answer to it."""
    import kiki.harness.confirmation as module

    source = module.__file__
    assert not hasattr(module, "fingerprint")
    assert not hasattr(module, "PendingConfirmation")
    assert not hasattr(ConfirmationRequest, "build")
    assert source.endswith("confirmation.py")


def test_only_the_broker_issues_grants() -> None:
    """Exactly one place in the tree mints an authorisation to run a tool."""
    from pathlib import Path

    minting = sorted(
        str(path.relative_to(Path("src")))
        for path in Path("src").rglob("*.py")
        if "ConfirmationGrant(" in path.read_text(encoding="utf-8")
    )
    assert minting == ["kiki/tools/confirmation.py"]
