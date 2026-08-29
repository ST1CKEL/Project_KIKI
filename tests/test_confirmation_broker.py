"""The binding between a human "yes" and exactly one action.

The four questions this has to answer without exception: is it the same run, the
same call, the same arguments, and has it been spent already. Everything else
here follows from those.
"""

from __future__ import annotations

import time

import pytest

from kiki.tools.confirmation import (
    REDACTED,
    ConfirmationBroker,
    ConfirmationError,
    binding,
    canonical_arguments,
    preview_digest,
    spec_digest,
)
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ActionPreview, ToolRegistry, ToolSpec

ARGS = {"percent": 30, "device": "speaker"}


def _spec(**kwargs) -> ToolSpec:
    values = dict(
        name="audio.set_volume",
        title="Lautstärke",
        description="Setzt die Lautstärke.",
        risk=RiskLevel.CONTROL,
        parameters={
            "type": "object",
            "properties": {"percent": {"type": "integer"}, "device": {"type": "string"}},
            "required": [],
            "additionalProperties": False,
        },
        handler=lambda params: {"volume": 30},
        effect="Ändert die Lautstärke.",
        auto_allow=True,
        requires_integration=False,
        model_callable=True,
    )
    values.update(kwargs)
    return ToolSpec(**values)


def _preview(spec: ToolSpec, arguments: dict) -> ActionPreview:
    return ToolRegistry().preview(spec, arguments, "Von KIKI angefordert.")


def _propose(broker, *, run_id="run-1", call_id="call-1", spec=None, arguments=None):
    spec = spec or _spec()
    arguments = ARGS if arguments is None else arguments
    return broker.request(
        run_id=run_id,
        call_id=call_id,
        spec=spec,
        arguments=arguments,
        preview=_preview(spec, arguments),
    )


def _redeem(broker, grant, *, run_id="run-1", call_id="call-1", spec=None, arguments=None):
    spec = spec or _spec()
    arguments = ARGS if arguments is None else arguments
    broker.redeem(
        grant,
        run_id=run_id,
        call_id=call_id,
        spec=spec,
        arguments=arguments,
        preview=_preview(spec, arguments),
    )


# --- the happy path ---------------------------------------------------------


def test_an_approved_action_can_be_executed_once() -> None:
    broker = ConfirmationBroker()
    request = _propose(broker)
    grant = broker.approve(request.id)

    _redeem(broker, grant)  # no exception is the whole assertion


def test_the_ui_only_ever_sees_an_opaque_id() -> None:
    """The UI must not compute or carry an authorisation value — it says "the
    person approved request X" and nothing else."""
    broker = ConfirmationBroker()
    request = _propose(broker)

    assert isinstance(request.id, str) and len(request.id) >= 12
    assert request.binding not in repr(request)
    # Everything the dialog needs is on the preview, not on a token.
    assert request.preview.title == "Lautstärke"
    assert request.preview.params == ARGS


def test_the_request_carries_what_was_displayed() -> None:
    broker = ConfirmationBroker()
    request = _propose(broker)
    assert request.tool_name == "audio.set_volume"
    assert request.preview.risk is RiskLevel.CONTROL


# --- the four refusals ------------------------------------------------------


def test_a_grant_for_another_run_is_refused() -> None:
    broker = ConfirmationBroker()
    grant = broker.approve(_propose(broker).id)

    with pytest.raises(ConfirmationError) as excinfo:
        _redeem(broker, grant, run_id="run-anders")
    assert excinfo.value.code == "confirmation_mismatch"


def test_a_grant_for_another_call_is_refused() -> None:
    broker = ConfirmationBroker()
    grant = broker.approve(_propose(broker).id)

    with pytest.raises(ConfirmationError) as excinfo:
        _redeem(broker, grant, call_id="call-anders")
    assert excinfo.value.code == "confirmation_mismatch"


@pytest.mark.parametrize(
    "changed",
    [
        {"percent": 31, "device": "speaker"},
        {"percent": 30, "device": "kopfhörer"},
        {"percent": 30},
        {"percent": 30, "device": "speaker", "extra": True},
    ],
)
def test_edited_arguments_invalidate_the_approval(changed) -> None:
    """One changed character after the dialog was shown voids the yes."""
    broker = ConfirmationBroker()
    grant = broker.approve(_propose(broker).id)

    with pytest.raises(ConfirmationError) as excinfo:
        _redeem(broker, grant, arguments=changed)
    assert excinfo.value.code == "confirmation_mismatch"


def test_a_grant_cannot_be_spent_twice() -> None:
    """A second press of the same button, a replayed dialog, a retried call."""
    broker = ConfirmationBroker()
    grant = broker.approve(_propose(broker).id)
    _redeem(broker, grant)

    with pytest.raises(ConfirmationError) as excinfo:
        _redeem(broker, grant)
    assert excinfo.value.code == "confirmation_already_used"


# --- the tool itself may not change underneath ------------------------------


@pytest.mark.parametrize(
    "changed",
    [
        {"risk": RiskLevel.WRITE},
        {"model_callable": False},
        {"auto_allow": False},
        {"requires_integration": True},
        {"allowed_in_panic": True},
        {"target": "remote"},
        {"allowed_profiles": ("observe",)},
    ],
)
def test_a_changed_tool_definition_invalidates_the_approval(changed) -> None:
    """The dialog described a tool. If that tool's security shape changed in the
    meantime, the person approved something that no longer exists."""
    broker = ConfirmationBroker()
    grant = broker.approve(_propose(broker).id)

    with pytest.raises(ConfirmationError) as excinfo:
        _redeem(broker, grant, spec=_spec(**changed))
    assert excinfo.value.code == "confirmation_mismatch"


def test_a_cosmetic_tool_change_does_not_invalidate_it() -> None:
    """A reworded description must not throw the user's decision away."""
    broker = ConfirmationBroker()
    spec = _spec()
    request = broker.request(
        run_id="run-1", call_id="call-1", spec=spec, arguments=ARGS,
        preview=_preview(spec, ARGS),
    )
    grant = broker.approve(request.id)

    reworded = _spec(description="Setzt die Wiedergabelautstärke.")
    broker.redeem(
        grant, run_id="run-1", call_id="call-1", spec=reworded, arguments=ARGS,
        preview=_preview(spec, ARGS),
    )


def test_a_different_preview_invalidates_the_approval() -> None:
    """What was shown and what runs must be the same thing."""
    broker = ConfirmationBroker()
    spec = _spec()
    grant = broker.approve(_propose(broker, spec=spec).id)
    forged = ActionPreview(
        tool=spec.name, title="Lautstärke", params=ARGS, target=spec.target,
        effect="Harmlos.", risk=RiskLevel.READ, reason="Von KIKI angefordert.",
    )

    with pytest.raises(ConfirmationError) as excinfo:
        broker.redeem(
            grant, run_id="run-1", call_id="call-1", spec=spec, arguments=ARGS,
            preview=forged,
        )
    assert excinfo.value.code == "confirmation_mismatch"


# --- lifetime ---------------------------------------------------------------


def test_an_unapproved_request_cannot_be_redeemed() -> None:
    broker = ConfirmationBroker()
    _propose(broker)
    with pytest.raises(ConfirmationError) as excinfo:
        _redeem(broker, None)
    assert excinfo.value.code == "no_pending_confirmation"


def test_approving_something_that_is_not_pending_is_refused() -> None:
    broker = ConfirmationBroker()
    with pytest.raises(ConfirmationError) as excinfo:
        broker.approve("gibtsnicht")
    assert excinfo.value.code == "no_pending_confirmation"


def test_the_same_request_cannot_be_approved_twice() -> None:
    broker = ConfirmationBroker()
    request = _propose(broker)
    broker.approve(request.id)

    with pytest.raises(ConfirmationError):
        broker.approve(request.id)


def test_a_rejected_request_can_never_be_approved() -> None:
    broker = ConfirmationBroker()
    request = _propose(broker)
    broker.reject(request.id)

    with pytest.raises(ConfirmationError) as excinfo:
        broker.approve(request.id)
    assert excinfo.value.code == "no_pending_confirmation"


def test_a_replacing_proposal_invalidates_the_first() -> None:
    broker = ConfirmationBroker()
    first = _propose(broker)
    _propose(broker, arguments={"percent": 50})

    with pytest.raises(ConfirmationError):
        broker.approve(first.id)
    assert len(broker.open_requests) == 1


def test_a_cancelled_run_takes_its_authorisations_with_it() -> None:
    broker = ConfirmationBroker()
    waiting = _propose(broker, call_id="call-1")
    granted = broker.approve(_propose(broker, call_id="call-2").id)

    assert broker.cancel_run("run-1") == 2

    with pytest.raises(ConfirmationError):
        broker.approve(waiting.id)
    with pytest.raises(ConfirmationError) as excinfo:
        _redeem(broker, granted, call_id="call-2")
    assert excinfo.value.code == "confirmation_already_used"


def test_cancelling_another_run_leaves_this_one_alone() -> None:
    broker = ConfirmationBroker()
    grant = broker.approve(_propose(broker).id)
    assert broker.cancel_run("run-fremd") == 0
    _redeem(broker, grant)


def test_a_forgotten_dialog_expires() -> None:
    broker = ConfirmationBroker(ttl_s=0.05)
    request = _propose(broker)
    time.sleep(0.06)

    with pytest.raises(ConfirmationError) as excinfo:
        broker.approve(request.id)
    assert excinfo.value.code == "no_pending_confirmation"


def test_an_old_grant_expires_before_it_is_spent() -> None:
    broker = ConfirmationBroker(ttl_s=0.05)
    grant = broker.approve(_propose(broker).id)
    time.sleep(0.06)

    with pytest.raises(ConfirmationError) as excinfo:
        _redeem(broker, grant)
    assert excinfo.value.code in {"authorization_expired", "confirmation_already_used"}


def test_shutdown_clears_everything() -> None:
    broker = ConfirmationBroker()
    waiting = _propose(broker, call_id="call-1")
    grant = broker.approve(_propose(broker, call_id="call-2").id)
    broker.clear()

    with pytest.raises(ConfirmationError):
        broker.approve(waiting.id)
    with pytest.raises(ConfirmationError):
        _redeem(broker, grant, call_id="call-2")


# --- the binding itself -----------------------------------------------------


def test_the_binding_is_a_full_sha256() -> None:
    """Not truncated: a shortened hash saves nothing and invites an argument
    about collisions that nobody wants to have about an authorisation."""
    spec = _spec()
    value = binding(
        run_id="run-1", call_id="call-1", spec=spec, arguments=ARGS,
        preview=_preview(spec, ARGS),
    )
    assert len(value) == 64
    assert all(char in "0123456789abcdef" for char in value)


def test_reordered_arguments_are_the_same_arguments() -> None:
    assert canonical_arguments({"b": 1, "a": 2}) == canonical_arguments({"a": 2, "b": 1})


def test_the_digests_cover_what_they_should() -> None:
    spec = _spec()
    assert "audio.set_volume" in spec_digest(spec)
    assert "control" in spec_digest(spec)
    assert "Lautstärke" in preview_digest(_preview(spec, ARGS))


# --- nothing secret escapes -------------------------------------------------


def test_a_grant_never_shows_its_token() -> None:
    broker = ConfirmationBroker()
    grant = broker.approve(_propose(broker).id)

    for text in (repr(grant), str(grant), f"{grant}"):
        assert grant.token not in text
        assert grant.binding not in text
    assert REDACTED in str(grant)


def test_a_confirmation_error_carries_only_a_category() -> None:
    broker = ConfirmationBroker()
    grant = broker.approve(_propose(broker).id)
    try:
        _redeem(broker, grant, arguments={"percent": 99})
    except ConfirmationError as exc:
        assert exc.code == "confirmation_mismatch"
        assert " " not in str(exc)
        assert "99" not in str(exc)


def test_exactly_one_place_redeems_grants() -> None:
    """The point of the consolidation: one broker, one consumer.

    A second module reaching for the broker would be the beginning of the very
    parallel authority chain this is meant to remove.
    """
    import io
    import tokenize
    from pathlib import Path

    def _mentions_in_code(path: Path) -> bool:
        """Comments and docstrings are allowed to name it; code is not."""
        source = path.read_text(encoding="utf-8")
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        except (tokenize.TokenError, IndentationError):
            return "ConfirmationBroker" in source
        return any(
            token.type not in (tokenize.COMMENT, tokenize.STRING)
            and "ConfirmationBroker" in token.string
            for token in tokens
        )

    root = Path(__file__).resolve().parents[1] / "src" / "kiki"
    users = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*.py") if _mentions_in_code(path)
    )
    assert users == ["tools/confirmation.py", "tools/executor.py"], users
