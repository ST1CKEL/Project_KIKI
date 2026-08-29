"""The production confirmation path, now running on bound grants.

`confirm=` still takes an `ActionPreview` and still returns a bool, so every
caller — ChatService, the application's approval card, the agent loop — is
untouched. What changed is underneath: the yes is registered before the dialog
opens and redeemed immediately before the side effect, against the values that
are really about to be used.
"""

from __future__ import annotations

import asyncio

from kiki.tools.audit import AuditLog
from kiki.tools.confirmation import ConfirmationBroker
from kiki.tools.executor import ToolExecutor
from kiki.tools.policy import Origin, RiskLevel, ToolPolicy
from kiki.tools.registry import ToolRegistry, ToolSpec


def _spec(handler=None, **kwargs) -> ToolSpec:
    ran: list[dict] = []

    def _default(params):
        ran.append(dict(params))
        return {"volume": params.get("percent")}

    values = dict(
        name="audio.set_volume",
        title="Lautstärke",
        description="Setzt die Lautstärke.",
        risk=RiskLevel.WRITE,
        parameters={
            "type": "object",
            "properties": {"percent": {"type": "integer"}},
            "required": [],
            "additionalProperties": False,
        },
        handler=handler or _default,
        effect="Ändert die Lautstärke.",
        auto_allow=True,
        requires_integration=False,
        model_callable=True,
    )
    values.update(kwargs)
    spec = ToolSpec(**values)
    return spec, ran


def _executor(db, spec, *, autonomy="trusted", broker=None) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(spec)
    return ToolExecutor(registry, ToolPolicy(autonomy), AuditLog(db), broker)


def _run(executor, *, confirm, params=None, **kwargs):
    return asyncio.run(
        executor.run(
            "audio.set_volume",
            {"percent": 30} if params is None else params,
            panic=False,
            integrations_enabled=True,
            confirm=confirm,
            origin=Origin.MODEL,
            **kwargs,
        )
    )


# --- the dialog behaves exactly as before -----------------------------------


def test_a_confirmed_action_runs(db) -> None:
    spec, ran = _spec()
    seen: list = []

    async def _confirm(preview):
        seen.append(preview)
        return True

    result = _run(_executor(db, spec), confirm=_confirm)

    assert result.ok is True
    assert result.data == {"volume": 30}
    assert ran == [{"percent": 30}]
    assert seen[0].title == "Lautstärke"
    assert seen[0].params == {"percent": 30}


def test_a_refused_action_does_not_run(db) -> None:
    spec, ran = _spec()

    async def _confirm(_preview):
        return False

    result = _run(_executor(db, spec), confirm=_confirm)

    assert result.ok is False
    assert "abgebrochen" in result.error
    assert ran == []


def test_a_synchronous_callback_still_works(db) -> None:
    """`ConfirmFn` may return a bool or an awaitable; both must keep working."""
    spec, ran = _spec()
    result = _run(_executor(db, spec), confirm=lambda _preview: True)

    assert result.ok is True
    assert ran == [{"percent": 30}]


def test_without_a_dialog_nothing_runs(db) -> None:
    spec, ran = _spec()
    result = _run(_executor(db, spec), confirm=None)

    assert result.ok is False
    assert ran == []


def test_a_read_tool_needs_no_confirmation(db) -> None:
    """The unconfirmed path must not have grown a broker round trip."""
    spec, ran = _spec(risk=RiskLevel.READ)
    result = _run(_executor(db, spec), confirm=None)

    assert result.ok is True
    assert ran == [{"percent": 30}]


# --- what the binding now prevents ------------------------------------------


def test_an_argument_edited_after_the_dialog_is_refused(db) -> None:
    """The case a bool could never catch: the dialog showed 30 percent, and
    something changed it to 100 before the tool ran."""
    spec, ran = _spec()

    async def _tamper(preview):
        preview.params["percent"] = 100
        return True

    result = _run(_executor(db, spec), confirm=_tamper)

    assert result.ok is False
    assert result.error == "Die Bestätigung gilt nicht mehr."
    assert ran == [], "das Tool darf mit den geänderten Werten nicht laufen"


def test_an_argument_added_after_the_dialog_is_refused(db) -> None:
    spec, ran = _spec()

    async def _tamper(preview):
        preview.params["device"] = "kopfhörer"
        return True

    assert _run(_executor(db, spec), confirm=_tamper).ok is False
    assert ran == []


def test_a_reworded_preview_is_refused(db) -> None:
    """What was displayed is part of the binding, not only the arguments."""
    spec, ran = _spec()

    async def _tamper(preview):
        preview.effect = "Völlig harmlos."
        return True

    assert _run(_executor(db, spec), confirm=_tamper).ok is False
    assert ran == []


def test_a_grant_cannot_be_reused_by_a_second_call(db) -> None:
    """Two identical confirmed calls need two confirmations, not one."""
    spec, ran = _spec()
    calls: list[int] = []

    async def _confirm(_preview):
        calls.append(1)
        return True

    executor = _executor(db, spec)
    assert _run(executor, confirm=_confirm).ok is True
    assert _run(executor, confirm=_confirm).ok is True

    assert len(calls) == 2, "jede Ausführung braucht ihre eigene Freigabe"
    assert len(ran) == 2


def test_an_expired_confirmation_is_refused(db) -> None:
    import time

    spec, ran = _spec()
    executor = _executor(db, spec, broker=ConfirmationBroker(ttl_s=0.02))

    async def _slow(_preview):
        time.sleep(0.05)
        return True

    result = _run(executor, confirm=_slow)

    assert result.ok is False
    assert result.error == "Die Bestätigung gilt nicht mehr."
    assert ran == []


def test_a_cancelled_run_invalidates_a_waiting_dialog(db) -> None:
    """Barge-in while the card is open: the yes that arrives afterwards is void."""
    spec, ran = _spec()
    executor = _executor(db, spec)

    async def _confirm(_preview):
        executor.confirmations.cancel_run("")
        return True

    result = _run(executor, confirm=_confirm)

    assert result.ok is False
    assert ran == []


# --- the audit keeps up -----------------------------------------------------


def _decisions(db) -> list[str]:
    return [
        row["decision"]
        for row in db.conn.execute("SELECT decision FROM audit_log ORDER BY id").fetchall()
    ]


def test_the_audit_still_tells_the_story(db) -> None:
    spec, _ran = _spec()
    _run(_executor(db, spec), confirm=lambda _p: True)
    assert _decisions(db) == ["confirm", "confirmed", "executed"]


def test_a_refusal_is_recorded_as_cancelled(db) -> None:
    spec, _ran = _spec()
    _run(_executor(db, spec), confirm=lambda _p: False)
    assert _decisions(db) == ["confirm", "cancelled"]


def test_a_broken_binding_is_recorded_as_denied(db) -> None:
    spec, _ran = _spec()

    async def _tamper(preview):
        preview.params["percent"] = 100
        return True

    _run(_executor(db, spec), confirm=_tamper)
    assert _decisions(db) == ["confirm", "confirmed", "denied"]


def test_the_denial_reason_is_a_category_not_a_message(db) -> None:
    spec, _ran = _spec()

    async def _tamper(preview):
        preview.params["percent"] = 100
        return True

    _run(_executor(db, spec), confirm=_tamper)
    errors = [
        row["error"]
        for row in db.conn.execute(
            "SELECT error FROM audit_log WHERE decision='denied'"
        ).fetchall()
    ]
    assert errors == ["confirmation_mismatch"]


# --- the run and call identity ----------------------------------------------


def test_a_caller_may_pass_its_own_run_and_call(db) -> None:
    """Forward compatibility: once runs exist, the binding covers them too."""
    spec, ran = _spec()
    result = _run(
        _executor(db, spec), confirm=lambda _p: True, run_id="run-1", call_id="call-1"
    )

    assert result.ok is True
    assert ran == [{"percent": 30}]


def test_a_confirmation_belongs_to_the_run_it_was_given_for(db) -> None:
    spec, ran = _spec()
    executor = _executor(db, spec)

    async def _confirm(_preview):
        # Something else cancels the run this dialog belongs to.
        executor.confirmations.cancel_run("run-1")
        return True

    result = _run(executor, confirm=_confirm, run_id="run-1", call_id="call-1")

    assert result.ok is False
    assert ran == []


def test_cancelling_a_different_run_leaves_this_one_alone(db) -> None:
    spec, ran = _spec()
    executor = _executor(db, spec)

    async def _confirm(_preview):
        executor.confirmations.cancel_run("run-fremd")
        return True

    result = _run(executor, confirm=_confirm, run_id="run-1", call_id="call-1")

    assert result.ok is True
    assert ran == [{"percent": 30}]


def test_nothing_leaks_into_the_result(db) -> None:
    spec, _ran = _spec()

    async def _tamper(preview):
        preview.params["percent"] = 100
        return True

    result = _run(_executor(db, spec), confirm=_tamper)
    assert "sha" not in result.error.lower()
    assert "grant" not in result.error.lower()
    assert len(result.error) < 80
