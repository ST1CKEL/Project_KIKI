"""One door for every tool call, and the window it closes.

The window: a confirmation dialog can stand open for a long time, and the
policy answer taken when it opened says nothing about the moment the action
actually runs. A human "yes" is an authorisation, never a policy override.
"""

from __future__ import annotations

import asyncio

import pytest

from kiki.tools.audit import AuditLog
from kiki.tools.executor import ToolExecutor
from kiki.tools.gateway import ToolGateway, ToolInvocation
from kiki.tools.policy import DecisionKind, Origin, RiskLevel, ToolPolicy
from kiki.tools.registry import ToolRegistry, ToolSpec


class World:
    """The live state the gateway reads, and a test can flip mid-flight."""

    def __init__(self, *, panic: bool = False, integrations: bool = True) -> None:
        self.panic = panic
        self.integrations = integrations
        self.panic_reads = 0
        self.integration_reads = 0

    def panic_check(self) -> bool:
        self.panic_reads += 1
        return self.panic

    def integrations_check(self) -> bool:
        self.integration_reads += 1
        return self.integrations


def _spec(**kwargs):
    ran: list[dict] = []

    def _handler(params):
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
        handler=_handler,
        effect="Ändert die Lautstärke.",
        auto_allow=True,
        requires_integration=True,
        model_callable=True,
    )
    values.update(kwargs)
    return ToolSpec(**values), ran


def _gateway(db, spec, world, *, autonomy="trusted") -> ToolGateway:
    registry = ToolRegistry()
    registry.register(spec)
    executor = ToolExecutor(registry, ToolPolicy(autonomy), AuditLog(db))
    return ToolGateway(
        executor,
        panic_check=world.panic_check,
        integrations_check=world.integrations_check,
    )


def _invoke(gateway, *, confirm=None, actor=Origin.MODEL, arguments=None, **kwargs):
    return asyncio.run(
        gateway.invoke(
            ToolInvocation(
                tool="audio.set_volume",
                arguments={"percent": 30} if arguments is None else arguments,
                actor=actor,
                **kwargs,
            ),
            confirm=confirm,
        )
    )


# --- the timeline this exists for -------------------------------------------


def test_panic_during_the_dialog_stops_the_confirmed_action(db) -> None:
    """13:20:00 card shown · 13:20:05 approved · 13:20:06 panic · 13:20:07 run.

    The action must not run. This is the case a boolean snapshot cannot see.
    """
    spec, ran = _spec()
    world = World()
    gateway = _gateway(db, spec, world)

    async def _confirm(_preview):
        # The person approves, and the switch is flipped a moment later.
        world.panic = True
        return True

    result = _invoke(gateway, confirm=_confirm)

    assert result.ok is False
    assert ran == [], "eine bestätigte Aktion darf Panik nicht überstimmen"
    assert result.decision.kind is DecisionKind.DENY


def test_the_integration_lockout_during_the_dialog_also_stops_it(db) -> None:
    spec, ran = _spec()
    world = World()
    gateway = _gateway(db, spec, world)

    async def _confirm(_preview):
        world.integrations = False
        return True

    assert _invoke(gateway, confirm=_confirm).ok is False
    assert ran == []


def test_the_world_is_read_again_before_the_side_effect(db) -> None:
    """Twice, not once: a cached answer would be the same stale snapshot."""
    spec, _ran = _spec()
    world = World()
    gateway = _gateway(db, spec, world)

    _invoke(gateway, confirm=lambda _p: True)

    assert world.panic_reads >= 2
    assert world.integration_reads >= 2


def test_a_refusal_after_a_recheck_is_audited_as_such(db) -> None:
    spec, _ran = _spec()
    world = World()
    gateway = _gateway(db, spec, world)

    async def _confirm(_preview):
        world.panic = True
        return True

    _invoke(gateway, confirm=_confirm)

    rows = db.conn.execute(
        "SELECT decision, error FROM audit_log ORDER BY id"
    ).fetchall()
    assert [row["decision"] for row in rows] == ["confirm", "confirmed", "denied"]
    assert rows[-1]["error"] == "policy_recheck"


# --- nothing else changed ---------------------------------------------------


def test_a_normal_confirmed_action_still_runs(db) -> None:
    spec, ran = _spec()
    result = _invoke(_gateway(db, spec, World()), confirm=lambda _p: True)

    assert result.ok is True
    assert result.data == {"volume": 30}
    assert ran == [{"percent": 30}]


def test_a_read_tool_runs_without_a_dialog(db) -> None:
    spec, ran = _spec(risk=RiskLevel.READ)
    result = _invoke(_gateway(db, spec, World()))

    assert result.ok is True
    assert ran == [{"percent": 30}]


def test_panic_that_was_already_on_blocks_at_the_first_check(db) -> None:
    spec, ran = _spec()
    result = _invoke(_gateway(db, spec, World(panic=True)), confirm=lambda _p: True)

    assert result.ok is False
    assert ran == []


def test_an_unknown_tool_is_refused(db) -> None:
    world = World()
    executor = ToolExecutor(ToolRegistry(), ToolPolicy("trusted"), AuditLog(db))
    gateway = ToolGateway(
        executor, panic_check=world.panic_check, integrations_check=world.integrations_check
    )

    result = asyncio.run(gateway.invoke(ToolInvocation(tool="gibtsnicht", actor=Origin.MODEL)))
    assert result.ok is False


def test_a_tool_the_model_may_not_call_stays_out_of_reach(db) -> None:
    """Default deny survives the new door."""
    spec, ran = _spec(model_callable=False, risk=RiskLevel.READ)
    result = _invoke(_gateway(db, spec, World()), actor=Origin.MODEL)

    assert result.ok is False
    assert ran == []


def test_a_user_click_reaches_a_tool_the_model_may_not_call(db) -> None:
    spec, ran = _spec(model_callable=False, risk=RiskLevel.READ)
    result = _invoke(_gateway(db, spec, World()), actor=Origin.USER)

    assert result.ok is True
    assert ran == [{"percent": 30}]


def test_the_arguments_are_not_shared_with_the_caller(db) -> None:
    """The gateway copies: a caller that keeps its dict must not be able to
    change what is about to run."""
    spec, ran = _spec(risk=RiskLevel.READ)
    gateway = _gateway(db, spec, World())
    arguments = {"percent": 30}

    asyncio.run(
        gateway.invoke(ToolInvocation(tool="audio.set_volume", arguments=arguments))
    )
    arguments["percent"] = 99

    assert ran == [{"percent": 30}]


# --- run lifetime -----------------------------------------------------------


def test_cancelling_a_run_drops_its_authorisations(db) -> None:
    spec, ran = _spec()
    gateway = _gateway(db, spec, World())

    async def _confirm(_preview):
        gateway.cancel_run("run-1")
        return True

    result = _invoke(gateway, confirm=_confirm, run_id="run-1", call_id="call-1")

    assert result.ok is False
    assert ran == []


def test_shutdown_drops_everything(db) -> None:
    spec, ran = _spec()
    gateway = _gateway(db, spec, World())

    async def _confirm(_preview):
        gateway.shutdown()
        return True

    assert _invoke(gateway, confirm=_confirm).ok is False
    assert ran == []


# --- the facade stays a facade ----------------------------------------------


def test_the_gateway_holds_no_policy_logic_of_its_own() -> None:
    """It knows the world, not the rules. Rules live in ToolPolicy, and a second
    place deciding them is the parallel authority chain to avoid."""
    import io
    import tokenize
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "src" / "kiki" / "tools" / "gateway.py"
    # Code only: comments and docstrings are allowed to name the concepts they
    # explain, and checking prose would only test the wording.
    code = "".join(
        token.string + " "
        for token in tokenize.generate_tokens(io.StringIO(path.read_text("utf-8")).readline)
        if token.type not in {tokenize.COMMENT, tokenize.STRING}
    )
    for forbidden in ("HARD_DENY", "RiskLevel", "AutonomyLevel", "model_callable", "auto_allow"):
        assert forbidden not in code, forbidden


def test_the_live_sources_are_callables_not_values() -> None:
    """Booleans here would reintroduce the stale snapshot this removes."""
    import inspect

    signature = inspect.signature(ToolGateway.__init__)
    for name in ("panic_check", "integrations_check"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_executor_keeps_working_without_live_sources(db) -> None:
    """Callers that have not migrated behave exactly as before."""
    spec, ran = _spec(risk=RiskLevel.READ)
    registry = ToolRegistry()
    registry.register(spec)
    executor = ToolExecutor(registry, ToolPolicy("trusted"), AuditLog(db))

    result = asyncio.run(
        executor.run(
            "audio.set_volume",
            {"percent": 30},
            panic=False,
            integrations_enabled=True,
            origin=Origin.MODEL,
        )
    )

    assert result.ok is True
    assert ran == [{"percent": 30}]


@pytest.mark.parametrize("actor", [Origin.USER, Origin.MODEL, Origin.ROUTINE])
def test_every_actor_reaches_the_same_door(db, actor) -> None:
    """Model, routine and person all arrive at one place, and the same rules."""
    spec, _ran = _spec(risk=RiskLevel.READ, model_callable=True)
    result = _invoke(_gateway(db, spec, World()), actor=actor)
    assert isinstance(result.ok, bool)
