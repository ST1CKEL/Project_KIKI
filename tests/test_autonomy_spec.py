"""Jarvis autonomy, tool by tool: the blanket says the level may, the spec
says which tools do.

The policy's level table stays untouched and keeps its own promises (Martin's
tests prove them). This layer sits where things actually run: an allowed,
model-initiated jarvis call that the spec does not cover becomes an approval
card over the same validated arguments. Nothing here can make anything run —
`sharpen` only turns an unattended run into a question.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from kiki.storage.database import Database
from kiki.tools.audit import AuditLog
from kiki.tools.autonomy import (
    JARVIS_SPEC,
    JARVIS_UNATTENDED_WRITES,
    NEVER_UNATTENDED,
    AutonomySpec,
    sharpen,
    spec_for,
)
from kiki.tools.executor import ToolExecutor
from kiki.tools.gateway import ToolGateway, ToolInvocation
from kiki.tools.policy import (
    AutonomyLevel,
    DecisionKind,
    Origin,
    PolicyDecision,
    RiskLevel,
    ToolPolicy,
)
from kiki.tools.registry import ToolRegistry, ToolSpec

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def _spec(name: str, risk: RiskLevel, *, auto_allow: bool = True) -> ToolSpec:
    return ToolSpec(
        name=name,
        title=name,
        description=f"{name} description",
        risk=risk,
        parameters=SCHEMA,
        handler=lambda _p: {"ran": name},
        effect=f"{name} effect",
        auto_allow=auto_allow,
        requires_integration=False,
        model_callable=True,
    )


def _allow(risk: RiskLevel) -> PolicyDecision:
    return PolicyDecision(
        kind=DecisionKind.ALLOW, reason="Vorlage", risk=risk, params={}
    )


def _gateway(
    tmp_path: Path, *specs: ToolSpec, autonomy: str = "jarvis"
) -> tuple[ToolGateway, ToolExecutor]:
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    executor = ToolExecutor(
        registry, ToolPolicy(autonomy), AuditLog(Database(tmp_path / "kiki.db"))
    )
    gateway = ToolGateway(
        executor,
        panic_check=lambda: False,
        integrations_check=lambda: True,
    )
    return gateway, executor


async def _invoke(gateway: ToolGateway, name: str, *, confirm: Any = None) -> Any:
    return await gateway.invoke(
        ToolInvocation(tool=name, arguments={}, actor=Origin.MODEL),
        confirm=confirm,
    )


# --- the shipped spec is data, reviewable in one place ------------------------


def test_the_jarvis_spec_names_its_tools_exactly():
    # The documented deal, and nothing more: machine power.
    assert JARVIS_UNATTENDED_WRITES == frozenset({"power.reboot", "power.poweroff"})
    # The second lock under the author veto: memory, routines, clipboard,
    # notification, notes. Even a flipped `auto_allow` does not open these.
    assert "memory_remember" in NEVER_UNATTENDED
    assert "memory_forget" in NEVER_UNATTENDED
    assert "routines.create" in NEVER_UNATTENDED
    assert "routines.delete" in NEVER_UNATTENDED
    assert "desktop.copy_text" in NEVER_UNATTENDED
    assert "desktop.show_notification" in NEVER_UNATTENDED
    assert "create_note" in NEVER_UNATTENDED
    # No tool is both declared capable and never unattended.
    assert not (JARVIS_UNATTENDED_WRITES & NEVER_UNATTENDED)


def test_an_unknown_level_names_no_tools():
    empty = spec_for("kein_level")
    assert empty.unattended_writes == frozenset()
    assert empty.never_unattended == frozenset()


# --- sharpen: pure, one direction ---------------------------------------------


def test_a_declared_write_keeps_the_documented_deal():
    decision = sharpen(
        _allow(RiskLevel.WRITE),
        name="power.reboot",
        spec=_spec("power.reboot", RiskLevel.WRITE),
        origin=Origin.MODEL,
        autonomy=JARVIS_SPEC,
    )
    assert decision.kind is DecisionKind.ALLOW


@pytest.mark.parametrize("risk", [RiskLevel.WRITE, RiskLevel.EXTERNAL])
def test_an_undeclared_write_gets_the_card(risk):
    decision = sharpen(
        _allow(risk),
        name=f"future_{risk.value}_tool",
        spec=_spec(f"future_{risk.value}_tool", risk),
        origin=Origin.MODEL,
        autonomy=JARVIS_SPEC,
    )
    assert decision.kind is DecisionKind.CONFIRM
    assert decision.params == {}
    assert "Freigabe" in decision.reason


def test_never_unattended_holds_even_when_declared_and_auto_allowed():
    # The hostile case: someone lists a locked tool as capable AND flips its
    # `auto_allow`. The second lock answers.
    hostile = AutonomySpec(
        level=AutonomyLevel.JARVIS,
        unattended_writes=frozenset({"memory_remember"}),
        never_unattended=NEVER_UNATTENDED,
    )
    decision = sharpen(
        _allow(RiskLevel.WRITE),
        name="memory_remember",
        spec=_spec("memory_remember", RiskLevel.WRITE, auto_allow=True),
        origin=Origin.MODEL,
        autonomy=hostile,
    )
    assert decision.kind is DecisionKind.CONFIRM


def test_lower_levels_are_not_sharpened():
    # Defense in depth, proven with the strongest case: even a WRITE allow --
    # which today's policy cannot produce below jarvis, but sharpen must not
    # lean on that -- passes through untouched. The jarvis spec has no
    # charter on other levels; if the policy's table ever widens one, this
    # layer does not silently follow.
    for level in ("strict", "balanced", "trusted"):
        for name in ("power.reboot", "future_write"):
            decision = _allow(RiskLevel.WRITE)
            out = sharpen(
                decision,
                name=name,
                spec=_spec(name, RiskLevel.WRITE),
                origin=Origin.MODEL,
                autonomy=spec_for(level),
            )
            assert out is decision, f"{level} must not be sharpened"


def test_denials_and_other_origins_pass_through_untouched():
    deny = PolicyDecision(kind=DecisionKind.DENY, reason="verboten")
    assert sharpen(deny, name="x", spec=None, origin=Origin.MODEL, autonomy=JARVIS_SPEC) is deny
    user = _allow(RiskLevel.WRITE)
    for origin in (Origin.USER, Origin.ROUTINE):
        decision = sharpen(
            user,
            name="power.reboot",
            spec=_spec("power.reboot", RiskLevel.WRITE),
            origin=origin,
            autonomy=JARVIS_SPEC,
        )
        assert decision is user


# --- end to end: the executor asks before it runs ------------------------------


def test_jarvis_runs_an_undeclared_write_only_with_a_card(tmp_path):
    counter = {"runs": 0}

    def _handler(_params: dict[str, Any]) -> dict[str, Any]:
        counter["runs"] += 1
        return {"ran": True}

    undeclared = ToolSpec(
        name="future_write",
        title="Schreiben",
        description="Ein künftiges Schreibwerkzeug.",
        risk=RiskLevel.WRITE,
        parameters=SCHEMA,
        handler=_handler,
        effect="Schreibt.",
        auto_allow=True,
        requires_integration=False,
        model_callable=True,
    )
    gateway, _executor = _gateway(tmp_path, undeclared)

    # No card on screen: nothing runs.
    result = asyncio.run(_invoke(gateway, "future_write"))
    assert result.ok is False
    assert counter["runs"] == 0

    # An approved card runs it exactly once.
    async def _approve(_preview: Any) -> bool:
        return True

    result = asyncio.run(_invoke(gateway, "future_write", confirm=_approve))
    assert result.ok is True
    assert counter["runs"] == 1


def test_jarvis_runs_the_declared_power_tools_unattended(tmp_path):
    counter = {"runs": 0}

    def _handler(_params: dict[str, Any]) -> dict[str, Any]:
        counter["runs"] += 1
        return {"ran": True}

    reboot = ToolSpec(
        name="power.reboot",
        title="Neustart",
        description="Startet die Maschine neu.",
        risk=RiskLevel.WRITE,
        parameters=SCHEMA,
        handler=_handler,
        effect="Startet neu.",
        auto_allow=True,
        requires_integration=False,
        model_callable=True,
    )
    gateway, _executor = _gateway(tmp_path, reboot)
    result = asyncio.run(_invoke(gateway, "power.reboot"))
    assert result.ok is True
    assert counter["runs"] == 1


def test_jarvis_still_denies_hard_deny_and_panic(tmp_path):
    read_tool = _spec("status_x", RiskLevel.READ)
    gateway, _executor = _gateway(tmp_path, read_tool)
    result = asyncio.run(_invoke(gateway, "run_shell"))
    assert result.ok is False
    assert result.decision.kind is DecisionKind.DENY

    panicky, _ex = _gateway(tmp_path, read_tool)
    panicky_executor = panicky.executor
    panicky_executor_gateway = ToolGateway(
        panicky_executor,
        panic_check=lambda: True,
        integrations_check=lambda: True,
    )
    result = asyncio.run(_invoke(panicky_executor_gateway, "status_x"))
    assert result.ok is False
    assert result.decision.kind is DecisionKind.DENY


def test_the_audit_names_the_real_decision(tmp_path):
    undeclared = _spec("future_write", RiskLevel.WRITE)
    gateway, executor = _gateway(tmp_path, undeclared)
    asyncio.run(_invoke(gateway, "future_write"))

    entries = [e for e in executor.audit.recent() if e.tool == "future_write"]
    assert entries, "the sharpened call must be audited"
    decisions = {e.decision for e in entries}
    # The card request is recorded as such, and an "allow" that was quietly
    # rethought afterwards never appears. The final "denied" (no card on
    # screen) is the honest outcome of the same decision.
    assert "confirm" in decisions
    assert "allow" not in decisions
