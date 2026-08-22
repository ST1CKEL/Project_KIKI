from __future__ import annotations

import asyncio

from kiki.tools.agent_tools import agent_plan_spec, agent_start_spec, agent_stop_spec
from kiki.tools.policy import DecisionKind, RiskLevel, ToolPolicy
from kiki.tools.registry import ToolSpec
from kiki.tools.test_tools import tests_run_profile_spec as run_profile_tool


def _spec(**kwargs) -> ToolSpec:
    values = dict(
        name="status_disk",
        title="Speicher",
        description="read",
        risk=RiskLevel.READ,
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda params: {"ok": True},
        effect="Liest freien Speicher.",
        auto_allow=True,
        requires_integration=True,
    )
    values.update(kwargs)
    return ToolSpec(**values)


def test_unknown_is_denied() -> None:
    decision = ToolPolicy().evaluate(
        name="launch_missiles",
        params={},
        spec=None,
        panic=False,
        integrations_enabled=True,
    )
    assert decision.kind is DecisionKind.DENY


def test_hard_deny_shell() -> None:
    fake = _spec(name="run_shell", risk=RiskLevel.READ, auto_allow=True)
    decision = ToolPolicy().evaluate(
        name="run_shell",
        params={"cmd": "rm -rf /"},
        spec=fake,
        panic=False,
        integrations_enabled=True,
    )
    assert decision.kind is DecisionKind.DENY
    assert "verboten" in decision.reason


def test_read_auto_allow() -> None:
    decision = ToolPolicy().evaluate(
        name="status_disk",
        params={},
        spec=_spec(),
        panic=False,
        integrations_enabled=True,
    )
    assert decision.kind is DecisionKind.ALLOW


def test_write_requires_confirm() -> None:
    spec = _spec(name="restart_service", risk=RiskLevel.WRITE, auto_allow=False)
    decision = ToolPolicy().evaluate(
        name="restart_service",
        params={},
        spec=spec,
        panic=False,
        integrations_enabled=True,
    )
    assert decision.kind is DecisionKind.CONFIRM


def test_panic_denies_even_reads() -> None:
    decision = ToolPolicy().evaluate(
        name="status_disk",
        params={},
        spec=_spec(),
        panic=True,
        integrations_enabled=True,
    )
    assert decision.kind is DecisionKind.DENY


def test_panic_still_allows_explicit_emergency_stop() -> None:
    decision = ToolPolicy().evaluate(
        name="agent.stop",
        params={"session_id": "session-1"},
        spec=agent_stop_spec(),
        panic=True,
        integrations_enabled=False,
        profile="observe",
    )
    assert decision.kind is DecisionKind.ALLOW
    assert decision.risk is RiskLevel.CONTROL


def test_unknown_params_denied() -> None:
    decision = ToolPolicy().evaluate(
        name="status_disk",
        params={"oops": 1},
        spec=_spec(),
        panic=False,
        integrations_enabled=True,
    )
    assert decision.kind is DecisionKind.DENY
    assert "unknown" in decision.reason


def test_executor_confirm_cancel(tools_env, db) -> None:
    registry, executor = tools_env
    ran = {"n": 0}

    def handler(_params):
        ran["n"] += 1
        return {"did": True}

    registry.register(
        _spec(name="open_browser", risk=RiskLevel.EXTERNAL, auto_allow=False, handler=handler)
    )

    async def no(_preview):
        return False

    result = asyncio.run(
        executor.run(
            "open_browser",
            {},
            panic=False,
            integrations_enabled=True,
            confirm=no,
        )
    )
    assert result.ok is False
    assert ran["n"] == 0
    rows = db.conn.execute("SELECT decision FROM audit_log ORDER BY id").fetchall()
    assert "cancelled" in {r["decision"] for r in rows}


def test_observe_allows_plan_not_tests() -> None:
    policy = ToolPolicy()
    plan = policy.evaluate(
        name="agent.plan",
        params={"workspace_id": "w", "task": "x", "profile": "observe"},
        spec=agent_plan_spec(),
        panic=False,
        integrations_enabled=True,
        profile="observe",
    )
    assert plan.kind is DecisionKind.ALLOW
    tests = policy.evaluate(
        name="tests.run_profile",
        params={"workspace_id": "w", "profile": "python_pytest"},
        spec=run_profile_tool(),
        panic=False,
        integrations_enabled=True,
        profile="observe",
    )
    assert tests.kind is DecisionKind.DENY
    start = policy.evaluate(
        name="agent.start_implementation",
        params={"workspace_id": "w", "task": "x", "profile": "develop"},
        spec=agent_start_spec(),
        panic=False,
        integrations_enabled=True,
        profile="develop",
    )
    assert start.kind is DecisionKind.CONFIRM
