from __future__ import annotations

from kiki.agents.models import AgentEventType, PermissionProfile, arguments_hash, parse_plan_text
from kiki.tools.agent_tools import agent_start_spec
from kiki.tools.policy import DecisionKind, ToolPolicy
from kiki.tools.test_tools import tests_run_profile_spec as run_profile_tool


def test_event_types_cover_spec() -> None:
    needed = {
        "session_started",
        "status_changed",
        "message",
        "plan",
        "tool_request",
        "tool_result",
        "file_change",
        "diff_available",
        "test_started",
        "test_output",
        "test_finished",
        "approval_required",
        "error",
        "session_finished",
    }
    assert needed <= {item.value for item in AgentEventType}


def test_arguments_hash_is_stable() -> None:
    assert arguments_hash({"b": 2, "a": 1}) == arguments_hash({"a": 1, "b": 2})
    assert arguments_hash({"a": 1}) != arguments_hash({"a": 2})


def test_parse_plan_text_keeps_raw() -> None:
    plan = parse_plan_text("1. foo\n2. bar\n")
    assert "foo" in plan.raw
    assert plan.steps[0].startswith("1.")


def test_observe_cannot_start_implementation() -> None:
    decision = ToolPolicy().evaluate(
        name="agent.start_implementation",
        params={"workspace_id": "w", "task": "do it", "profile": "develop"},
        spec=agent_start_spec(),
        panic=False,
        integrations_enabled=True,
        profile=PermissionProfile.OBSERVE.value,
    )
    assert decision.kind is DecisionKind.DENY


def test_develop_start_requires_confirm() -> None:
    decision = ToolPolicy().evaluate(
        name="agent.start_implementation",
        params={"workspace_id": "w", "task": "do it", "profile": "develop"},
        spec=agent_start_spec(),
        panic=False,
        integrations_enabled=True,
        profile="develop",
    )
    assert decision.kind is DecisionKind.CONFIRM


def test_operator_is_disabled() -> None:
    decision = ToolPolicy().evaluate(
        name="agent.start_implementation",
        params={"workspace_id": "w", "task": "do it", "profile": "develop"},
        spec=agent_start_spec(),
        panic=False,
        integrations_enabled=True,
        profile="operator",
    )
    assert decision.kind is DecisionKind.DENY
    assert "operator" in decision.reason


def test_free_command_param_denied() -> None:
    decision = ToolPolicy().evaluate(
        name="tests.run_profile",
        params={"workspace_id": "w", "profile": "python_pytest", "cmd": "rm -rf /"},
        spec=run_profile_tool(),
        panic=False,
        integrations_enabled=True,
        profile="develop",
    )
    assert decision.kind is DecisionKind.DENY
    assert "unknown" in decision.reason
