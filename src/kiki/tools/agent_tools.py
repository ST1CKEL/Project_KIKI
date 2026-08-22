"""Declared agent tools. No free-form commands."""

from __future__ import annotations

from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec

_EMPTY = lambda params: {"ok": False, "error": "handler not bound"}  # noqa: E731


def agent_plan_spec() -> ToolSpec:
    return ToolSpec(
        name="agent.plan",
        title="Agentenplan",
        description="Startet eine plan-only OpenCode-Session im registrierten Workspace.",
        risk=RiskLevel.READ,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["workspace_id", "task"],
            "properties": {
                "workspace_id": {"type": "string"},
                "task": {"type": "string"},
                "model": {"type": "string"},
                "profile": {"type": "string", "enum": ["observe", "develop"]},
            },
        },
        handler=_EMPTY,
        effect="Liest das Repository und erzeugt einen Plan. Keine Dateiänderungen.",
        auto_allow=True,
        requires_integration=False,
        allowed_profiles=("observe", "develop"),
    )


def agent_start_spec() -> ToolSpec:
    return ToolSpec(
        name="agent.start_implementation",
        title="Umsetzung starten",
        description="Startet eine Umsetzungs-Session. Nur develop, nur mit Einzelfreigabe.",
        risk=RiskLevel.WRITE,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["workspace_id", "task"],
            "properties": {
                "workspace_id": {"type": "string"},
                "task": {"type": "string"},
                "model": {"type": "string"},
                "profile": {"type": "string", "enum": ["develop"]},
                "plan_session_id": {"type": "string"},
                "approval_id": {"type": "string"},
            },
        },
        handler=_EMPTY,
        effect="Darf Dateien im Workspace ändern. Kein sudo, kein Push, kein Home-Zugriff.",
        auto_allow=False,
        requires_integration=False,
        allowed_profiles=("develop",),
    )


def agent_stop_spec() -> ToolSpec:
    return ToolSpec(
        name="agent.stop",
        title="Agent stoppen",
        description="Beendet die Prozessgruppe einer laufenden Session.",
        risk=RiskLevel.CONTROL,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["session_id"],
            "properties": {"session_id": {"type": "string"}},
        },
        handler=_EMPTY,
        effect="Sendet SIGTERM an die Prozessgruppe.",
        auto_allow=True,
        requires_integration=False,
        allowed_profiles=("observe", "develop"),
        allowed_in_panic=True,
    )


def agent_availability_spec() -> ToolSpec:
    return ToolSpec(
        name="agent.availability_check",
        title="OpenCode prüfen",
        description="Prüft, ob die OpenCode-Binary verfügbar ist.",
        risk=RiskLevel.READ,
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_EMPTY,
        effect="Führt nur einen Versionsaufruf aus.",
        auto_allow=True,
        requires_integration=False,
        allowed_profiles=("observe", "develop"),
    )


def git_status_spec() -> ToolSpec:
    return ToolSpec(
        name="git.status",
        title="Git-Status",
        description="Liest Branch und Dirty-Status eines registrierten Workspace.",
        risk=RiskLevel.READ,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["workspace_id"],
            "properties": {"workspace_id": {"type": "string"}},
        },
        handler=_EMPTY,
        effect="Nur git status/rev-parse, keine Mutation.",
        auto_allow=True,
        requires_integration=False,
        allowed_profiles=("observe", "develop"),
    )
