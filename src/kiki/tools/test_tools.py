"""Declared test profiles. The parameter is a name, never a shell string."""

from __future__ import annotations

from kiki.runners.local import TEST_PROFILES
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec

_EMPTY = lambda params: {"ok": False, "error": "handler not bound"}  # noqa: E731


def tests_run_profile_spec() -> ToolSpec:
    return ToolSpec(
        name="tests.run_profile",
        title="Testprofil ausführen",
        description="Startet ein fest zugeordnetes Testkommando im Workspace.",
        risk=RiskLevel.WRITE,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["workspace_id", "profile"],
            "properties": {
                "workspace_id": {"type": "string"},
                "profile": {"type": "string", "enum": sorted(TEST_PROFILES)},
                "timeout_seconds": {"type": "integer"},
                "approval_id": {"type": "string"},
            },
        },
        handler=_EMPTY,
        effect="Führt nur das konfigurierte Profil-Argv aus. Kein freies Kommando.",
        auto_allow=False,
        requires_integration=False,
        allowed_profiles=("develop",),
    )
