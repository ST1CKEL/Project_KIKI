"""Fail-closed runners for later sandbox backends."""

from __future__ import annotations

from kiki.runners.process import ProcessHandle, RunnerError
from kiki.workspaces.models import Workspace


class UnavailableRunner:
    """Podman/Distrobox/SSH are not active in this phase."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def run_argv(
        self,
        argv: list[str],
        *,
        workspace: Workspace,
        extra_env: dict[str, str] | None = None,
    ) -> ProcessHandle:
        del argv, workspace, extra_env
        raise RunnerError(
            "unavailable",
            f"{self.name}-Runner ist in dieser Phase deaktiviert (fail closed, kein --privileged).",
        )
