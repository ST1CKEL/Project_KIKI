"""Runner protocol. Active: LocalWorkspaceRunner. Others fail closed."""

from __future__ import annotations

from typing import Protocol

from kiki.runners.process import ProcessHandle


class WorkspaceRunner(Protocol):
    async def run_argv(
        self,
        argv: list[str],
        *,
        cwd: str,
        timeout: float,
    ) -> ProcessHandle: ...
