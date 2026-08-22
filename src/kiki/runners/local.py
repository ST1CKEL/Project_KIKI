"""Run a fixed argv inside a registered workspace directory."""

from __future__ import annotations

from pathlib import Path

from kiki.runners.process import ProcessHandle, RunnerError, sanitized_env, spawn
from kiki.workspaces.models import Workspace

TEST_PROFILES: dict[str, tuple[str, ...]] = {
    "python_pytest": ("python3", "-m", "pytest", "-q"),
    "node_npm_test": ("npm", "test"),
    "node_pnpm_test": ("pnpm", "test"),
    "rust_cargo_test": ("cargo", "test"),
    "go_test": ("go", "test", "./..."),
}

_MIN_TIMEOUT = 30
_MAX_TIMEOUT = 1800


class LocalWorkspaceRunner:
    def __init__(
        self,
        *,
        profiles: dict[str, tuple[str, ...]] | None = None,
        min_timeout: int = _MIN_TIMEOUT,
        max_timeout: int = _MAX_TIMEOUT,
    ) -> None:
        self._profiles = dict(profiles or TEST_PROFILES)
        self._min_timeout = min_timeout
        self._max_timeout = max_timeout

    def profile_argv(self, name: str) -> list[str]:
        argv = self._profiles.get(name)
        if argv is None:
            raise RunnerError("unknown_profile", f"Unbekanntes Testprofil: {name}")
        return list(argv)

    def clamp_timeout(self, value: int | float | None) -> float:
        if value is None:
            return float(max(self._min_timeout, min(self._max_timeout, 300)))
        return float(max(self._min_timeout, min(self._max_timeout, int(value))))

    async def run_argv(
        self,
        argv: list[str],
        *,
        workspace: Workspace,
        extra_env: dict[str, str] | None = None,
    ) -> ProcessHandle:
        cwd = Path(workspace.canonical_path)
        if not cwd.is_dir():
            raise RunnerError("not_a_directory", f"Workspace fehlt: {cwd}")
        env = sanitized_env(home=str(cwd), extra=extra_env)
        return await spawn(argv, cwd=cwd, env=env)

    async def run_profile(
        self,
        profile: str,
        *,
        workspace: Workspace,
    ) -> ProcessHandle:
        return await self.run_argv(self.profile_argv(profile), workspace=workspace)
