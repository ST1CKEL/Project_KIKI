"""Workspace records and read-only Git snapshots. No process control."""

from __future__ import annotations

from dataclasses import dataclass

VALID_RISK_PROFILES: tuple[str, ...] = ("observe", "develop", "operator")


class WorkspaceError(Exception):
    """Validation or registry failure. Fail closed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Workspace:
    id: str
    display_name: str
    canonical_path: str
    remote_url: str | None
    active_branch: str | None
    git_head: str | None
    risk_profile: str
    created_at: str
    last_used_at: str


@dataclass(frozen=True)
class GitSnapshot:
    toplevel: str
    branch: str | None
    head: str | None
    detached: bool
    dirty: bool
    untracked: bool
    remote_url: str | None


@dataclass(frozen=True)
class GitDiff:
    stat: str
    patch: str
    truncated: bool
