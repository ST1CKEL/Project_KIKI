"""Register, inspect and forget workspaces. Disk repos are never deleted."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from kiki.storage.workspace_repository import WorkspaceRepository, new_id
from kiki.workspaces.git_service import inspect_git
from kiki.workspaces.models import GitSnapshot, Workspace, WorkspaceError
from kiki.workspaces.validator import (
    expand_allowed_roots,
    validate_candidate,
    validate_risk_profile,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class WorkspaceRegistry:
    def __init__(self, repo: WorkspaceRepository, *, allowed_roots: Sequence[str]) -> None:
        self._repo = repo
        self._allowed_roots = expand_allowed_roots(allowed_roots)
        if not self._allowed_roots:
            raise WorkspaceError("no_roots", "Keine erlaubten Workspace-Roots konfiguriert.")

    def register(
        self,
        raw_path: str,
        *,
        display_name: str | None = None,
        risk_profile: str = "observe",
    ) -> Workspace:
        canonical = validate_candidate(
            raw_path,
            allowed_roots=self._allowed_roots,
            require_git_root=False,
        )
        snapshot = inspect_git(canonical)
        canonical = validate_candidate(
            raw_path,
            allowed_roots=self._allowed_roots,
            require_git_root=True,
            git_toplevel=snapshot.toplevel,
        )
        profile = validate_risk_profile(risk_profile)
        existing = self._repo.get_by_path(str(canonical))
        if existing:
            raise WorkspaceError(
                "already_registered",
                f"Workspace bereits registriert: {canonical}",
            )
        name = (display_name or "").strip() or canonical.name
        ts = _now()
        record = Workspace(
            id=new_id(),
            display_name=name,
            canonical_path=str(canonical),
            remote_url=snapshot.remote_url,
            active_branch=snapshot.branch,
            git_head=snapshot.head,
            risk_profile=profile,
            created_at=ts,
            last_used_at=ts,
        )
        return self._repo.add(record)

    def list(self) -> list[Workspace]:
        return self._repo.list()

    def get(self, workspace_id: str) -> Workspace | None:
        return self._repo.get(workspace_id)

    def touch(self, workspace_id: str) -> None:
        self.require(workspace_id)
        self._repo.touch(workspace_id)

    def require(self, workspace_id: str) -> Workspace:
        found, _snapshot = self._require_current(workspace_id)
        return found

    def require_path(self, raw_path: str) -> Workspace:
        canonical = validate_candidate(
            raw_path,
            allowed_roots=self._allowed_roots,
            require_git_root=False,
        )
        found = self._repo.get_by_path(str(canonical))
        if found is None:
            raise WorkspaceError("not_registered", f"Nicht registriert: {canonical}")
        current, _snapshot = self._require_current(found.id)
        return current

    def inspect(self, workspace_id: str) -> tuple[Workspace, GitSnapshot]:
        record, snapshot = self._require_current(workspace_id)
        self._repo.update_git_snapshot(
            record.id,
            branch=snapshot.branch,
            head=snapshot.head,
            remote_url=snapshot.remote_url,
        )
        updated = self._repo.get(workspace_id)
        if updated is None:
            raise WorkspaceError("not_registered", f"Unbekannte Workspace-ID: {workspace_id}")
        return updated, snapshot

    def remove(self, workspace_id: str) -> Path:
        """Unregister. Returns the path; the Git repo on disk stays untouched."""
        record = self._repo.get(workspace_id)
        if record is None:
            raise WorkspaceError("not_registered", f"Unbekannte Workspace-ID: {workspace_id}")
        path = Path(record.canonical_path)
        if not self._repo.remove(workspace_id):
            raise WorkspaceError("not_registered", f"Unbekannte Workspace-ID: {workspace_id}")
        return path

    def _require_current(self, workspace_id: str) -> tuple[Workspace, GitSnapshot]:
        """Revalidate the persisted path and Git root before any trusted use."""
        record = self._repo.get(workspace_id)
        if record is None:
            raise WorkspaceError("not_registered", f"Unbekannte Workspace-ID: {workspace_id}")
        canonical = validate_candidate(
            record.canonical_path,
            allowed_roots=self._allowed_roots,
            require_git_root=False,
        )
        if str(canonical) != record.canonical_path:
            raise WorkspaceError(
                "workspace_changed",
                "Der registrierte Workspace-Pfad zeigt nicht mehr auf das freigegebene Verzeichnis.",
            )
        snapshot = inspect_git(canonical)
        validate_candidate(
            record.canonical_path,
            allowed_roots=self._allowed_roots,
            require_git_root=True,
            git_toplevel=snapshot.toplevel,
        )
        return record, snapshot
