"""SQLite persistence for registered coding workspaces."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from threading import Lock

from kiki.storage.database import Database
from kiki.workspaces.models import Workspace, WorkspaceError


def _now() -> str:
    return datetime.now(UTC).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


class WorkspaceRepository:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._lock = Lock()

    def add(self, workspace: Workspace) -> Workspace:
        with self._lock:
            existing = self._db.conn.execute(
                "SELECT id, registered FROM workspaces WHERE canonical_path = ?",
                (workspace.canonical_path,),
            ).fetchone()
            if existing and bool(existing["registered"]):
                raise WorkspaceError(
                    "already_registered",
                    f"Workspace bereits registriert: {workspace.canonical_path}",
                )
            if existing:
                workspace_id = str(existing["id"])
                self._db.conn.execute(
                    "UPDATE workspaces SET display_name = ?, remote_url = ?, active_branch = ?, "
                    "git_head = ?, risk_profile = ?, last_used_at = ?, registered = 1 WHERE id = ?",
                    (
                        workspace.display_name,
                        workspace.remote_url,
                        workspace.active_branch,
                        workspace.git_head,
                        workspace.risk_profile,
                        workspace.last_used_at,
                        workspace_id,
                    ),
                )
                row = self._db.conn.execute(
                    "SELECT * FROM workspaces WHERE id = ?",
                    (workspace_id,),
                ).fetchone()
                self._db.conn.commit()
                return _from_row(row)
            self._db.conn.execute(
                "INSERT INTO workspaces("
                "id, display_name, canonical_path, remote_url, active_branch, git_head, "
                "risk_profile, created_at, last_used_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    workspace.id,
                    workspace.display_name,
                    workspace.canonical_path,
                    workspace.remote_url,
                    workspace.active_branch,
                    workspace.git_head,
                    workspace.risk_profile,
                    workspace.created_at,
                    workspace.last_used_at,
                ),
            )
            self._db.conn.commit()
        return workspace

    def get(self, workspace_id: str) -> Workspace | None:
        with self._lock:
            row = self._db.conn.execute(
                "SELECT * FROM workspaces WHERE id = ? AND registered = 1",
                (workspace_id,),
            ).fetchone()
        return _from_row(row) if row else None

    def get_by_path(self, canonical_path: str) -> Workspace | None:
        with self._lock:
            row = self._db.conn.execute(
                "SELECT * FROM workspaces WHERE canonical_path = ? AND registered = 1",
                (canonical_path,),
            ).fetchone()
        return _from_row(row) if row else None

    def list(self) -> list[Workspace]:
        with self._lock:
            rows = self._db.conn.execute(
                "SELECT * FROM workspaces WHERE registered = 1 ORDER BY last_used_at DESC, display_name ASC"
            ).fetchall()
        return [_from_row(row) for row in rows]

    def touch(self, workspace_id: str) -> None:
        with self._lock:
            cur = self._db.conn.execute(
                "UPDATE workspaces SET last_used_at = ? WHERE id = ? AND registered = 1",
                (_now(), workspace_id),
            )
            if cur.rowcount == 0:
                self._db.conn.rollback()
                raise WorkspaceError("not_registered", f"Unbekannte Workspace-ID: {workspace_id}")
            self._db.conn.commit()

    def update_git_snapshot(
        self,
        workspace_id: str,
        *,
        branch: str | None,
        head: str | None,
        remote_url: str | None,
    ) -> None:
        with self._lock:
            cur = self._db.conn.execute(
                "UPDATE workspaces SET active_branch = ?, git_head = ?, remote_url = ?, "
                "last_used_at = ? WHERE id = ? AND registered = 1",
                (branch, head, remote_url, _now(), workspace_id),
            )
            if cur.rowcount == 0:
                self._db.conn.rollback()
                raise WorkspaceError("not_registered", f"Unbekannte Workspace-ID: {workspace_id}")
            self._db.conn.commit()

    def remove(self, workspace_id: str) -> bool:
        """Remove from the allowlist while retaining referenced local history."""
        with self._lock:
            cur = self._db.conn.execute(
                "UPDATE workspaces SET registered = 0 WHERE id = ? AND registered = 1",
                (workspace_id,),
            )
            self._db.conn.commit()
            return cur.rowcount > 0


def _from_row(row: object) -> Workspace:
    data = dict(row)  # type: ignore[arg-type]
    return Workspace(
        id=str(data["id"]),
        display_name=str(data["display_name"]),
        canonical_path=str(data["canonical_path"]),
        remote_url=data["remote_url"],
        active_branch=data["active_branch"],
        git_head=data["git_head"],
        risk_profile=str(data["risk_profile"]),
        created_at=str(data["created_at"]),
        last_used_at=str(data["last_used_at"]),
    )
