from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from kiki.storage.database import SCHEMA_VERSION, Database
from kiki.storage.workspace_repository import WorkspaceRepository
from kiki.workspaces.models import Workspace, WorkspaceError


def _workspace(path: Path, *, workspace_id: str = "workspace-1", name: str = "App") -> Workspace:
    return Workspace(
        id=workspace_id,
        display_name=name,
        canonical_path=str(path),
        remote_url="https://example.test/acme/app.git",
        active_branch="main",
        git_head="abc123",
        risk_profile="observe",
        created_at="2026-01-01T00:00:00+00:00",
        last_used_at="2026-01-01T00:00:00+00:00",
    )


def test_repository_roundtrip_and_snapshot_update(tmp_path: Path, db: Database) -> None:
    repository = WorkspaceRepository(db)
    record = _workspace(tmp_path / "app")

    assert repository.add(record) == record
    assert repository.get(record.id) == record
    assert repository.get_by_path(record.canonical_path) == record
    assert repository.list() == [record]

    repository.update_git_snapshot(
        record.id,
        branch="feature/safe-workspaces",
        head="def456",
        remote_url="https://example.test/acme/renamed.git",
    )
    updated = repository.get(record.id)
    assert updated is not None
    assert updated.active_branch == "feature/safe-workspaces"
    assert updated.git_head == "def456"
    assert updated.remote_url == "https://example.test/acme/renamed.git"


def test_unknown_workspace_updates_fail_closed(db: Database) -> None:
    repository = WorkspaceRepository(db)

    with pytest.raises(WorkspaceError) as touch_error:
        repository.touch("missing")
    assert touch_error.value.code == "not_registered"

    with pytest.raises(WorkspaceError) as snapshot_error:
        repository.update_git_snapshot(
            "missing",
            branch="main",
            head=None,
            remote_url=None,
        )
    assert snapshot_error.value.code == "not_registered"


def test_remove_preserves_session_history_and_allows_reregistration(
    tmp_path: Path,
    db: Database,
) -> None:
    repository = WorkspaceRepository(db)
    record = repository.add(_workspace(tmp_path / "app"))
    db.conn.execute(
        "INSERT INTO agent_sessions("
        "id, workspace_id, agent_name, task_text, status, permission_profile, kind, started_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "session-1",
            record.id,
            "fake-agent",
            "Plan erstellen",
            "finished",
            "observe",
            "plan",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    db.conn.commit()

    assert repository.remove(record.id) is True
    assert repository.get(record.id) is None
    assert repository.get_by_path(record.canonical_path) is None
    assert repository.list() == []
    stored = db.conn.execute(
        "SELECT registered FROM workspaces WHERE id = ?",
        (record.id,),
    ).fetchone()
    assert stored["registered"] == 0
    sessions = db.conn.execute(
        "SELECT COUNT(*) AS count FROM agent_sessions WHERE workspace_id = ?",
        (record.id,),
    ).fetchone()
    assert sessions["count"] == 1

    replacement = _workspace(
        tmp_path / "app",
        workspace_id="new-id-is-not-used",
        name="App erneut",
    )
    reactivated = repository.add(replacement)
    assert reactivated.id == record.id
    assert reactivated.display_name == "App erneut"
    assert repository.get(record.id) == reactivated


def test_v3_migration_adds_allowlist_state_and_purges_remote_credentials(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta(key, value) VALUES ('schema_version', '3');
        CREATE TABLE workspaces (
            id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            canonical_path TEXT NOT NULL UNIQUE,
            remote_url TEXT,
            active_branch TEXT,
            git_head TEXT,
            risk_profile TEXT NOT NULL DEFAULT 'observe',
            created_at TEXT NOT NULL,
            last_used_at TEXT NOT NULL
        );
        CREATE TABLE agent_sessions (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            agent_version TEXT,
            model_name TEXT,
            task_text TEXT NOT NULL,
            status TEXT NOT NULL,
            permission_profile TEXT NOT NULL,
            kind TEXT NOT NULL,
            git_branch_before TEXT,
            git_head_before TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            exit_code INTEGER,
            summary TEXT
        );
        CREATE TABLE test_runs (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            workspace_id TEXT NOT NULL,
            profile TEXT NOT NULL,
            argv_json TEXT NOT NULL,
            status TEXT NOT NULL,
            exit_code INTEGER,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            summary TEXT
        );
        INSERT INTO workspaces(
            id, display_name, canonical_path, remote_url, active_branch, git_head,
            risk_profile, created_at, last_used_at
        ) VALUES (
            'legacy', 'Legacy', '/tmp/legacy',
            'https://user:secret@example.test/repo.git', 'main', 'abc',
            'observe', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
        );
        """
    )
    conn.close()

    migrated = Database(path)
    row = migrated.conn.execute(
        "SELECT registered, remote_url FROM workspaces WHERE id = 'legacy'"
    ).fetchone()
    version = migrated.conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    session_columns = {
        column["name"] for column in migrated.conn.execute("PRAGMA table_info(agent_sessions)")
    }
    test_columns = {
        column["name"] for column in migrated.conn.execute("PRAGMA table_info(test_runs)")
    }

    assert int(version["value"]) == SCHEMA_VERSION
    assert row["registered"] == 1
    assert row["remote_url"] is None
    assert "plan_session_id" in session_columns
    assert {"output_text", "output_truncated"} <= test_columns
    migrated.close()


def test_v5_migration_resumes_after_first_column_was_already_added(tmp_path: Path) -> None:
    path = tmp_path / "partial.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta(key, value) VALUES ('schema_version', '4');
        CREATE TABLE workspaces (
            id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            canonical_path TEXT NOT NULL UNIQUE,
            remote_url TEXT,
            active_branch TEXT,
            git_head TEXT,
            risk_profile TEXT NOT NULL DEFAULT 'observe',
            created_at TEXT NOT NULL,
            last_used_at TEXT NOT NULL,
            registered INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE agent_sessions (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            agent_version TEXT,
            model_name TEXT,
            task_text TEXT NOT NULL,
            status TEXT NOT NULL,
            permission_profile TEXT NOT NULL,
            kind TEXT NOT NULL,
            git_branch_before TEXT,
            git_head_before TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            exit_code INTEGER,
            summary TEXT,
            plan_session_id TEXT REFERENCES agent_sessions(id)
        );
        CREATE TABLE test_runs (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            workspace_id TEXT NOT NULL,
            profile TEXT NOT NULL,
            argv_json TEXT NOT NULL,
            status TEXT NOT NULL,
            exit_code INTEGER,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            summary TEXT
        );
        """
    )
    conn.close()

    migrated = Database(path)
    columns = {column["name"] for column in migrated.conn.execute("PRAGMA table_info(test_runs)")}
    version = migrated.conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()

    assert {"output_text", "output_truncated"} <= columns
    assert int(version["value"]) == SCHEMA_VERSION
    migrated.close()
