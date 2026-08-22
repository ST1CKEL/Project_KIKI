"""SQLite connection with WAL and schema migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 6

_MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_messages_conversation
        ON messages(conversation_id, created_at);
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'explicit'
    );
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        tool TEXT NOT NULL,
        params_json TEXT NOT NULL,
        decision TEXT NOT NULL,
        result TEXT,
        error TEXT
    );
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """,
    2: """
    CREATE TABLE IF NOT EXISTS workspaces (
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
    CREATE INDEX IF NOT EXISTS idx_workspaces_last_used
        ON workspaces(last_used_at DESC);
    """,
    3: """
    CREATE TABLE IF NOT EXISTS agent_sessions (
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
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
    );
    CREATE INDEX IF NOT EXISTS idx_agent_sessions_workspace
        ON agent_sessions(workspace_id, started_at);
    CREATE TABLE IF NOT EXISTS agent_events (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        ts TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES agent_sessions(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_agent_events_session
        ON agent_events(session_id, ts);
    CREATE TABLE IF NOT EXISTS approval_requests (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        tool TEXT NOT NULL,
        params_json TEXT NOT NULL,
        params_hash TEXT NOT NULL,
        risk_class TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS approval_decisions (
        id TEXT PRIMARY KEY,
        approval_id TEXT NOT NULL UNIQUE,
        decided_at TEXT NOT NULL,
        approved INTEGER NOT NULL,
        actor TEXT NOT NULL,
        consumed INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (approval_id) REFERENCES approval_requests(id)
    );
    CREATE TABLE IF NOT EXISTS test_runs (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        workspace_id TEXT NOT NULL,
        profile TEXT NOT NULL,
        argv_json TEXT NOT NULL,
        status TEXT NOT NULL,
        exit_code INTEGER,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        summary TEXT,
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
    );
    CREATE TABLE IF NOT EXISTS agent_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        session_id TEXT,
        actor TEXT NOT NULL,
        event_type TEXT NOT NULL,
        risk_class TEXT NOT NULL,
        requested_action TEXT NOT NULL,
        resolved_arguments_hash TEXT NOT NULL,
        policy_decision TEXT NOT NULL,
        approval_id TEXT,
        result_status TEXT,
        result_summary TEXT
    );
    """,
    4: """
    ALTER TABLE workspaces
        ADD COLUMN registered INTEGER NOT NULL DEFAULT 1
        CHECK (registered IN (0, 1));
    CREATE INDEX IF NOT EXISTS idx_workspaces_registered_last_used
        ON workspaces(registered, last_used_at DESC);
    -- Earlier versions stored origin URLs verbatim. Purge that optional metadata
    -- once so credentials from legacy HTTPS remotes cannot survive migration.
    UPDATE workspaces SET remote_url = NULL;
    """,
    5: """
    ALTER TABLE agent_sessions
        ADD COLUMN plan_session_id TEXT REFERENCES agent_sessions(id);
    CREATE INDEX IF NOT EXISTS idx_agent_sessions_plan
        ON agent_sessions(plan_session_id);
    ALTER TABLE test_runs
        ADD COLUMN output_text TEXT;
    ALTER TABLE test_runs
        ADD COLUMN output_truncated INTEGER NOT NULL DEFAULT 0
        CHECK (output_truncated IN (0, 1));
    """,
    6: """
    ALTER TABLE memories ADD COLUMN updated_at TEXT;
    UPDATE memories SET updated_at = created_at WHERE updated_at IS NULL;
    CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
    """,
}


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def _migrate(self) -> None:
        self._conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        current = int(row["value"]) if row else 0
        for version in sorted(_MIGRATIONS):
            if version > current:
                self._conn.executescript(self._resumable_migration_sql(version))
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                    (str(version),),
                )
                current = version
        self._conn.commit()

    def _resumable_migration_sql(self, version: int) -> str:
        if version == 4:
            columns = self._table_columns("workspaces")
            add_registered = ""
            if "registered" not in columns:
                add_registered = (
                    "ALTER TABLE workspaces ADD COLUMN registered INTEGER NOT NULL DEFAULT 1 "
                    "CHECK (registered IN (0, 1));"
                )
            return "\n".join(
                (
                    add_registered,
                    "CREATE INDEX IF NOT EXISTS idx_workspaces_registered_last_used "
                    "ON workspaces(registered, last_used_at DESC);",
                    "UPDATE workspaces SET remote_url = NULL;",
                )
            )
        if version == 5:
            session_columns = self._table_columns("agent_sessions")
            test_columns = self._table_columns("test_runs")
            statements: list[str] = []
            if "plan_session_id" not in session_columns:
                statements.append(
                    "ALTER TABLE agent_sessions ADD COLUMN plan_session_id TEXT "
                    "REFERENCES agent_sessions(id);"
                )
            statements.append(
                "CREATE INDEX IF NOT EXISTS idx_agent_sessions_plan ON agent_sessions(plan_session_id);"
            )
            if "output_text" not in test_columns:
                statements.append("ALTER TABLE test_runs ADD COLUMN output_text TEXT;")
            if "output_truncated" not in test_columns:
                statements.append(
                    "ALTER TABLE test_runs ADD COLUMN output_truncated INTEGER NOT NULL DEFAULT 0 "
                    "CHECK (output_truncated IN (0, 1));"
                )
            return "\n".join(statements)
        if version == 6:
            statements = [
                # A database stamped past v1 without this table would otherwise
                # be unmigratable; recreate it before altering.
                "CREATE TABLE IF NOT EXISTS memories ("
                "id TEXT PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL, "
                "created_at TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'explicit');"
            ]
            if "updated_at" not in self._table_columns("memories"):
                statements.append("ALTER TABLE memories ADD COLUMN updated_at TEXT;")
            statements.append(
                "UPDATE memories SET updated_at = created_at WHERE updated_at IS NULL;"
            )
            statements.append(
                "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);"
            )
            return "\n".join(statements)
        return _MIGRATIONS[version]

    def _table_columns(self, table: str) -> set[str]:
        return {str(row["name"]) for row in self._conn.execute(f"PRAGMA table_info({table})")}

    def close(self) -> None:
        self._conn.close()
