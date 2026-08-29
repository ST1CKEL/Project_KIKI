from __future__ import annotations

import os
from pathlib import Path

import pytest

from kiki.config.settings import load_settings
from kiki.storage.chat_repository import ChatRepository
from kiki.storage.database import Database
from kiki.storage.secrets import MemorySecretStore
from kiki.tools.audit import AuditLog
from kiki.tools.executor import ToolExecutor
from kiki.tools.policy import ToolPolicy
from kiki.tools.registry import ToolRegistry


@pytest.fixture(autouse=True, scope="session")
def _hermetic_state_home(tmp_path_factory) -> None:
    """Chat turns write run traces under XDG state. Keep the suite out of the
    user's real home; a test that sets the variable itself still wins."""
    os.environ.setdefault("XDG_STATE_HOME", str(tmp_path_factory.mktemp("state")))


@pytest.fixture
def settings(tmp_path: Path):
    return load_settings(tmp_path / "missing.toml")


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "kiki.sqlite3")


@pytest.fixture
def chats(db: Database) -> ChatRepository:
    return ChatRepository(db)


@pytest.fixture
def secrets() -> MemorySecretStore:
    return MemorySecretStore()


@pytest.fixture
def tools_env(db: Database) -> tuple[ToolRegistry, ToolExecutor]:
    registry = ToolRegistry()
    executor = ToolExecutor(registry, ToolPolicy(), AuditLog(db))
    return registry, executor
