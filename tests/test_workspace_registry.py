from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kiki.storage.database import SCHEMA_VERSION, Database
from kiki.storage.workspace_repository import WorkspaceRepository
from kiki.workspaces.git_service import inspect_git
from kiki.workspaces.models import WorkspaceError
from kiki.workspaces.registry import WorkspaceRegistry

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git fehlt")


def _git_env(home: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "GIT_AUTHOR_NAME": "KIKI Test",
        "GIT_AUTHOR_EMAIL": "kiki@test.local",
        "GIT_COMMITTER_NAME": "KIKI Test",
        "GIT_COMMITTER_EMAIL": "kiki@test.local",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "LC_ALL": "C",
    }


def init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    env = _git_env(path)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, env=env, capture_output=True)
    (path / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True, env=env, capture_output=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-m", "init"],
        cwd=path,
        check=True,
        env=env,
        capture_output=True,
    )


def run_git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        env=_git_env(path),
        capture_output=True,
        text=True,
    )


def make_registry(tmp_path: Path, db: Database) -> tuple[WorkspaceRegistry, Path]:
    root = tmp_path / "Projects"
    root.mkdir()
    repo = WorkspaceRepository(db)
    registry = WorkspaceRegistry(repo, allowed_roots=[str(root)])
    return registry, root


def test_schema_version(db: Database) -> None:
    row = db.conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    assert int(row["value"]) == SCHEMA_VERSION
    assert SCHEMA_VERSION >= 2
    cols = {r["name"] for r in db.conn.execute("PRAGMA table_info(workspaces)").fetchall()}
    assert "canonical_path" in cols
    assert "risk_profile" in cols


def test_register_git_repo_and_status(tmp_path: Path, db: Database) -> None:
    registry, root = make_registry(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    record = registry.register(str(repo), display_name="App")
    assert record.display_name == "App"
    assert record.canonical_path == str(repo.resolve())
    assert record.risk_profile == "observe"
    assert record.active_branch == "main"
    assert record.git_head
    updated, snap = registry.inspect(record.id)
    assert updated.id == record.id
    assert snap.dirty is False
    assert snap.branch == "main"
    (repo / "dirty.txt").write_text("x", encoding="utf-8")
    _, dirty = registry.inspect(record.id)
    assert dirty.dirty is True
    assert dirty.untracked is True


def test_not_a_git_repo(tmp_path: Path, db: Database) -> None:
    registry, root = make_registry(tmp_path, db)
    folder = root / "plain"
    folder.mkdir()
    with pytest.raises(WorkspaceError) as exc:
        registry.register(str(folder))
    assert exc.value.code == "not_git"


def test_subdirectory_is_not_repo_root(tmp_path: Path, db: Database) -> None:
    registry, root = make_registry(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    nested = repo / "src"
    nested.mkdir()
    with pytest.raises(WorkspaceError) as exc:
        registry.register(str(nested))
    assert exc.value.code == "not_repo_root"


def test_unregistered_path_is_blocked(tmp_path: Path, db: Database) -> None:
    registry, root = make_registry(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    with pytest.raises(WorkspaceError) as exc:
        registry.require_path(str(repo))
    assert exc.value.code == "not_registered"
    registry.register(str(repo))
    found = registry.require_path(str(repo))
    assert found.canonical_path == str(repo.resolve())
    with pytest.raises(WorkspaceError) as exc:
        registry.require("missing-id")
    assert exc.value.code == "not_registered"


def test_outside_root_cannot_register(tmp_path: Path, db: Database) -> None:
    registry, _root = make_registry(tmp_path, db)
    other = tmp_path / "elsewhere"
    init_repo(other)
    with pytest.raises(WorkspaceError) as exc:
        registry.register(str(other))
    assert exc.value.code == "outside_root"


def test_remove_does_not_delete_repo(tmp_path: Path, db: Database) -> None:
    registry, root = make_registry(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    record = registry.register(str(repo))
    path = registry.remove(record.id)
    assert path == repo.resolve()
    assert (repo / "README.md").is_file()
    assert registry.get(record.id) is None
    assert registry.list() == []


def test_duplicate_register(tmp_path: Path, db: Database) -> None:
    registry, root = make_registry(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    registry.register(str(repo))
    with pytest.raises(WorkspaceError) as exc:
        registry.register(str(repo))
    assert exc.value.code == "already_registered"


def test_inspect_git_direct(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    init_repo(repo)
    snap = inspect_git(repo)
    assert snap.toplevel == str(repo.resolve())
    assert snap.branch == "main"
    assert snap.head
    assert snap.dirty is False


def test_develop_profile_stored(tmp_path: Path, db: Database) -> None:
    registry, root = make_registry(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    record = registry.register(str(repo), risk_profile="develop")
    assert record.risk_profile == "develop"


def test_remote_url_is_sanitized_before_storage(tmp_path: Path, db: Database) -> None:
    registry, root = make_registry(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    run_git(
        repo,
        "remote",
        "add",
        "origin",
        "https://alice:super-secret@example.test/acme/app.git?token=also-secret#fragment",
    )

    record = registry.register(str(repo))

    assert record.remote_url == "https://example.test/acme/app.git"
    stored = db.conn.execute(
        "SELECT remote_url FROM workspaces WHERE id = ?",
        (record.id,),
    ).fetchone()
    assert stored["remote_url"] == "https://example.test/acme/app.git"
    assert "secret" not in stored["remote_url"]


def test_registered_workspace_blocks_external_symlink_swap(tmp_path: Path, db: Database) -> None:
    registry, root = make_registry(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    record = registry.register(str(repo))
    moved = tmp_path / "moved-outside"
    repo.rename(moved)
    repo.symlink_to(moved, target_is_directory=True)

    with pytest.raises(WorkspaceError) as exc:
        registry.require(record.id)
    assert exc.value.code == "symlink_escape"

    removed = registry.remove(record.id)
    assert removed == repo
    assert (moved / "README.md").is_file()


def test_registered_workspace_blocks_internal_symlink_retarget(tmp_path: Path, db: Database) -> None:
    registry, root = make_registry(tmp_path, db)
    repo = root / "app"
    other = root / "other"
    init_repo(repo)
    init_repo(other)
    record = registry.register(str(repo))
    repo.rename(root / "original")
    repo.symlink_to(other, target_is_directory=True)

    with pytest.raises(WorkspaceError) as exc:
        registry.require(record.id)
    assert exc.value.code == "workspace_changed"


def test_require_path_rejects_alias_outside_allowed_root(tmp_path: Path, db: Database) -> None:
    registry, root = make_registry(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    registry.register(str(repo))
    alias = tmp_path / "external-alias"
    alias.symlink_to(repo, target_is_directory=True)

    with pytest.raises(WorkspaceError) as exc:
        registry.require_path(str(alias))
    assert exc.value.code == "outside_root"


def test_git_inspection_disables_repository_fsmonitor(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    init_repo(repo)
    marker = tmp_path / "fsmonitor-ran"
    helper = tmp_path / "fsmonitor-helper"
    helper.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)
    run_git(repo, "config", "core.fsmonitor", str(helper))

    snapshot = inspect_git(repo)

    assert snapshot.dirty is False
    assert not marker.exists()
