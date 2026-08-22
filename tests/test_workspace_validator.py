from __future__ import annotations

import os
from pathlib import Path

import pytest

from kiki.config.settings import SettingsError, default_mapping, settings_from_mapping
from kiki.workspaces.git_service import git_env
from kiki.workspaces.models import WorkspaceError
from kiki.workspaces.validator import expand_allowed_roots, validate_candidate, validate_risk_profile


def test_within_allowed_root(tmp_path: Path) -> None:
    root = tmp_path / "Projects"
    repo = root / "app"
    repo.mkdir(parents=True)
    found = validate_candidate(str(repo), allowed_roots=[str(root)], require_git_root=False)
    assert found == repo.resolve()


def test_outside_root_is_denied(tmp_path: Path) -> None:
    allowed = tmp_path / "Projects"
    allowed.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(WorkspaceError) as exc:
        validate_candidate(str(other), allowed_roots=[str(allowed)], require_git_root=False)
    assert exc.value.code == "outside_root"


def test_home_as_root_is_forbidden() -> None:
    with pytest.raises(WorkspaceError) as exc:
        expand_allowed_roots(["~"])
    assert exc.value.code == "forbidden_root"


def test_slash_root_is_forbidden() -> None:
    with pytest.raises(WorkspaceError) as exc:
        expand_allowed_roots(["/"])
    assert exc.value.code == "forbidden_root"


def test_symlink_escape_to_etc(tmp_path: Path) -> None:
    root = tmp_path / "Projects"
    root.mkdir()
    link = root / "escape"
    link.symlink_to("/etc")
    with pytest.raises(WorkspaceError) as exc:
        validate_candidate(str(link), allowed_roots=[str(root)], require_git_root=False)
    assert exc.value.code == "symlink_escape"


def test_symlink_from_outside_into_jail_is_denied(tmp_path: Path) -> None:
    root = tmp_path / "Projects"
    inner = root / "app"
    inner.mkdir(parents=True)
    outsider = tmp_path / "outside"
    outsider.mkdir()
    link = outsider / "alias"
    link.symlink_to(inner)
    with pytest.raises(WorkspaceError) as exc:
        validate_candidate(str(link), allowed_roots=[str(root)], require_git_root=False)
    assert exc.value.code == "outside_root"


def test_dotdot_escape_is_denied(tmp_path: Path) -> None:
    root = tmp_path / "Projects"
    root.mkdir()
    secret = tmp_path / "secret"
    secret.mkdir()
    sneaky = root / ".." / "secret"
    with pytest.raises(WorkspaceError) as exc:
        validate_candidate(str(sneaky), allowed_roots=[str(root)], require_git_root=False)
    assert exc.value.code == "outside_root"


def test_missing_path(tmp_path: Path) -> None:
    root = tmp_path / "Projects"
    root.mkdir()
    with pytest.raises(WorkspaceError) as exc:
        validate_candidate(str(root / "nope"), allowed_roots=[str(root)], require_git_root=False)
    assert exc.value.code == "not_found"


def test_file_is_not_a_workspace(tmp_path: Path) -> None:
    root = tmp_path / "Projects"
    root.mkdir()
    blob = root / "README"
    blob.write_text("x", encoding="utf-8")
    with pytest.raises(WorkspaceError) as exc:
        validate_candidate(str(blob), allowed_roots=[str(root)], require_git_root=False)
    assert exc.value.code == "not_a_directory"


def test_git_root_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "Projects"
    nested = root / "app" / "src"
    nested.mkdir(parents=True)
    with pytest.raises(WorkspaceError) as exc:
        validate_candidate(
            str(nested),
            allowed_roots=[str(root)],
            require_git_root=True,
            git_toplevel=str(root / "app"),
        )
    assert exc.value.code == "not_repo_root"


def test_missing_git_toplevel(tmp_path: Path) -> None:
    root = tmp_path / "Projects"
    repo = root / "app"
    repo.mkdir(parents=True)
    with pytest.raises(WorkspaceError) as exc:
        validate_candidate(str(repo), allowed_roots=[str(root)], require_git_root=True)
    assert exc.value.code == "not_git"


def test_risk_profile_validation() -> None:
    assert validate_risk_profile("observe") == "observe"
    assert validate_risk_profile("develop") == "develop"
    with pytest.raises(WorkspaceError) as exc:
        validate_risk_profile("godmode")
    assert exc.value.code == "invalid_profile"


def test_git_env_never_copies_secrets() -> None:
    os.environ["AWS_SECRET_ACCESS_KEY"] = "should-not-leak"
    try:
        env = git_env()
    finally:
        os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "SSH_AUTH_SOCK" not in env
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert "/usr/bin" in env["PATH"]


def test_config_rejects_home_and_empty_roots() -> None:
    data = default_mapping()
    data["workspaces"]["allowed_roots"] = ["~"]
    with pytest.raises(SettingsError):
        settings_from_mapping(data)
    data["workspaces"]["allowed_roots"] = []
    with pytest.raises(SettingsError):
        settings_from_mapping(data)
