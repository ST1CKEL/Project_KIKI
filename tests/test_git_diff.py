from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from kiki.agents.broker import AgentBroker
from kiki.agents.models import AgentError
from kiki.agents.session_service import SessionService
from kiki.runners.local import LocalWorkspaceRunner
from kiki.storage.agent_session_repository import AgentSessionRepository
from kiki.storage.approval_repository import ApprovalRepository
from kiki.storage.audit_repository import AgentAuditRepository
from kiki.storage.database import Database
from kiki.workspaces.git_service import read_diff
from kiki.workspaces.registry import WorkspaceRegistry
from tests.test_workspace_registry import (  # noqa: F401
    _git_env,
    init_repo,
    make_registry,
    pytestmark,
    run_git,
)


def _service(tmp_path: Path, db: Database) -> tuple[SessionService, WorkspaceRegistry]:
    registry, _root = make_registry(tmp_path, db)
    return (
        SessionService(
            registry,
            AgentSessionRepository(db),
            ApprovalRepository(db),
            AgentAuditRepository(db),
            AgentBroker(opencode_binary="/no/opencode"),
            LocalWorkspaceRunner(),
        ),
        registry,
    )


def test_read_diff_shows_uncommitted_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    init_repo(repo)
    readme = repo / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
    diff = read_diff(repo)
    assert "README" in diff.stat or "changed" in diff.patch


def test_read_diff_disables_repository_textconv(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    init_repo(repo)
    (repo / ".gitattributes").write_text("*.note diff=leaky\n", encoding="utf-8")
    note = repo / "example.note"
    note.write_text("before\n", encoding="utf-8")
    run_git(repo, "add", ".gitattributes", "example.note")
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-m", "add note"],
        cwd=repo,
        check=True,
        env=_git_env(repo),
        capture_output=True,
    )
    marker = tmp_path / "textconv-ran"
    helper = tmp_path / "textconv-helper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        f"Path({str(marker)!r}).touch()\n"
        "print(Path(sys.argv[1]).read_text(encoding='utf-8'))\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)
    run_git(repo, "config", "diff.leaky.textconv", str(helper))
    note.write_text("after\n", encoding="utf-8")

    diff = read_diff(repo)

    assert "example.note" in diff.patch
    assert not marker.exists()


def test_git_diff_and_open_need_registry(tmp_path: Path, db: Database) -> None:
    service, registry = _service(tmp_path, db)
    root = tmp_path / "Projects"
    repo = root / "app"
    init_repo(repo)
    readme = repo / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
    workspace = registry.register(str(repo))

    async def _diff():
        return await service.git_diff(workspace.id, profile="observe")

    diff = asyncio.run(_diff())
    assert "README" in diff.stat or "changed" in diff.patch

    with pytest.raises(AgentError) as exc:
        service.open_workspace_folder(workspace.id, profile="observe")
    assert exc.value.code == "no_approval"

    params = {"workspace_id": workspace.id}
    approval = service.request_approval("workspace.open_in_file_manager", params, profile="observe")
    service.decide_approval(approval.id, approved=False)
    with pytest.raises(AgentError):
        service.open_workspace_folder(workspace.id, profile="observe", approval_id=approval.id)

    with pytest.raises(AgentError) as term:
        service.open_terminal(workspace.id, profile="observe")
    assert term.value.code == "no_approval"

    with pytest.raises(AgentError) as browser:
        service.open_browser_url("file:///etc/passwd", profile="observe")
    assert browser.value.code == "invalid_url"

    bad_file = {
        "workspace_id": workspace.id,
        "path": "/etc/passwd",
    }
    req = service.request_approval("workspace.open_file", bad_file, profile="observe")
    service.decide_approval(req.id, approved=True)
    with pytest.raises(AgentError) as leaked:
        service.open_workspace_file(workspace.id, "/etc/passwd", profile="observe", approval_id=req.id)
    assert leaked.value.code == "outside_root"
