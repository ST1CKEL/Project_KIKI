from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from kiki.agents.broker import AgentBroker
from kiki.agents.models import AgentError, AgentEvent, AgentEventType, SessionKind
from kiki.agents.session_service import SessionService
from kiki.runners.local import LocalWorkspaceRunner
from kiki.runners.process import RunnerError
from kiki.storage.agent_session_repository import AgentSessionRepository
from kiki.storage.approval_repository import ApprovalRepository
from kiki.storage.audit_repository import AgentAuditRepository
from kiki.storage.database import Database
from kiki.storage.workspace_repository import WorkspaceRepository
from kiki.workspaces.registry import WorkspaceRegistry

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git fehlt")

_FAKE = """#!/usr/bin/env python3
import sys
args = sys.argv[1:]
if "--version" in args or args[:1] == ["version"]:
    print("opencode 0.0-test")
    raise SystemExit(0)
if args[:1] == ["run"]:
    print("PLAN: inspect, implement, test")
    raise SystemExit(0)
raise SystemExit(2)
"""

_MUTATING_FAKE = """#!/usr/bin/env python3
import pathlib
import sys
args = sys.argv[1:]
if "--version" in args or args[:1] == ["version"]:
    print("opencode 0.0-test")
    raise SystemExit(0)
if args[:1] == ["run"]:
    pathlib.Path("README.md").write_text("changed by plan\\n", encoding="utf-8")
    print("PLAN: changed the tree")
    raise SystemExit(0)
raise SystemExit(2)
"""

_MUTATING_UNTRACKED_FAKE = """#!/usr/bin/env python3
import pathlib
import sys
args = sys.argv[1:]
if "--version" in args or args[:1] == ["version"]:
    print("opencode 0.0-test")
    raise SystemExit(0)
if args[:1] == ["run"]:
    pathlib.Path("notes.txt").write_text("changed by plan\\n", encoding="utf-8")
    print("PLAN: changed untracked content")
    raise SystemExit(0)
raise SystemExit(2)
"""

_SLOW_FAKE = """#!/usr/bin/env python3
import sys
import time
args = sys.argv[1:]
if "--version" in args or args[:1] == ["version"]:
    print("opencode 0.0-test")
    raise SystemExit(0)
if args[:1] == ["run"]:
    print("PLAN: running", flush=True)
    time.sleep(0.8)
    raise SystemExit(0)
raise SystemExit(2)
"""

_IGNORE_TERM_FAKE = """#!/usr/bin/env python3
import os
import pathlib
import signal
import sys
import time
args = sys.argv[1:]
if "--version" in args or args[:1] == ["version"]:
    print("opencode 0.0-test")
    raise SystemExit(0)
if args[:1] == ["run"]:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pathlib.Path(".agent-pid").write_text(str(os.getpid()), encoding="utf-8")
    print("PLAN: waiting", flush=True)
    time.sleep(30)
    raise SystemExit(0)
raise SystemExit(2)
"""


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


def _service(
    tmp_path: Path,
    db: Database,
    *,
    plan_first: bool = True,
    runner: LocalWorkspaceRunner | None = None,
) -> tuple[SessionService, WorkspaceRegistry, Path]:
    root = tmp_path / "Projects"
    root.mkdir()
    fake = tmp_path / "opencode"
    fake.write_text(_FAKE, encoding="utf-8")
    fake.chmod(0o755)
    registry = WorkspaceRegistry(WorkspaceRepository(db), allowed_roots=[str(root)])
    service = SessionService(
        registry,
        AgentSessionRepository(db),
        ApprovalRepository(db),
        AgentAuditRepository(db),
        AgentBroker(opencode_binary=str(fake), stop_grace_seconds=0.2),
        runner or LocalWorkspaceRunner(),
        plan_first=plan_first,
    )
    return service, registry, root


def test_plan_session_and_audit(tmp_path: Path, db: Database) -> None:
    service, registry, root = _service(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    workspace = registry.register(str(repo))

    async def _run():
        session = await service.run_plan(workspace.id, "Add a login form", profile="observe", agent_timeout=15)
        return session

    session = asyncio.run(_run())
    assert session.kind is SessionKind.PLAN
    assert session.git_head_before
    audits = AgentAuditRepository(db).recent()
    actions = {row.requested_action for row in audits}
    assert "agent.plan" in actions
    assert any(row.policy_decision in {"allow", "executed"} for row in audits)


def test_plan_detects_and_rejects_worktree_mutation(tmp_path: Path, db: Database) -> None:
    service, registry, root = _service(tmp_path, db)
    (tmp_path / "opencode").write_text(_MUTATING_FAKE, encoding="utf-8")
    (tmp_path / "opencode").chmod(0o755)
    repo = root / "app"
    init_repo(repo)
    workspace = registry.register(str(repo))

    plan = asyncio.run(service.run_plan(workspace.id, "Read only", agent_timeout=15))

    assert plan.status.value == "failed"
    assert plan.summary == "observe-worktree-changed"
    events = service.list_events(plan.id)
    assert any("Observe-Verstoß" in event.text for event in events)


def test_plan_detects_mutation_of_existing_untracked_file(tmp_path: Path, db: Database) -> None:
    service, registry, root = _service(tmp_path, db)
    (tmp_path / "opencode").write_text(_MUTATING_UNTRACKED_FAKE, encoding="utf-8")
    (tmp_path / "opencode").chmod(0o755)
    repo = root / "app"
    init_repo(repo)
    (repo / "notes.txt").write_text("private draft\n", encoding="utf-8")
    workspace = registry.register(str(repo))

    plan = asyncio.run(service.run_plan(workspace.id, "Read only", agent_timeout=15))

    assert plan.status.value == "failed"
    assert plan.summary == "observe-worktree-changed"


def test_plan_fails_closed_when_final_fingerprint_breaks(
    tmp_path: Path,
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, registry, root = _service(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    workspace = registry.register(str(repo))
    calls = 0

    def _fingerprint(_path: Path) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return "before"
        raise OSError("repository disappeared")

    monkeypatch.setattr("kiki.agents.session_service.worktree_fingerprint", _fingerprint)
    plan = asyncio.run(service.run_plan(workspace.id, "Inspect", agent_timeout=15))

    assert plan.status.value == "failed"
    assert plan.summary == "observe-check-failed"
    assert any("Observe-Prüfung fehlgeschlagen" in event.text for event in service.list_events(plan.id))


def test_implementation_blocked_without_approval(tmp_path: Path, db: Database) -> None:
    service, registry, root = _service(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    workspace = registry.register(str(repo))
    plan = asyncio.run(service.run_plan(workspace.id, "Change files", agent_timeout=15))

    async def _run() -> None:
        await service.start_implementation(
            workspace.id,
            "Change files",
            profile="develop",
            plan_session_id=plan.id,
        )

    with pytest.raises(AgentError) as exc:
        asyncio.run(_run())
    assert exc.value.code == "no_approval"


def test_plan_first_requires_matching_finished_plan(tmp_path: Path, db: Database) -> None:
    service, registry, root = _service(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    workspace = registry.register(str(repo))
    params = {
        "workspace_id": workspace.id,
        "task": "Change files",
        "model": "",
        "profile": "develop",
    }

    with pytest.raises(AgentError) as missing:
        service.request_approval("agent.start_implementation", params, profile="develop")
    assert missing.value.code == "plan_required"

    plan = asyncio.run(service.run_plan(workspace.id, "Original task", agent_timeout=15))
    with pytest.raises(AgentError) as changed:
        service.request_approval(
            "agent.start_implementation",
            {**params, "plan_session_id": plan.id},
            profile="develop",
        )
    assert changed.value.code == "plan_stale"


def test_plan_first_can_be_disabled_but_supplied_plan_must_be_valid(
    tmp_path: Path,
    db: Database,
) -> None:
    service, registry, root = _service(tmp_path, db, plan_first=False)
    repo = root / "app"
    init_repo(repo)
    workspace = registry.register(str(repo))
    params = {
        "workspace_id": workspace.id,
        "task": "Change files",
        "model": "",
        "profile": "develop",
    }

    request = service.request_approval("agent.start_implementation", params, profile="develop")
    service.decide_approval(request.id, approved=True)
    session = asyncio.run(
        service.start_implementation(
            workspace.id,
            "Change files",
            profile="develop",
            approval_id=request.id,
            agent_timeout=15,
        )
    )
    assert session.status.value == "finished"
    assert session.plan_session_id is None

    with pytest.raises(AgentError) as invalid:
        service.request_approval(
            "agent.start_implementation",
            {**params, "plan_session_id": "missing"},
            profile="develop",
        )
    assert invalid.value.code == "invalid_plan"


def test_observe_cannot_implement(tmp_path: Path, db: Database) -> None:
    service, registry, root = _service(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    workspace = registry.register(str(repo))

    async def _run() -> None:
        await service.start_implementation(workspace.id, "Change files", profile="observe")

    with pytest.raises(AgentError) as exc:
        asyncio.run(_run())
    assert exc.value.code == "denied"


def test_approval_bound_to_params(tmp_path: Path, db: Database) -> None:
    service, registry, root = _service(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    workspace = registry.register(str(repo))
    plan = asyncio.run(service.run_plan(workspace.id, "Change files", agent_timeout=15))
    params = {
        "workspace_id": workspace.id,
        "task": "Change files",
        "model": "",
        "profile": "develop",
        "plan_session_id": plan.id,
    }
    request = service.request_approval("agent.start_implementation", params, profile="develop")
    service.decide_approval(request.id, approved=True)

    async def _wrong() -> None:
        await service.start_implementation(
            workspace.id,
            "Change files",
            profile="develop",
            model="different-model",
            plan_session_id=plan.id,
            approval_id=request.id,
        )

    with pytest.raises(AgentError) as exc:
        asyncio.run(_wrong())
    assert exc.value.code == "no_approval"

    request2 = service.request_approval("agent.start_implementation", params, profile="develop")
    service.decide_approval(request2.id, approved=True)

    async def _right():
        return await service.start_implementation(
            workspace.id,
            "Change files",
            profile="develop",
            model="",
            plan_session_id=plan.id,
            approval_id=request2.id,
            agent_timeout=15,
        )

    session = asyncio.run(_right())
    assert session.kind is SessionKind.IMPLEMENT
    assert session.plan_session_id == plan.id
    events = AgentSessionRepository(db).list_events(session.id)
    assert any(event.type.value == "session_finished" for event in events)


def test_tests_need_develop_approval(tmp_path: Path, db: Database) -> None:
    service, registry, root = _service(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    workspace = registry.register(str(repo))

    async def _run() -> None:
        await service.run_tests(workspace.id, "python_pytest", profile="observe")

    with pytest.raises(AgentError) as exc:
        asyncio.run(_run())
    assert exc.value.code == "denied"


def test_test_output_is_captured_persisted_and_linked(tmp_path: Path, db: Database) -> None:
    service, registry, root = _service(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    workspace = registry.register(str(repo))
    plan = asyncio.run(service.run_plan(workspace.id, "Inspect tests", agent_timeout=15))
    params = {"workspace_id": workspace.id, "profile": "python_pytest"}
    approval = service.request_approval("tests.run_profile", params, profile="develop")
    service.decide_approval(approval.id, approved=True)

    result = asyncio.run(
        service.run_tests(
            workspace.id,
            "python_pytest",
            profile="develop",
            approval_id=approval.id,
            session_id=plan.id,
        )
    )

    assert isinstance(result["output"], str)
    assert result["output"].strip()
    stored = db.conn.execute(
        "SELECT session_id, output_text, output_truncated FROM test_runs WHERE id = ?",
        (result["test_id"],),
    ).fetchone()
    assert stored["session_id"] == plan.id
    assert stored["output_text"] == result["output"]
    events = AgentSessionRepository(db).list_events(plan.id)
    assert {event.type.value for event in events} >= {
        "test_started",
        "test_output",
        "test_finished",
    }


def test_large_event_payload_roundtrips_as_valid_bounded_json(tmp_path: Path, db: Database) -> None:
    service, registry, root = _service(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    workspace = registry.register(str(repo))
    plan = asyncio.run(service.run_plan(workspace.id, "Inspect", agent_timeout=15))
    repository = AgentSessionRepository(db)

    repository.add_event(
        plan.id,
        AgentEvent(
            type=AgentEventType.TEST_OUTPUT,
            text="x" * 20_000,
            data={
                "truncated": False,
                "text": "must-not-override",
                "_payload_truncated": False,
                "key" * 3000: "large-key",
                **{f"meta-{index}": "m" * 200 for index in range(100)},
                "not-a-number": float("nan"),
            },
        ),
    )

    raw = db.conn.execute(
        "SELECT payload_json, json_valid(payload_json) AS valid "
        "FROM agent_events WHERE session_id = ? ORDER BY ts DESC LIMIT 1",
        (plan.id,),
    ).fetchone()
    stored = repository.list_events(plan.id)[-1]
    assert raw["valid"] == 1
    assert len(raw["payload_json"]) <= 8000
    assert stored.text.startswith("x" * 100)
    assert len(stored.text) < 8000
    assert stored.data["_payload_truncated"] is True
    assert stored.data["truncated"] is False


def test_failed_test_spawn_is_finalized_and_audited(tmp_path: Path, db: Database) -> None:
    runner = LocalWorkspaceRunner(
        profiles={"python_pytest": ("kiki-command-that-does-not-exist",)},
    )
    service, registry, root = _service(tmp_path, db, runner=runner)
    repo = root / "app"
    init_repo(repo)
    workspace = registry.register(str(repo))
    plan = asyncio.run(service.run_plan(workspace.id, "Inspect tests", agent_timeout=15))
    params = {"workspace_id": workspace.id, "profile": "python_pytest"}
    approval = service.request_approval("tests.run_profile", params, profile="develop")
    service.decide_approval(approval.id, approved=True)

    with pytest.raises(RunnerError):
        asyncio.run(
            service.run_tests(
                workspace.id,
                "python_pytest",
                profile="develop",
                approval_id=approval.id,
                session_id=plan.id,
            )
        )

    stored = db.conn.execute("SELECT * FROM test_runs ORDER BY started_at DESC LIMIT 1").fetchone()
    assert stored["status"] == "error"
    assert stored["finished_at"] is not None
    assert "RunnerError" in stored["output_text"]
    events = AgentSessionRepository(db).list_events(plan.id)
    assert events[-1].type is AgentEventType.TEST_FINISHED
    audits = AgentAuditRepository(db).recent()
    assert any(row.requested_action == "tests.run_profile" and row.result_status == "error" for row in audits)


def test_stop_finished_session_preserves_finished(tmp_path: Path, db: Database) -> None:
    service, registry, root = _service(tmp_path, db)
    repo = root / "app"
    init_repo(repo)
    workspace = registry.register(str(repo))
    plan = asyncio.run(service.run_plan(workspace.id, "Inspect", agent_timeout=15))

    with pytest.raises(AgentError) as exc:
        asyncio.run(service.stop(plan.id, panic=True))

    assert exc.value.code == "not_running"
    stored = service.get_session(plan.id)
    assert stored is not None
    assert stored.status.value == "finished"


def test_stop_waits_for_process_exit_and_preserves_stopped_summary(tmp_path: Path, db: Database) -> None:
    service, registry, root = _service(tmp_path, db)
    fake = tmp_path / "opencode"
    fake.write_text(_IGNORE_TERM_FAKE, encoding="utf-8")
    fake.chmod(0o755)
    repo = root / "app"
    init_repo(repo)
    workspace = registry.register(str(repo))

    async def _run() -> tuple[object, object]:
        task = asyncio.create_task(service.run_plan(workspace.id, "Wait", agent_timeout=15))
        session = None
        for _ in range(100):
            sessions = service.list_sessions(workspace.id)
            if sessions and (repo / ".agent-pid").exists():
                session = sessions[0]
                break
            await asyncio.sleep(0.02)
        assert session is not None
        await service.stop(session.id, panic=True)
        stopped = await task
        fake.write_text(_FAKE, encoding="utf-8")
        fake.chmod(0o755)
        following = await service.run_plan(workspace.id, "Next", agent_timeout=15)
        return stopped, following

    stopped, following = asyncio.run(_run())
    pid = int((repo / ".agent-pid").read_text(encoding="utf-8"))
    with pytest.raises(OSError):
        os.kill(pid, 0)
    assert stopped.status.value == "failed"
    assert stopped.summary == "stopped"
    assert following.status.value == "finished"


def test_parallel_session_start_allows_only_one(tmp_path: Path, db: Database) -> None:
    service, registry, root = _service(tmp_path, db)
    fake = tmp_path / "opencode"
    fake.write_text(_SLOW_FAKE, encoding="utf-8")
    fake.chmod(0o755)
    repo = root / "app"
    init_repo(repo)
    workspace = registry.register(str(repo))

    async def _run():
        return await asyncio.gather(
            service.run_plan(workspace.id, "One", agent_timeout=15),
            service.run_plan(workspace.id, "Two", agent_timeout=15),
            return_exceptions=True,
        )

    results = asyncio.run(_run())
    errors = [result for result in results if isinstance(result, AgentError)]
    sessions = service.list_sessions(workspace.id)
    assert len(errors) == 1
    assert errors[0].code == "session_active"
    assert len(sessions) == 1
