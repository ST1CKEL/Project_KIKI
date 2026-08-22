from __future__ import annotations

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
from kiki.storage.workspace_repository import WorkspaceRepository
from kiki.workspaces.registry import WorkspaceRegistry


def _service(tmp_path: Path, db: Database) -> SessionService:
    root = tmp_path / "Projects"
    root.mkdir()
    return SessionService(
        WorkspaceRegistry(
            WorkspaceRepository(db),
            allowed_roots=[str(root)],
        ),
        AgentSessionRepository(db),
        ApprovalRepository(db),
        AgentAuditRepository(db),
        AgentBroker(opencode_binary="/no/opencode"),
        LocalWorkspaceRunner(),
    )


def test_clipboard_approval_is_exact_one_time_and_redacted(
    tmp_path: Path,
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, db)
    secret_text = "einmaliger sichtbarer Inhalt"
    request = service.request_approval(
        "desktop.copy_text",
        {"text": secret_text},
        profile="observe",
    )
    stored = db.conn.execute(
        "SELECT params_json FROM approval_requests WHERE id = ?",
        (request.id,),
    ).fetchone()["params_json"]
    assert secret_text not in stored
    assert "<redacted:" in stored

    copied: list[str] = []
    monkeypatch.setattr(
        "kiki.agents.session_service.copy_text_to_clipboard",
        lambda text: copied.append(text) or len(text),
    )
    service.decide_approval(request.id, approved=True)
    assert service.copy_desktop_text(secret_text, approval_id=request.id) == len(secret_text)
    assert copied == [secret_text]

    with pytest.raises(AgentError) as reused:
        service.copy_desktop_text(secret_text, approval_id=request.id)
    assert reused.value.code == "no_approval"
    summaries = [row.result_summary or "" for row in service.recent_audit(30)]
    assert all(secret_text not in summary for summary in summaries)


def test_notification_needs_matching_approval_and_redacts_content(
    tmp_path: Path,
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, db)
    params = {"title": "KIKI", "body": "Der Test ist fertig"}
    request = service.request_approval("desktop.show_notification", params, profile="observe")
    service.decide_approval(request.id, approved=True)
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "kiki.agents.session_service.show_desktop_notification",
        lambda title, body: sent.append((title, body)) or (title, body),
    )
    with pytest.raises(AgentError) as mismatch:
        service.show_notification("KIKI", "Anderer Inhalt", approval_id=request.id)
    assert mismatch.value.code == "no_approval"
    assert sent == []

    second = service.request_approval("desktop.show_notification", params, profile="observe")
    service.decide_approval(second.id, approved=True)
    assert service.show_notification(**params, approval_id=second.id) == (
        "KIKI",
        "Der Test ist fertig",
    )
    assert sent == [("KIKI", "Der Test ist fertig")]
    stored = db.conn.execute(
        "SELECT params_json FROM approval_requests WHERE id = ?",
        (second.id,),
    ).fetchone()["params_json"]
    assert "Der Test ist fertig" not in stored
