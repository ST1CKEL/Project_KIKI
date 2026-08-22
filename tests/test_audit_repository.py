from __future__ import annotations

from kiki.agents.models import arguments_hash
from kiki.storage.audit_repository import AgentAuditRepository
from kiki.storage.database import SCHEMA_VERSION, Database


def test_agent_audit_roundtrip(db: Database) -> None:
    assert SCHEMA_VERSION >= 3
    repo = AgentAuditRepository(db)
    digest = arguments_hash({"workspace_id": "w", "task": "x"})
    repo.record(
        actor="user",
        event_type="policy",
        risk_class="write",
        requested_action="agent.start_implementation",
        arguments_hash=digest,
        policy_decision="deny",
        result_status="denied",
        result_summary="no approval",
    )
    rows = repo.recent()
    assert rows[0].requested_action == "agent.start_implementation"
    assert rows[0].resolved_arguments_hash == digest
    assert "sk-" not in (rows[0].result_summary or "")
    assert rows[0].policy_decision == "deny"
