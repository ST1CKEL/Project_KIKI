"""What the security log may keep, and what it must never keep.

The audit is long-lived and lives in SQLite next to the conversations. It exists
to prove *that* an action was decided and run — not to hold a copy of what the
action touched. These fixtures are the values that must never come out of it
again, whichever path put them in.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from kiki.tools.audit import (
    BLOCKED,
    UNSAFE,
    AuditLog,
    error_code,
    result_code,
    safe_parameters,
)
from kiki.tools.executor import ToolExecutor
from kiki.tools.policy import Origin, RiskLevel, ToolPolicy
from kiki.tools.registry import ToolRegistry, ToolSpec

# The dangerous fixtures. None of these may appear in an audit row.
SECRETS = (
    "/home/martin/secret.txt",
    "sk-test-secret",
    "ghp_testtoken",
    "https://example.invalid/private",
)
LONG_MESSAGE = (
    "Eine lange Nutzernachricht, die niemand in einem Sicherheitsprotokoll "
    "wiederfinden sollte, weil sie dort schlicht nichts zu suchen hat. " * 4
)


def _spec(**kwargs) -> ToolSpec:
    values = dict(
        name="write_note",
        title="Notiz",
        description="Legt eine Notiz an.",
        risk=RiskLevel.WRITE,
        parameters={
            "type": "object",
            "properties": {"body": {"type": "string"}, "percent": {"type": "integer"}},
            "required": [],
            "additionalProperties": False,
        },
        handler=lambda params: {"created": True, "note": "milch.md"},
        effect="Schreibt eine Notiz.",
        auto_allow=True,
        requires_integration=False,
        model_callable=True,
    )
    values.update(kwargs)
    return ToolSpec(**values)


def _rows(db) -> list[dict]:
    return [
        dict(row)
        for row in db.conn.execute(
            "SELECT tool, params_json, decision, result, error FROM audit_log ORDER BY id"
        ).fetchall()
    ]


def _blob(db) -> str:
    return json.dumps(_rows(db), ensure_ascii=False)


# --- the allowlist itself ---------------------------------------------------


def test_a_value_nobody_allowed_is_reduced_to_its_shape() -> None:
    """Forgetting to declare anything must be the safe direction."""
    safe = safe_parameters({"body": LONG_MESSAGE, "count": 3, "flag": True})

    assert safe["body"] == f"<str:{len(LONG_MESSAGE)}>"
    assert safe["count"] == "<int>"
    assert safe["flag"] == "<bool>"


def test_an_allowlisted_scalar_is_kept() -> None:
    safe = safe_parameters({"percent": 30, "muted": False}, allow=("percent", "muted"))
    assert safe == {"muted": False, "percent": 30}


@pytest.mark.parametrize("secret", SECRETS)
def test_an_allowlisted_value_that_looks_dangerous_is_still_refused(secret) -> None:
    """A tool author who allowlists `path` must not be able to write a home
    directory into the log by accident."""
    safe = safe_parameters({"path": secret}, allow=("path",))
    assert safe["path"] == UNSAFE
    assert secret not in json.dumps(safe)


def test_an_allowlisted_value_that_is_too_long_is_refused() -> None:
    safe = safe_parameters({"note": LONG_MESSAGE}, allow=("note",))
    assert safe["note"] == UNSAFE


def test_a_sensitive_parameter_wins_over_an_allowlist_entry() -> None:
    """Declaring something sensitive must not be undoable by a later allowlist."""
    safe = safe_parameters({"body": "kurz"}, allow=("body",), block=("body",))
    assert safe["body"] == BLOCKED


def test_a_container_is_never_stored_even_when_allowlisted() -> None:
    safe = safe_parameters({"items": [1, 2, 3]}, allow=("items",))
    assert safe["items"] == "<list:3>"


def test_the_number_of_keys_is_bounded() -> None:
    safe = safe_parameters({f"k{index}": index for index in range(200)})
    assert len(safe) <= 20


def test_nonsense_parameters_do_not_crash_the_log() -> None:
    assert safe_parameters(None) == {}
    assert safe_parameters(["nicht", "dict"]) == {}


# --- results and errors -----------------------------------------------------


def test_a_result_is_reduced_to_its_field_names() -> None:
    """Which fields came back is structure; what they held is content."""
    code = result_code({"note": "/home/martin/geheim.md", "created": True})
    assert code == "ok:created,note"
    assert "/home/" not in code


@pytest.mark.parametrize("secret", SECRETS)
def test_no_result_carries_a_secret(secret) -> None:
    assert secret not in result_code({"value": secret})


def test_an_exception_becomes_its_class_name() -> None:
    exc = RuntimeError("konnte /home/martin/secret.txt nicht lesen: sk-test-secret")
    assert error_code(exc) == "RuntimeError"


def test_a_dangerous_error_string_is_refused() -> None:
    assert error_code("token sk-test-secret abgelehnt") == UNSAFE


def test_a_harmless_error_string_survives() -> None:
    assert error_code("no_confirmation_ui") == "no_confirmation_ui"


# --- through the real executor ----------------------------------------------


def test_a_confirmed_write_leaves_no_secret_behind(db) -> None:
    """The whole path: proposal, confirmation, execution — three audit rows."""
    registry = ToolRegistry()
    registry.register(_spec(sensitive_parameters=("body",)))
    executor = ToolExecutor(registry, ToolPolicy("trusted"), AuditLog(db))

    async def _confirm(_preview) -> bool:
        return True

    result = asyncio.run(
        executor.run(
            "write_note",
            {"body": f"{LONG_MESSAGE} {' '.join(SECRETS)}"},
            panic=False,
            integrations_enabled=True,
            confirm=_confirm,
            origin=Origin.MODEL,
        )
    )

    assert result.ok is True
    blob = _blob(db)
    for secret in SECRETS:
        assert secret not in blob, secret
    assert LONG_MESSAGE[:40] not in blob
    assert BLOCKED in blob


def test_a_failing_tool_records_a_category_not_a_message(db) -> None:
    def _boom(_params):
        raise RuntimeError("konnte /home/martin/secret.txt nicht lesen: sk-test-secret")

    registry = ToolRegistry()
    # READ: this test is about the failure path, not about confirmation.
    registry.register(_spec(handler=_boom, risk=RiskLevel.READ))
    executor = ToolExecutor(registry, ToolPolicy("trusted"), AuditLog(db))

    result = asyncio.run(
        executor.run(
            "write_note", {}, panic=False, integrations_enabled=True, origin=Origin.MODEL
        )
    )

    assert result.ok is False
    rows = [row for row in _rows(db) if row["decision"] == "error"]
    assert rows and rows[0]["error"] == "RuntimeError"
    for secret in SECRETS:
        assert secret not in _blob(db)


def test_a_tool_result_is_never_copied_into_the_log(db) -> None:
    registry = ToolRegistry()
    registry.register(
        _spec(
            handler=lambda params: {"content": LONG_MESSAGE, "path": SECRETS[0]},
            risk=RiskLevel.READ,
        )
    )
    executor = ToolExecutor(registry, ToolPolicy("trusted"), AuditLog(db))

    asyncio.run(
        executor.run(
            "write_note", {}, panic=False, integrations_enabled=True, origin=Origin.MODEL
        )
    )

    rows = [row for row in _rows(db) if row["decision"] == "executed"]
    assert rows and rows[0]["result"] == "ok:content,path"
    assert SECRETS[0] not in _blob(db)


def test_an_unknown_tool_still_stores_only_shapes(db) -> None:
    """No ToolSpec means no allowlist — the case where caution costs least."""
    executor = ToolExecutor(ToolRegistry(), ToolPolicy("trusted"), AuditLog(db))

    asyncio.run(
        executor.run(
            "gibtsnicht",
            {"body": SECRETS[1]},
            panic=False,
            integrations_enabled=True,
            origin=Origin.MODEL,
        )
    )

    blob = _blob(db)
    assert SECRETS[1] not in blob
    assert "<str:" in blob


def test_an_allowlisted_argument_survives_the_real_path(db) -> None:
    """The log stays useful: a volume level is exactly what one wants to see."""
    registry = ToolRegistry()
    registry.register(
        _spec(
            name="audio.set_volume",
            risk=RiskLevel.CONTROL,
            audit_parameters=("percent",),
            handler=lambda params: {"volume": 30},
        )
    )
    executor = ToolExecutor(registry, ToolPolicy("trusted"), AuditLog(db))

    asyncio.run(
        executor.run(
            "audio.set_volume",
            {"percent": 30},
            panic=False,
            integrations_enabled=True,
            origin=Origin.MODEL,
        )
    )

    stored = [json.loads(row["params_json"]) for row in _rows(db)]
    assert {"percent": 30} in stored


# --- the log file is a sink too ---------------------------------------------


def test_a_tool_failure_writes_no_traceback_to_the_log(db, caplog) -> None:
    def _boom(_params):
        raise RuntimeError("konnte /home/martin/secret.txt nicht lesen: sk-test-secret")

    registry = ToolRegistry()
    registry.register(_spec(handler=_boom, risk=RiskLevel.READ))
    executor = ToolExecutor(registry, ToolPolicy("trusted"), AuditLog(db))

    with caplog.at_level("DEBUG"):
        asyncio.run(
            executor.run(
                "write_note", {}, panic=False, integrations_enabled=True, origin=Origin.MODEL
            )
        )

    for secret in SECRETS:
        assert secret not in caplog.text, secret
    assert "Traceback" not in caplog.text


def test_the_executor_never_serialises_a_payload_into_the_audit() -> None:
    """Guard against the line coming back: the old code dumped the whole
    result, truncated at 2000 characters."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "src" / "kiki" / "tools" / "executor.py"
    ).read_text(encoding="utf-8")
    assert "json.dumps(payload" not in source
    assert "log.exception(" not in source
