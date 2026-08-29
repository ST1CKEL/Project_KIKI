"""Append-only security log for tool requests, confirmations and results.

What this is for: proving afterwards *that* something ran, who asked for it and
how it was decided. It is not a debugging trace and not a copy of the data a
tool touched.

Allowlist, not redaction
------------------------
A parameter value is stored only when its `ToolSpec` explicitly named it in
`audit_parameters`. Everything else is reduced to its shape — the key stays, the
value becomes `<str:142>`. That way a new tool, or a new argument on an existing
one, is silently safe instead of silently leaking: forgetting to redact is the
normal human error, and this makes forgetting the harmless direction.

The sanitising happens here, in the sink, so no call site can bypass it by
passing something raw.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from kiki.storage.database import Database

# An allowlisted value is still bounded and still screened: a tool author who
# allowlists "path" should not be able to write a home directory into the log.
MAX_VALUE_CHARS = 120
MAX_RESULT_CHARS = 200
MAX_KEYS = 20
_UNSAFE = re.compile(
    r"(?:https?|ftp|ws)://|/home/|/Users/|/root/|\bsk-|\bghp_|\bgho_|\bxox[baprs]-"
    r"|\bapi[_-]?key\b|\btoken\b|\bsecret\b|\bpasswor[dt]\b",
    re.IGNORECASE,
)

BLOCKED = "[gesperrt]"
UNSAFE = "[unsicher]"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _shape(value: Any) -> str:
    """What a value was, never what it said."""
    if value is None:
        return "<none>"
    if isinstance(value, bool):
        return "<bool>"
    if isinstance(value, int):
        return "<int>"
    if isinstance(value, float):
        return "<float>"
    if isinstance(value, str):
        return f"<str:{len(value)}>"
    if isinstance(value, list | tuple):
        return f"<list:{len(value)}>"
    if isinstance(value, dict):
        return f"<dict:{len(value)}>"
    return "<objekt>"


def _allowlisted(value: Any) -> Any:
    """An explicitly permitted value, if it is small and plainly harmless."""
    if isinstance(value, bool | int | float) or value is None:
        return value
    if not isinstance(value, str):
        # Containers are never worth their risk in a security log.
        return _shape(value)
    if len(value) > MAX_VALUE_CHARS or _UNSAFE.search(value):
        return UNSAFE
    return value


def safe_parameters(
    params: dict[str, Any] | None,
    *,
    allow: tuple[str, ...] = (),
    block: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Reduce call parameters to something a long-lived log may keep.

    `block` wins over `allow`: a parameter a tool declared sensitive can never be
    opened up again by an allowlist entry added later.
    """
    if not isinstance(params, dict):
        return {}
    allowed = set(allow)
    blocked = set(block)
    out: dict[str, Any] = {}
    for key in sorted(params)[:MAX_KEYS]:
        name = str(key)
        if name in blocked:
            out[name] = BLOCKED
        elif name in allowed:
            out[name] = _allowlisted(params[key])
        else:
            out[name] = _shape(params[key])
    return out


def result_code(payload: Any) -> str:
    """The shape of a tool's answer: which fields came back, never their values."""
    if isinstance(payload, dict):
        keys = ",".join(sorted(str(key) for key in payload)[:MAX_KEYS])
        return f"ok:{keys}"[:MAX_RESULT_CHARS]
    return f"ok:{_shape(payload)}"


def error_code(exc: BaseException | str) -> str:
    """A category. An exception message quotes whatever it choked on."""
    if isinstance(exc, BaseException):
        return type(exc).__name__
    text = str(exc).strip()
    if not text or _UNSAFE.search(text):
        return UNSAFE
    return text[:MAX_VALUE_CHARS]


@dataclass(frozen=True)
class AuditEntry:
    id: int
    ts: str
    tool: str
    params_json: str
    decision: str
    result: str | None
    error: str | None


class AuditLog:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._lock = Lock()

    def record(
        self,
        tool: str,
        params: dict[str, Any],
        decision: str,
        *,
        spec: Any = None,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        """Store one decision. Everything handed in is reduced before it lands.

        `spec` carries the tool's own allowlist. Without it nothing but shapes is
        kept — an unknown tool is the case where caution costs least.
        """
        safe = safe_parameters(
            params,
            allow=tuple(getattr(spec, "audit_parameters", ()) or ()),
            block=tuple(getattr(spec, "sensitive_parameters", ()) or ()),
        )
        with self._lock:
            self._db.conn.execute(
                "INSERT INTO audit_log(ts, tool, params_json, decision, result, error) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    _now(),
                    tool,
                    json.dumps(safe, ensure_ascii=False, default=str),
                    decision,
                    _bounded(result),
                    _bounded(error),
                ),
            )
            self._db.conn.commit()

    def recent(self, limit: int = 50) -> list[AuditEntry]:
        with self._lock:
            rows = self._db.conn.execute(
                "SELECT id, ts, tool, params_json, decision, result, error "
                "FROM audit_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [AuditEntry(**dict(row)) for row in rows]


def _bounded(text: str | None) -> str | None:
    """Last guard on the free-text columns.

    Callers are expected to pass a category already; this makes sure a slip
    cannot put a path or a token into the log anyway.
    """
    if text is None:
        return None
    value = str(text)
    if _UNSAFE.search(value):
        return UNSAFE
    return value[:MAX_RESULT_CHARS]
