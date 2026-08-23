"""Local JSONL traces: what the harness did, never what it was told.

One file per run, one JSON object per line, flushed and closed after every
event — a crash mid-run still leaves everything up to that point on disk.

What a trace is for is debugging and regression, so it records the observable
behaviour: which tool, which arguments, which category of failure, how long, and
how it ended. What it deliberately never records is the user's text, a model
prompt, a chain of thought, an exception string, a URL, a home path or a secret.
`user_text_length` is the only thing said about the input at all.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVENTS: frozenset[str] = frozenset(
    {
        "run_started",
        "model_action_received",
        "tool_requested",
        "tool_completed",
        "run_cancelled",
        "run_failed",
        "run_finished",
    }
)

# Anything a value could smuggle in. Checked on the way out, so a careless tool
# cannot turn a trace into a leak.
_FORBIDDEN = re.compile(
    r"(?:https?|ftp|ws)://|/home/|/Users/|/root/|\bsk-|\bghp_|\bapi[_-]?key\b|\btoken\b",
    re.IGNORECASE,
)
MAX_VALUE_CHARS = 200


class TraceWriteError(Exception):
    """The trace could not be written. The run ends on this, never silently."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def sanitize(value: Any, *, depth: int = 0) -> Any:
    """Reduce a value to something a trace may hold.

    Strings are bounded and screened; containers are walked once. Anything that
    looks like a URL, a home path or a credential becomes `[entfernt]` rather
    than being trimmed, because a trimmed secret is still a secret.
    """
    if depth > 3:
        return "[tief]"
    if isinstance(value, str):
        if _FORBIDDEN.search(value):
            return "[entfernt]"
        return value if len(value) <= MAX_VALUE_CHARS else value[:MAX_VALUE_CHARS] + "…"
    if isinstance(value, bool | int | float) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): sanitize(item, depth=depth + 1) for key, item in list(value.items())[:20]}
    if isinstance(value, list | tuple):
        return [sanitize(item, depth=depth + 1) for item in list(value)[:20]]
    return "[objekt]"


class TraceRecorder:
    """Writes one run's trace. The directory is always given, never guessed."""

    def __init__(self, trace_dir: Path | str, run_id: str) -> None:
        self._dir = Path(trace_dir)
        self._path = self._dir / f"{run_id}.jsonl"
        self._run_id = run_id
        self._sequence = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def sequence(self) -> int:
        return self._sequence

    def write(self, event: str, **fields: Any) -> None:
        """Append one event. Raises `TraceWriteError`; never fails quietly."""
        if event not in EVENTS:
            raise TraceWriteError(f"unbekanntes Ereignis: {event}")
        record: dict[str, Any] = {
            "event": event,
            "run_id": self._run_id,
            "timestamp": _now(),
            "sequence": self._sequence,
        }
        for key, value in fields.items():
            record[key] = sanitize(value)
        line = json.dumps(record, ensure_ascii=False, sort_keys=False)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            # Opened and closed per event: a crash in the next step still leaves
            # everything written so far readable.
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
        except OSError as exc:
            raise TraceWriteError(type(exc).__name__) from exc
        self._sequence += 1

    def read(self) -> list[dict[str, Any]]:
        """The trace as parsed records. For tests and for reading one back."""
        if not self._path.is_file():
            return []
        with self._path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
