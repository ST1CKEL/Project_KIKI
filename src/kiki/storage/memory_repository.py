"""Explicit local memories.

Everything KIKI remembers about the user lands here, is visible in the settings,
and can be deleted there. Nothing is written without the user seeing it first —
the tools that reach this repository are confirmation-bound.

Growth is bounded on purpose: a memory is a short fact, not a diary. Long or
duplicate entries are rejected at this layer so no caller can quietly fill the
system prompt.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock

from kiki.storage.database import Database

MAX_CONTENT_CHARS = 400
MAX_MEMORIES = 200
VALID_KINDS: tuple[str, ...] = ("fact", "preference", "note")

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class MemoryError_(ValueError):
    """The memory could not be stored."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def clean_content(raw: str) -> str:
    """Collapse a memory to a single tidy line.

    Newlines and control characters are removed rather than escaped: memories
    are injected into the system prompt, and a multi-line entry could otherwise
    imitate the structure around it.
    """
    text = _CONTROL.sub(" ", str(raw or ""))
    return " ".join(text.split())


def clean_kind(raw: str) -> str:
    kind = str(raw or "").strip().lower()
    return kind if kind in VALID_KINDS else "note"


@dataclass(frozen=True)
class Memory:
    id: str
    kind: str
    content: str
    created_at: str
    source: str
    updated_at: str | None = None


class MemoryRepository:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._lock = Lock()

    def add(self, content: str, *, kind: str = "note", source: str = "explicit") -> Memory:
        text = clean_content(content)
        if not text:
            raise MemoryError_("Eine Erinnerung darf nicht leer sein.")
        if len(text) > MAX_CONTENT_CHARS:
            raise MemoryError_(
                f"Eine Erinnerung darf höchstens {MAX_CONTENT_CHARS} Zeichen haben "
                f"(war {len(text)})."
            )
        stamp = _now()
        item = Memory(
            id=str(uuid.uuid4()),
            kind=clean_kind(kind),
            content=text,
            created_at=stamp,
            source=source,
            updated_at=stamp,
        )
        with self._lock:
            # SQLite's NOCASE folds ASCII only, so "Fedora"/"fedora" dedupe but
            # "Öl"/"öl" do not. Good enough to stop the common repeat.
            existing = self._db.conn.execute(
                "SELECT id FROM memories WHERE content = ? COLLATE NOCASE", (text,)
            ).fetchone()
            if existing is not None:
                raise MemoryError_("Das ist schon gemerkt.")
            total = int(self._db.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"])
            if total >= MAX_MEMORIES:
                raise MemoryError_(
                    f"Gedächtnis ist voll ({MAX_MEMORIES}). Lösche zuerst etwas in den Einstellungen."
                )
            self._db.conn.execute(
                "INSERT INTO memories(id, kind, content, created_at, source, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (item.id, item.kind, item.content, item.created_at, item.source, item.updated_at),
            )
            self._db.conn.commit()
        return item

    def update(self, memory_id: str, content: str) -> Memory | None:
        text = clean_content(content)
        if not text:
            raise MemoryError_("Eine Erinnerung darf nicht leer sein.")
        if len(text) > MAX_CONTENT_CHARS:
            raise MemoryError_(f"Eine Erinnerung darf höchstens {MAX_CONTENT_CHARS} Zeichen haben.")
        with self._lock:
            cursor = self._db.conn.execute(
                "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
                (text, _now(), memory_id),
            )
            self._db.conn.commit()
            if cursor.rowcount == 0:
                return None
        return self.get(memory_id)

    def get(self, memory_id: str) -> Memory | None:
        with self._lock:
            row = self._db.conn.execute(
                "SELECT id, kind, content, created_at, source, updated_at "
                "FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return Memory(**dict(row)) if row else None

    def list(self, *, kind: str | None = None, limit: int | None = None) -> list[Memory]:
        query = (
            "SELECT id, kind, content, created_at, source, updated_at FROM memories"
            + (" WHERE kind = ?" if kind else "")
            + " ORDER BY created_at DESC"
            + (" LIMIT ?" if limit is not None else "")
        )
        params: list[object] = []
        if kind:
            params.append(kind)
        if limit is not None:
            params.append(max(0, int(limit)))
        with self._lock:
            rows = self._db.conn.execute(query, params).fetchall()
        return [Memory(**dict(row)) for row in rows]

    def search(self, query: str, *, limit: int = 10) -> list[Memory]:
        needle = clean_content(query)
        if not needle:
            return []
        # LIKE with escaped wildcards: a query is a search term, not a pattern.
        escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            rows = self._db.conn.execute(
                "SELECT id, kind, content, created_at, source, updated_at FROM memories "
                "WHERE content LIKE ? ESCAPE '\\' COLLATE NOCASE "
                "ORDER BY created_at DESC LIMIT ?",
                (f"%{escaped}%", max(1, int(limit))),
            ).fetchall()
        return [Memory(**dict(row)) for row in rows]

    def count(self) -> int:
        with self._lock:
            row = self._db.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        return int(row["n"])

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            cursor = self._db.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._db.conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> int:
        """Forget everything. Only ever called from a confirmed UI action."""
        with self._lock:
            cursor = self._db.conn.execute("DELETE FROM memories")
            self._db.conn.commit()
            return cursor.rowcount
