"""Conversations and messages."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock

from kiki.ai.provider import ChatMessage
from kiki.storage.database import Database


def _now() -> str:
    return datetime.now(UTC).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class StoredMessage:
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str

    def to_chat(self) -> ChatMessage:
        return ChatMessage(role=self.role, content=self.content)


class ChatRepository:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._lock = Lock()

    def create_conversation(self, title: str = "Neuer Chat") -> Conversation:
        cid = new_id()
        ts = _now()
        with self._lock:
            self._db.conn.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (cid, title, ts, ts),
            )
            self._db.conn.commit()
        return Conversation(id=cid, title=title, created_at=ts, updated_at=ts)

    def list_conversations(self) -> list[Conversation]:
        with self._lock:
            rows = self._db.conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [Conversation(**dict(row)) for row in rows]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            row = self._db.conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return Conversation(**dict(row)) if row else None

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        with self._lock:
            self._db.conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now(), conversation_id),
            )
            self._db.conn.commit()

    def delete_conversation(self, conversation_id: str) -> None:
        with self._lock:
            self._db.conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            self._db.conn.commit()

    def add_message(self, conversation_id: str, role: str, content: str) -> StoredMessage:
        mid = new_id()
        ts = _now()
        with self._lock:
            self._db.conn.execute(
                "INSERT INTO messages(id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (mid, conversation_id, role, content, ts),
            )
            self._db.conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (ts, conversation_id),
            )
            self._db.conn.commit()
        return StoredMessage(
            id=mid,
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=ts,
        )

    def list_messages(self, conversation_id: str) -> list[StoredMessage]:
        with self._lock:
            rows = self._db.conn.execute(
                "SELECT id, conversation_id, role, content, created_at "
                "FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            ).fetchall()
        return [StoredMessage(**dict(row)) for row in rows]

    def history(self, conversation_id: str) -> list[ChatMessage]:
        return [m.to_chat() for m in self.list_messages(conversation_id)]
