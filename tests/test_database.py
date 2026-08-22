from __future__ import annotations

from kiki.storage.chat_repository import ChatRepository
from kiki.storage.database import Database
from kiki.storage.memory_repository import MemoryRepository


def test_conversation_roundtrip(chats: ChatRepository) -> None:
    conv = chats.create_conversation("Hallo")
    chats.add_message(conv.id, "user", "hi")
    chats.add_message(conv.id, "assistant", "moin")
    messages = chats.list_messages(conv.id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert chats.history(conv.id)[0].content == "hi"
    chats.rename_conversation(conv.id, "Linux-Hilfe")
    assert chats.get_conversation(conv.id).title == "Linux-Hilfe"


def test_delete_cascades(db: Database, chats: ChatRepository) -> None:
    conv = chats.create_conversation()
    chats.add_message(conv.id, "user", "bye")
    chats.delete_conversation(conv.id)
    assert chats.list_messages(conv.id) == []
    remaining = db.conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()
    assert remaining["n"] == 0


def test_memory_explicit_only(db: Database) -> None:
    repo = MemoryRepository(db)
    item = repo.add("Homelab steht in der Garage", kind="note", source="explicit")
    assert repo.list()[0].id == item.id
    repo.delete(item.id)
    assert repo.list() == []
