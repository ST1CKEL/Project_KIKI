"""Tools that let KIKI keep, look up and drop memories.

Remembering and forgetting are WRITE actions, so they reach the approval card in
every autonomy level — the user sees the exact wording before it is stored. That
is deliberate: what KIKI remembers shapes every later answer, so it is not
something she may quietly decide on her own.
"""

from __future__ import annotations

from typing import Any

from kiki.storage.memory_repository import (
    MAX_CONTENT_CHARS,
    VALID_KINDS,
    MemoryError_,
    MemoryRepository,
)
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec

RECALL_LIMIT = 10


def memory_remember_spec(memories: MemoryRepository) -> ToolSpec:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        try:
            item = memories.add(
                params["content"], kind=params.get("kind", "note"), source="chat"
            )
        except MemoryError_ as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "id": item.id, "kind": item.kind, "content": item.content}

    return ToolSpec(
        name="memory_remember",
        title="Merken",
        description=(
            "Merkt eine kurze Tatsache oder Vorliebe des Nutzers dauerhaft. "
            "Nur benutzen, wenn der Nutzer ausdrücklich darum bittet oder klar "
            "etwas Dauerhaftes über sich mitteilt. Keine Gesprächsinhalte."
        ),
        risk=RiskLevel.WRITE,
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_CONTENT_CHARS,
                },
                "kind": {"type": "string", "enum": list(VALID_KINDS)},
            },
            "required": ["content"],
            "additionalProperties": False,
        },
        handler=handler,
        effect=(
            "Speichert diesen Text dauerhaft in KIKIs lokalem Gedächtnis. Er geht "
            "ab jetzt bei jeder Antwort in den Systemprompt ein, bis du ihn in den "
            "Einstellungen löschst."
        ),
        target="lokales Gedächtnis",
        auto_allow=True,
        requires_integration=True,
        model_callable=True,
    )


def memory_recall_spec(memories: MemoryRepository) -> ToolSpec:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        query = str(params.get("query") or "").strip()
        found = (
            memories.search(query, limit=RECALL_LIMIT)
            if query
            else memories.list(limit=RECALL_LIMIT)
        )
        return {
            "count": len(found),
            "memories": [
                {"id": m.id, "kind": m.kind, "content": m.content} for m in found
            ],
        }

    return ToolSpec(
        name="memory_recall",
        title="Gedächtnis durchsuchen",
        description=(
            "Sucht im lokalen Gedächtnis. Die wichtigsten Erinnerungen stehen "
            "bereits im Systemprompt; dieses Werkzeug findet ältere oder weitere."
        ),
        risk=RiskLevel.READ,
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "maxLength": 200}},
            "additionalProperties": False,
        },
        handler=handler,
        effect="Liest aus dem lokalen Gedächtnis. Keine Änderung.",
        target="lokales Gedächtnis",
        auto_allow=True,
        requires_integration=True,
        model_callable=True,
    )


def memory_forget_spec(memories: MemoryRepository) -> ToolSpec:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        memory_id = str(params["memory_id"])
        item = memories.get(memory_id)
        if item is None:
            return {"ok": False, "error": "Diese Erinnerung gibt es nicht."}
        return {"ok": memories.delete(memory_id), "content": item.content}

    return ToolSpec(
        name="memory_forget",
        title="Vergessen",
        description=(
            "Löscht genau eine Erinnerung. Die ID kommt aus memory_recall — "
            "niemals raten."
        ),
        risk=RiskLevel.WRITE,
        parameters={
            "type": "object",
            "properties": {"memory_id": {"type": "string", "minLength": 1, "maxLength": 64}},
            "required": ["memory_id"],
            "additionalProperties": False,
        },
        handler=handler,
        effect="Löscht diese Erinnerung endgültig aus dem lokalen Gedächtnis.",
        target="lokales Gedächtnis",
        auto_allow=True,
        requires_integration=True,
        model_callable=True,
    )


class MemorySkill:
    id = "memory"
    name = "Gedächtnis"
    description = "Merken, nachschlagen und vergessen — lokal, sichtbar, löschbar."

    def __init__(self, memories: MemoryRepository) -> None:
        self._memories = memories

    def tools(self) -> list[ToolSpec]:
        return [
            memory_remember_spec(self._memories),
            memory_recall_spec(self._memories),
            memory_forget_spec(self._memories),
        ]
