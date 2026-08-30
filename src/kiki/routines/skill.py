"""Tools for managing routines from chat and voice.

`routines.create` is deliberately auto_allow=False in every level, jarvis
included: the confirmation card is the whole point. It shows the recipe word
for word — trigger, tool, arguments, cooldown — because that exact call is
what the engine will later fire without asking again.
"""

from __future__ import annotations

import logging
from typing import Any

from kiki.routines.models import (
    KNOWN_METRICS,
    Routine,
    RoutineError,
    build_routine,
    build_trigger,
)
from kiki.routines.repository import RoutineRepository
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolRegistry, ToolSpec
from kiki.tools.schemas import validate_params

log = logging.getLogger(__name__)


class RoutinesSkill:
    id = "routines"
    name = "Routinen"
    description = "Wenn-Dann-Routinen erstellen, anzeigen, ein-/ausschalten und löschen."

    def __init__(self, repository: RoutineRepository, tools: ToolRegistry) -> None:
        self._repository = repository
        self._tools = tools

    def tools(self) -> list[ToolSpec]:
        return [
            self._create_spec(),
            self._list_spec(),
            self._delete_spec(),
            self._toggle_spec(),
        ]

    def _create_spec(self) -> ToolSpec:
        return ToolSpec(
            name="routines.create",
            title="Routine erstellen",
            description=(
                "Erstellt eine Wenn-Dann-Routine. `metric` ist eine von "
                f"{', '.join(KNOWN_METRICS)}; `op` ist lt/gt/eq; `value` 0–100. "
                "`tool` und `arguments` beschreiben den Werkzeugaufruf, der bei "
                "Erfüllung ausgelöst wird. Die Freigabekarte zeigt das komplette "
                "Rezept — genau dieser Aufruf läuft später ohne erneute Frage."
            ),
            risk=RiskLevel.WRITE,
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 80},
                    "metric": {"type": "string", "enum": sorted(KNOWN_METRICS)},
                    "op": {"type": "string", "enum": ["lt", "gt", "eq"]},
                    "value": {"type": "number"},
                    "tool": {"type": "string", "minLength": 1, "maxLength": 64},
                    "arguments": {"type": "object"},
                    "cooldown_min": {"type": "integer"},
                },
                "required": ["name", "metric", "op", "value", "tool", "arguments"],
                "additionalProperties": False,
            },
            handler=self._create,
            effect="Legt eine dauerhafte Routine an, die später ohne Rückfrage handelt.",
            target="Routinen-Speicher",
            auto_allow=False,
            model_callable=True,
        )

    def _list_spec(self) -> ToolSpec:
        return ToolSpec(
            name="routines.list",
            title="Routinen auflisten",
            description="Listet die gespeicherten Routinen mit Auslöser, Werkzeug und Status.",
            risk=RiskLevel.READ,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=self._list,
            effect="Liest den Routinen-Speicher. Keine Änderung.",
            target="Routinen-Speicher",
            auto_allow=True,
            model_callable=True,
        )

    def _delete_spec(self) -> ToolSpec:
        return ToolSpec(
            name="routines.delete",
            title="Routine löschen",
            description="Löscht eine Routine endgültig. Die ID kommt von routines.list.",
            risk=RiskLevel.WRITE,
            parameters={
                "type": "object",
                "properties": {"id": {"type": "string", "minLength": 1, "maxLength": 64}},
                "required": ["id"],
                "additionalProperties": False,
            },
            handler=self._delete,
            effect="Entfernt die Routine dauerhaft.",
            target="Routinen-Speicher",
            auto_allow=False,
            model_callable=True,
        )

    def _toggle_spec(self) -> ToolSpec:
        return ToolSpec(
            name="routines.toggle",
            title="Routine ein-/ausschalten",
            description="Aktiviert oder deaktiviert eine Routine, ohne sie zu löschen.",
            risk=RiskLevel.CONTROL,
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "minLength": 1, "maxLength": 64},
                    "enabled": {"type": "boolean"},
                },
                "required": ["id", "enabled"],
                "additionalProperties": False,
            },
            handler=self._toggle,
            effect="Schaltet eine Routine aktiv oder inaktiv.",
            target="Routinen-Speicher",
            auto_allow=False,
            model_callable=True,
        )

    def _validate_action(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        """Returns a German error when the stored action is not routinable."""
        spec = self._tools.get(tool_name)
        if spec is None:
            return f"Unbekanntes Werkzeug „{tool_name}“ — Default Deny."
        if spec.risk is RiskLevel.EXTERNAL:
            return (
                f"„{tool_name}“ ist extern und braucht bei jedem Aufruf eine "
                "aktuelle Nutzerbestätigung; deshalb ist es nicht routinenfähig."
            )
        if not spec.auto_allow:
            return f"„{tool_name}“ ist nicht routinenfähig (erlaubt keine unbeaufsichtigten Aufrufe)."
        try:
            validate_params(spec.parameters, arguments)
        except Exception as exc:
            return f"Argumente passen nicht zu „{tool_name}“: {exc}"
        return None

    def _create(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            trigger = build_trigger(
                str(params["metric"]), str(params["op"]), params["value"]
            )
            routine = build_routine(
                name=str(params["name"]),
                trigger=trigger,
                tool_name=str(params["tool"]),
                arguments=params.get("arguments") or {},
                cooldown_min=int(params.get("cooldown_min", 30)),
            )
        except RoutineError as exc:
            return {"ok": False, "error": str(exc)}
        problem = self._validate_action(routine.tool_name, routine.arguments)
        if problem is not None:
            return {"ok": False, "error": problem}
        self._repository.add(routine)
        return {
            "ok": True,
            "routine": self._describe(routine),
            "hint": "Die Routine ist aktiv und feuert bei Erfüllung des Auslösers.",
        }

    def _list(self, _params: dict[str, Any]) -> dict[str, Any]:
        routines = self._repository.list()
        return {
            "ok": True,
            "count": len(routines),
            "routines": [self._describe(r) for r in routines],
        }

    def _delete(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._repository.delete(str(params["id"])):
            return {"ok": False, "error": "Routine nicht gefunden."}
        return {"ok": True}

    def _toggle(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._repository.set_enabled(str(params["id"]), bool(params["enabled"])):
            return {"ok": False, "error": "Routine nicht gefunden."}
        return {"ok": True, "enabled": bool(params["enabled"])}

    @staticmethod
    def _describe(routine: Routine) -> dict[str, Any]:
        return {
            "id": routine.id,
            "name": routine.name,
            "enabled": routine.enabled,
            "trigger": routine.trigger.describe(),
            "tool": routine.tool_name,
            "arguments": routine.arguments,
            "cooldown_min": routine.cooldown_min,
            "fired_count": routine.fired_count,
        }
