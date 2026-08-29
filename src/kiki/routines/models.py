"""Routine data model and validation. Nothing here executes anything."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# What a trigger can watch. Mirrors the metrics IntegrationMetrics reports;
# adding one means touching both places, on purpose.
KNOWN_METRICS: dict[str, str] = {
    "battery.percent": "Akkuladung in Prozent (nur im Akkubetrieb)",
    "disk.used_percent": "Belegter Speicherplatz des Heimatverzeichnisses in Prozent",
}

_OPS: dict[str, str] = {"lt": "<", "gt": ">", "eq": "≈"}

_MAX_NAME = 80
_MAX_COOLDOWN_MIN = 24 * 60
_EQ_TOLERANCE = 0.5


class RoutineError(ValueError):
    """Invalid recipe. The message is safe to show in a tool result."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RoutineTrigger:
    metric: str
    op: str
    value: float

    def matches(self, current: float) -> bool:
        if self.op == "lt":
            return current < self.value
        if self.op == "gt":
            return current > self.value
        return abs(current - self.value) <= _EQ_TOLERANCE

    def describe(self) -> str:
        label = KNOWN_METRICS.get(self.metric, self.metric)
        short = label.split(" (")[0]
        symbol = _OPS.get(self.op, self.op)
        return f"{short} {symbol} {self.value:g} %"


@dataclass(frozen=True)
class Routine:
    id: str
    name: str
    enabled: bool
    trigger: RoutineTrigger
    tool_name: str
    arguments: dict[str, Any]
    cooldown_min: int = 30
    created_at: str = ""
    last_fired_at: str | None = None
    fired_count: int = 0

    def describe(self) -> str:
        state = "aktiv" if self.enabled else "aus"
        return f"{self.name}: wenn {self.trigger.describe()} → {self.tool_name} ({state})"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def build_trigger(metric: str, op: str, value: float) -> RoutineTrigger:
    cleaned_metric = str(metric or "").strip()
    if cleaned_metric not in KNOWN_METRICS:
        raise RoutineError(
            "unknown_metric",
            f"Unbekannte Messgröße „{cleaned_metric}“. Bekannt: {', '.join(KNOWN_METRICS)}.",
        )
    cleaned_op = str(op or "").strip().lower()
    if cleaned_op not in _OPS:
        raise RoutineError("unknown_op", "Vergleich muss lt, gt oder eq sein.")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RoutineError("invalid_value", "Der Schwellenwert muss eine Zahl sein.")
    if not math.isfinite(float(value)) or not 0 <= float(value) <= 100:
        raise RoutineError("invalid_value", "Der Schwellenwert muss zwischen 0 und 100 liegen.")
    return RoutineTrigger(metric=cleaned_metric, op=cleaned_op, value=float(value))


def build_routine(
    *,
    name: str,
    trigger: RoutineTrigger,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    cooldown_min: int = 30,
    enabled: bool = True,
    routine_id: str | None = None,
    created_at: str | None = None,
    last_fired_at: str | None = None,
    fired_count: int = 0,
) -> Routine:
    cleaned_name = " ".join(str(name or "").split())
    if not cleaned_name:
        raise RoutineError("invalid_name", "Die Routine braucht einen Namen.")
    if len(cleaned_name) > _MAX_NAME:
        raise RoutineError("invalid_name", f"Der Name darf höchstens {_MAX_NAME} Zeichen haben.")
    cleaned_tool = str(tool_name or "").strip()
    if not cleaned_tool:
        raise RoutineError("invalid_tool", "Die Routine braucht ein Werkzeug.")
    if not isinstance(arguments or {}, dict):
        raise RoutineError("invalid_arguments", "Argumente müssen ein Objekt sein.")
    cooldown = int(cooldown_min)
    if not 1 <= cooldown <= _MAX_COOLDOWN_MIN:
        raise RoutineError(
            "invalid_cooldown",
            f"Die Abklingzeit muss zwischen 1 und {_MAX_COOLDOWN_MIN} Minuten liegen.",
        )
    return Routine(
        id=routine_id or uuid.uuid4().hex,
        name=cleaned_name,
        enabled=bool(enabled),
        trigger=trigger,
        tool_name=cleaned_tool,
        arguments=dict(arguments or {}),
        cooldown_min=cooldown,
        created_at=created_at or _now_iso(),
        last_fired_at=last_fired_at,
        fired_count=int(fired_count),
    )
