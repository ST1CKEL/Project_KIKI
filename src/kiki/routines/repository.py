"""SQLite storage for routines. Plain CRUD, no execution."""

from __future__ import annotations

import json
import logging
from typing import Any

from kiki.routines.models import Routine, RoutineTrigger
from kiki.storage.database import Database

log = logging.getLogger(__name__)


class RoutineRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def list(self) -> list[Routine]:
        rows = self._db.conn.execute("SELECT * FROM routines ORDER BY created_at, id")
        routines: list[Routine] = []
        for row in rows:
            routine = self._row_to_routine(row)
            if routine is not None:
                routines.append(routine)
        return routines

    def get(self, routine_id: str) -> Routine | None:
        row = self._db.conn.execute(
            "SELECT * FROM routines WHERE id = ?", (routine_id,)
        ).fetchone()
        return self._row_to_routine(row) if row is not None else None

    def add(self, routine: Routine) -> None:
        self._db.conn.execute(
            """
            INSERT INTO routines (
                id, name, enabled, metric, op, value, tool_name,
                arguments_json, cooldown_min, created_at, last_fired_at, fired_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                routine.id,
                routine.name,
                1 if routine.enabled else 0,
                routine.trigger.metric,
                routine.trigger.op,
                routine.trigger.value,
                routine.tool_name,
                json.dumps(routine.arguments, ensure_ascii=False, sort_keys=True),
                routine.cooldown_min,
                routine.created_at,
                routine.last_fired_at,
                routine.fired_count,
            ),
        )
        self._db.conn.commit()

    def delete(self, routine_id: str) -> bool:
        cursor = self._db.conn.execute("DELETE FROM routines WHERE id = ?", (routine_id,))
        self._db.conn.commit()
        return cursor.rowcount > 0

    def set_enabled(self, routine_id: str, enabled: bool) -> bool:
        cursor = self._db.conn.execute(
            "UPDATE routines SET enabled = ? WHERE id = ?", (1 if enabled else 0, routine_id)
        )
        self._db.conn.commit()
        return cursor.rowcount > 0

    def record_fired(self, routine_id: str, fired_at_iso: str) -> None:
        self._db.conn.execute(
            """
            UPDATE routines
               SET last_fired_at = ?, fired_count = fired_count + 1
             WHERE id = ?
            """,
            (fired_at_iso, routine_id),
        )
        self._db.conn.commit()

    def _row_to_routine(self, row: Any) -> Routine | None:
        try:
            arguments = json.loads(row["arguments_json"] or "{}")
        except json.JSONDecodeError:
            # A row the engine cannot parse is skipped loudly, never executed.
            log.error("routine %s has unparsable arguments; skipping", row["id"])
            return None
        if not isinstance(arguments, dict):
            log.error("routine %s has non-object arguments; skipping", row["id"])
            return None
        return Routine(
            id=str(row["id"]),
            name=str(row["name"]),
            enabled=bool(row["enabled"]),
            trigger=RoutineTrigger(
                metric=str(row["metric"]), op=str(row["op"]), value=float(row["value"])
            ),
            tool_name=str(row["tool_name"]),
            arguments=arguments,
            cooldown_min=int(row["cooldown_min"]),
            created_at=str(row["created_at"]),
            last_fired_at=row["last_fired_at"],
            fired_count=int(row["fired_count"]),
        )
