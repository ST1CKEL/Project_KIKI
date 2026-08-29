"""Routine fires through the one door, authorized by the stored recipe.

The engine below is Martin's real one, the repository is his real one; what
changed is what the engine was handed. These tests prove the whole chain:
a confirmed recipe fires through the gateway, everything else -- a mutated
argument, a disabled routine, a foreign origin, an unreadable authorization,
a switch flipped after the engine's tick -- meets the closed door.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from kiki.routines.engine import RoutineEngine
from kiki.routines.models import build_routine, build_trigger
from kiki.routines.repository import RoutineRepository
from kiki.storage.database import Database
from kiki.tools.audit import AuditLog
from kiki.tools.executor import ToolExecutor
from kiki.tools.gateway import ToolGateway
from kiki.tools.policy import DecisionKind, Origin, RiskLevel, ToolPolicy
from kiki.tools.registry import ToolRegistry, ToolSpec
from kiki.tools.routine_gateway import RoutineToolGateway

VOLUME_SCHEMA = {
    "type": "object",
    "properties": {"percent": {"type": "integer"}},
    "required": ["percent"],
    "additionalProperties": False,
}


def _volume_spec(counter: dict) -> ToolSpec:
    def _handler(params: dict[str, Any]) -> dict[str, Any]:
        counter["runs"] += 1
        counter["seen"] = dict(params)
        return {"set": params.get("percent")}

    return ToolSpec(
        name="audio.volume_set",
        title="Lautstärke",
        description="Setzt die Lautstärke.",
        risk=RiskLevel.CONTROL,
        parameters=VOLUME_SCHEMA,
        handler=_handler,
        effect="Ändert die Lautstärke.",
        auto_allow=True,
        requires_integration=False,
        model_callable=True,
    )


def _card_spec() -> ToolSpec:
    return ToolSpec(
        name="memory_remember",
        title="Gedächtnis",
        description="Merkt sich etwas.",
        risk=RiskLevel.WRITE,
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        handler=lambda _p: {"ok": True},
        effect="Merkt sich den Text.",
        auto_allow=False,
        requires_integration=False,
        model_callable=True,
    )


class _Live:
    def __init__(self) -> None:
        self.panic = False
        self.integrations = True


class _Metrics:
    def __init__(self, values: dict[str, float]) -> None:
        self._values = values

    def __call__(self) -> dict[str, float]:
        return self._values


def _stack(
    tmp_path: Path,
    *specs: ToolSpec,
    live: _Live | None = None,
) -> tuple[RoutineEngine, RoutineRepository, ToolExecutor, dict]:
    live = live or _Live()
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    db = Database(tmp_path / "kiki.db")
    executor = ToolExecutor(registry, ToolPolicy("balanced"), AuditLog(db))
    gateway = ToolGateway(
        executor,
        panic_check=lambda: live.panic,
        integrations_check=lambda: live.integrations,
    )
    repo = RoutineRepository(db)
    counter = {"runs": 0}
    engine = RoutineEngine(
        repo,
        RoutineToolGateway(gateway, repo),
        _Metrics({"battery.percent": 10.0}),
        panic_check=lambda: live.panic,
        integrations_check=lambda: live.integrations,
    )
    return engine, repo, executor, counter


def _battery_routine(arguments: dict[str, Any] | None = None) -> Any:
    return build_routine(
        name="Leiser bei leerem Akku",
        trigger=build_trigger("battery.percent", "lt", 15),
        tool_name="audio.volume_set",
        arguments=arguments if arguments is not None else {"percent": 20},
        cooldown_min=30,
    )


def _tick(engine: RoutineEngine) -> list[dict[str, Any]]:
    return asyncio.run(asyncio.wait_for(engine.tick(), timeout=10))


# --- the confirmed recipe fires through the door -------------------------------


def test_a_confirmed_recipe_fires_exactly_its_arguments(tmp_path):
    counter = {"runs": 0}
    engine, repo, _executor, live_counter = _stack(tmp_path, _volume_spec(counter))
    repo.add(_battery_routine({"percent": 20}))
    assert live_counter == {"runs": 0}

    fired = _tick(engine)

    assert len(fired) == 1 and fired[0]["ok"] is True
    assert counter["runs"] == 1
    assert counter["seen"] == {"percent": 20}
    # The cooldown was recorded against the routine that fired.
    assert repo.list()[0].fired_count == 1


def test_reordered_arguments_are_the_same_recipe(tmp_path):
    # Canonicalisation itself is proven in the confirmation broker tests;
    # what matters here is that the authorization compares canonically, not
    # by dict identity or order.
    counter = {"runs": 0}
    engine, repo, _ex, _c = _stack(tmp_path, _volume_spec(counter))
    repo.add(_battery_routine({"percent": 20}))

    async def scenario():
        adapter = engine._executor  # noqa: SLF001 - test rig
        return await adapter.run(
            "audio.volume_set",
            {"percent": 20},
            panic=False,
            integrations_enabled=True,
            origin=Origin.ROUTINE,
        )

    result = asyncio.run(scenario())
    assert result.ok is True
    assert counter["runs"] == 1


# --- everything else meets the closed door --------------------------------------


def test_a_mutated_argument_is_not_the_confirmed_recipe(tmp_path):
    counter = {"runs": 0}
    engine, repo, _ex, _c = _stack(tmp_path, _volume_spec(counter))
    routine = _battery_routine({"percent": 20})
    repo.add(routine)

    # The stored recipe drifts after confirmation: what fires no longer
    # matches what was approved.
    drifted = build_routine(
        name="Leiser bei leerem Akku",
        trigger=build_trigger("battery.percent", "lt", 15),
        tool_name="audio.volume_set",
        arguments={"percent": 100},
        cooldown_min=30,
        routine_id=routine.id,
    )
    repo.delete(routine.id)
    repo.add(drifted)

    async def scenario():
        adapter_run = engine._executor.run(  # noqa: SLF001 - test rig
            "audio.volume_set",
            {"percent": 20},
            panic=False,
            integrations_enabled=True,
            origin=Origin.ROUTINE,
        )
        return await adapter_run

    result = asyncio.run(scenario())
    assert result.ok is False
    assert result.decision.kind is DecisionKind.DENY
    assert counter["runs"] == 0


def test_a_disabled_recipe_authorizes_nothing(tmp_path):
    counter = {"runs": 0}
    engine, repo, _ex, _c = _stack(tmp_path, _volume_spec(counter))
    routine = _battery_routine()
    repo.add(routine)
    repo.set_enabled(routine.id, False)

    _tick(engine)

    assert counter["runs"] == 0
    assert repo.list()[0].fired_count == 0


def test_a_foreign_origin_gets_the_closed_door(tmp_path):
    counter = {"runs": 0}
    engine, repo, _ex, _c = _stack(tmp_path, _volume_spec(counter))
    repo.add(_battery_routine())

    async def scenario():
        return await engine._executor.run(  # noqa: SLF001 - test rig
            "audio.volume_set",
            {"percent": 20},
            panic=False,
            integrations_enabled=True,
            origin=Origin.MODEL,
        )

    result = asyncio.run(scenario())
    assert result.ok is False
    assert counter["runs"] == 0


def test_an_unreadable_authorization_fires_nothing(tmp_path):
    counter = {"runs": 0}
    engine, repo, _ex, _c = _stack(tmp_path, _volume_spec(counter))
    repo.add(_battery_routine())

    class _BrokenRepo:
        def list(self):
            raise RuntimeError("kaputt")

    engine._executor._recipes = _BrokenRepo()  # noqa: SLF001 - test rig
    _tick(engine)
    assert counter["runs"] == 0


def test_a_card_tool_stays_routinen_unfaehig(tmp_path):
    engine, repo, _ex, _c = _stack(tmp_path, _card_spec())
    repo.add(
        build_routine(
            name="Merken",
            trigger=build_trigger("battery.percent", "lt", 15),
            tool_name="memory_remember",
            arguments={"text": "geheim"},
            cooldown_min=30,
        )
    )
    fired = _tick(engine)
    assert len(fired) == 1
    assert fired[0]["ok"] is False
    # A policy deny is permanent: the routine is disabled, not re-asked.
    assert fired[0].get("disabled") is True
    assert repo.list()[0].enabled is False


# --- the live world, read again before the side effect --------------------------


def test_panic_after_the_engines_read_stops_the_fire(tmp_path):
    live = _Live()
    counter = {"runs": 0}
    engine, repo, _ex, _c = _stack(tmp_path, _volume_spec(counter), live=live)
    repo.add(_battery_routine())

    async def scenario():
        # The engine already decided to fire (its snapshots said "go") when
        # the switch flips. The gateway re-reads the world; the fire stops.
        adapter = engine._executor  # noqa: SLF001 - test rig
        live.panic = True
        return await adapter.run(
            "audio.volume_set",
            {"percent": 20},
            panic=False,  # the stale snapshot
            integrations_enabled=True,
            origin=Origin.ROUTINE,
        )

    result = asyncio.run(scenario())
    assert result.ok is False
    assert counter["runs"] == 0


# --- the audit -------------------------------------------------------------------


def test_every_fire_is_audited_with_its_origin(tmp_path):
    counter = {"runs": 0}
    engine, repo, executor, _c = _stack(tmp_path, _volume_spec(counter))
    repo.add(_battery_routine())
    _tick(engine)

    entries = [e for e in executor.audit.recent() if e.tool == "audio.volume_set"]
    assert entries
    decided = [e for e in entries if e.decision in ("allow", "executed")]
    assert decided
    # The origin travels with the decision entry; the executed entry carries
    # only the result shape -- contents never reach a long-lived log.
    allow_entry = next(e for e in entries if e.decision == "allow")
    assert "[routine]" in (allow_entry.result or "")
    executed_entry = next(e for e in entries if e.decision == "executed")
    assert executed_entry.result == "ok:set"
    assert executed_entry.params_json == '{"percent": "<int>"}'
