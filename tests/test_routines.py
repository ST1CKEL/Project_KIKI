"""Routines: model validation, storage, engine decisions, policy origin."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from kiki.routines.engine import RoutineEngine
from kiki.routines.metrics import IntegrationMetrics
from kiki.routines.models import (
    RoutineError,
    build_routine,
    build_trigger,
)
from kiki.routines.repository import RoutineRepository
from kiki.routines.skill import RoutinesSkill
from kiki.storage.database import Database
from kiki.tools.audit import AuditLog
from kiki.tools.executor import ToolExecutor
from kiki.tools.policy import AutonomyLevel, DecisionKind, Origin, RiskLevel, ToolPolicy
from kiki.tools.registry import ToolRegistry, ToolSpec

# --- Modelle ---------------------------------------------------------------


def test_build_trigger_validates_metric_op_and_range() -> None:
    with pytest.raises(RoutineError):
        build_trigger("cpu.temp", "lt", 50)
    with pytest.raises(RoutineError):
        build_trigger("battery.percent", "like", 50)
    with pytest.raises(RoutineError):
        build_trigger("battery.percent", "lt", 150)
    trigger = build_trigger("battery.percent", "lt", 15)
    assert trigger.describe().startswith("Akkuladung")


def test_build_routine_cleans_name_and_defaults() -> None:
    trigger = build_trigger("disk.used_percent", "gt", 90)
    routine = build_routine(
        name="  Speicher   voll  ", trigger=trigger, tool_name="desktop.show_notification"
    )
    assert routine.name == "Speicher voll"
    assert routine.enabled is True
    assert routine.cooldown_min == 30
    assert len(routine.id) == 32
    with pytest.raises(RoutineError):
        build_routine(name="", trigger=trigger, tool_name="x")


def test_trigger_matches_lt_gt_eq() -> None:
    assert build_trigger("battery.percent", "lt", 15).matches(14.9)
    assert not build_trigger("battery.percent", "lt", 15).matches(15.0)
    assert build_trigger("disk.used_percent", "gt", 90).matches(90.1)
    assert build_trigger("battery.percent", "eq", 50).matches(50.3)
    assert not build_trigger("battery.percent", "eq", 50).matches(51.2)


# --- Storage ---------------------------------------------------------------


def test_repository_roundtrip_and_migration_to_v7(tmp_path) -> None:
    db = Database(tmp_path / "kiki.sqlite3")
    assert db.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[
        0
    ] == "7"
    repo = RoutineRepository(db)
    trigger = build_trigger("battery.percent", "lt", 15)
    routine = build_routine(name="Akku-Notiz", trigger=trigger, tool_name="media.play_pause")
    repo.add(routine)

    stored = repo.list()
    assert len(stored) == 1
    assert stored[0].name == "Akku-Notiz"
    assert stored[0].trigger.metric == "battery.percent"
    assert stored[0].trigger.op == "lt"
    assert stored[0].trigger.value == 15.0
    assert stored[0].enabled is True

    assert repo.set_enabled(routine.id, False) is True
    assert repo.get(routine.id).enabled is False
    repo.record_fired(routine.id, "2026-08-26T12:00:00+00:00")
    fired = repo.get(routine.id)
    assert fired.fired_count == 1
    assert fired.last_fired_at == "2026-08-26T12:00:00+00:00"
    assert repo.delete(routine.id) is True
    assert repo.delete(routine.id) is False
    assert repo.list() == []


def test_repository_skips_unparsable_arguments(tmp_path) -> None:
    db = Database(tmp_path / "kiki.sqlite3")
    repo = RoutineRepository(db)
    db.conn.execute(
        "INSERT INTO routines (id, name, enabled, metric, op, value, tool_name,"
        " arguments_json, cooldown_min, created_at) VALUES"
        " ('bad', 'Kaputt', 1, 'battery.percent', 'lt', 10, 'x', '{kein json', 30, 't')"
    )
    db.conn.commit()
    assert repo.list() == []


# --- Metriken --------------------------------------------------------------


class FakeIntegration:
    def __init__(self, data: dict[str, Any], *, available: bool = True) -> None:
        self.data = data
        self.available = available

    def snapshot(self):
        from kiki.integrations.base import IntegrationSnapshot

        return IntegrationSnapshot("fake", "Fake", self.available, self.data, None)


def test_integration_metrics_battery_only_when_discharging() -> None:
    metrics = IntegrationMetrics(
        FakeIntegration({"present": True, "percentage": 14.0, "state": "discharging"}),
        FakeIntegration({"used_percent": 91.2}),
    )
    assert metrics.snapshot() == {"battery.percent": 14.0, "disk.used_percent": 91.2}
    charging = IntegrationMetrics(
        FakeIntegration({"present": True, "percentage": 14.0, "state": "charging"}),
        FakeIntegration({"used_percent": 91.2}),
    )
    assert "battery.percent" not in charging.snapshot()


def test_integration_metrics_tolerates_throwing_integration() -> None:
    class Throwing:
        def snapshot(self):
            raise RuntimeError("D-Bus weg")

    metrics = IntegrationMetrics(Throwing(), FakeIntegration({"used_percent": 50.0}))
    assert metrics.snapshot() == {"disk.used_percent": 50.0}


# --- Engine ----------------------------------------------------------------


def _tool_registry() -> tuple[ToolRegistry, list[dict[str, Any]]]:
    registry = ToolRegistry()
    executed: list[dict[str, Any]] = []

    def handler(params: dict[str, Any]) -> dict[str, Any]:
        executed.append(dict(params))
        return {"ok": True}

    registry.register(
        ToolSpec(
            name="media.play_pause",
            title="Play/Pause",
            description="test",
            risk=RiskLevel.CONTROL,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=handler,
            effect="test",
            auto_allow=True,
            model_callable=True,
        )
    )
    registry.register(
        ToolSpec(
            name="clipboard.write",
            title="Zwischenablage",
            description="test",
            risk=RiskLevel.WRITE,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=handler,
            effect="test",
            auto_allow=False,
            model_callable=True,
        )
    )
    return registry, executed


def _engine(tmp_path, *, clock=time.time, panic: bool = False, integrations: bool = True):
    db = Database(tmp_path / "kiki.sqlite3")
    repo = RoutineRepository(db)
    registry, executed = _tool_registry()
    executor = ToolExecutor(registry, ToolPolicy(AutonomyLevel.BALANCED.value), AuditLog(db))
    metrics = lambda: {"battery.percent": 10.0}  # noqa: E731
    engine = RoutineEngine(
        repo,
        executor,
        metrics,
        panic_check=lambda: panic,
        integrations_check=lambda: integrations,
        clock=clock,
    )
    return repo, engine, executed


def test_engine_fires_matching_routine_through_executor(tmp_path) -> None:
    repo, engine, executed = _engine(tmp_path)
    repo.add(
        build_routine(
            name="Akku Pause",
            trigger=build_trigger("battery.percent", "lt", 15),
            tool_name="media.play_pause",
        )
    )
    fired = asyncio.run(engine.tick())
    assert len(fired) == 1
    assert fired[0]["ok"] is True
    assert len(executed) == 1
    stored = repo.list()[0]
    assert stored.fired_count == 1
    assert stored.last_fired_at is not None


def test_engine_respects_cooldown(tmp_path) -> None:
    now = time.time()
    # First fire at "now", a second attempt one minute later must stay silent,
    # a third after the cooldown passes fires again.
    repo, engine_now, executed = _engine(tmp_path, clock=lambda: now)
    repo.add(
        build_routine(
            name="Akku Pause",
            trigger=build_trigger("battery.percent", "lt", 15),
            tool_name="media.play_pause",
            cooldown_min=30,
        )
    )
    assert len(asyncio.run(engine_now.tick())) == 1

    repo2, engine_soon, executed_soon = _engine(tmp_path, clock=lambda: now + 60)
    assert len(asyncio.run(engine_soon.tick())) == 0
    assert executed_soon == []

    repo3, engine_later, _ = _engine(tmp_path, clock=lambda: now + 31 * 60)
    assert len(asyncio.run(engine_later.tick())) == 1
    assert len(executed) == 1


def test_engine_skips_disabled_and_disables_unroutinable(tmp_path) -> None:
    repo, engine, executed = _engine(tmp_path)
    routine = build_routine(
        name="Aus",
        trigger=build_trigger("battery.percent", "lt", 15),
        tool_name="media.play_pause",
        enabled=False,
    )
    repo.add(routine)
    unroutinable = build_routine(
        name="Karte nötig",
        trigger=build_trigger("battery.percent", "lt", 15),
        tool_name="clipboard.write",
    )
    repo.add(unroutinable)
    fired = asyncio.run(engine.tick())
    assert executed == []
    # The unroutinable recipe is reported once and then switched off, so a
    # permanent policy deny cannot re-deny itself into the audit every tick.
    assert len(fired) == 1
    assert fired[0]["ok"] is False
    assert fired[0]["disabled"] is True
    assert repo.get(unroutinable.id).enabled is False


def test_engine_honours_panic(tmp_path) -> None:
    db = Database(tmp_path / "kiki.sqlite3")
    repo = RoutineRepository(db)
    registry, executed = _tool_registry()
    executor = ToolExecutor(registry, ToolPolicy(), AuditLog(db))
    engine = RoutineEngine(
        repo,
        executor,
        lambda: {"battery.percent": 10.0},
        panic_check=lambda: True,
        integrations_check=lambda: True,
    )
    repo.add(
        build_routine(
            name="Akku",
            trigger=build_trigger("battery.percent", "lt", 15),
            tool_name="media.play_pause",
        )
    )
    assert asyncio.run(engine.tick()) == []
    assert executed == []


def test_policy_routine_origin_allows_only_routinable_tools() -> None:
    policy = ToolPolicy(AutonomyLevel.BALANCED.value)
    allowed = policy.evaluate(
        name="media.play_pause",
        params={},
        spec=_tool_registry()[0].get("media.play_pause"),
        panic=False,
        integrations_enabled=True,
        origin=Origin.ROUTINE,
    )
    assert allowed.kind is DecisionKind.ALLOW
    denied = policy.evaluate(
        name="clipboard.write",
        params={},
        spec=_tool_registry()[0].get("clipboard.write"),
        panic=False,
        integrations_enabled=True,
        origin=Origin.ROUTINE,
    )
    assert denied.kind is DecisionKind.DENY
    assert "nicht routinenfähig" in denied.reason


# --- Skill -----------------------------------------------------------------


def _skill(tmp_path) -> tuple[RoutinesSkill, RoutineRepository]:
    db = Database(tmp_path / "kiki.sqlite3")
    repo = RoutineRepository(db)
    registry, _executed = _tool_registry()
    return RoutinesSkill(repo, registry), repo


def _specs(skill: RoutinesSkill) -> dict[str, ToolSpec]:
    return {spec.name: spec for spec in skill.tools()}


def test_skill_create_stores_validated_routine(tmp_path) -> None:
    skill, repo = _skill(tmp_path)
    result = _specs(skill)["routines.create"].handler(
        {
            "name": "Akku Pause",
            "metric": "battery.percent",
            "op": "lt",
            "value": 15,
            "tool": "media.play_pause",
            "arguments": {},
        }
    )
    assert result["ok"] is True
    assert len(repo.list()) == 1


def test_skill_create_rejects_unroutinable_tool(tmp_path) -> None:
    skill, repo = _skill(tmp_path)
    result = _specs(skill)["routines.create"].handler(
        {
            "name": "Karte",
            "metric": "battery.percent",
            "op": "lt",
            "value": 15,
            "tool": "clipboard.write",
            "arguments": {},
        }
    )
    assert result["ok"] is False
    assert "nicht routinenfähig" in result["error"]
    assert repo.list() == []


def test_skill_create_rejects_external_tool_even_when_auto_allowed(tmp_path) -> None:
    db = Database(tmp_path / "kiki.sqlite3")
    repo = RoutineRepository(db)
    registry, _executed = _tool_registry()
    registry.register(
        ToolSpec(
            name="external.notify",
            title="Extern",
            description="Verlässt die lokale Grenze.",
            risk=RiskLevel.EXTERNAL,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=lambda _params: {"ok": True},
            effect="Extern.",
            auto_allow=True,
        )
    )
    skill = RoutinesSkill(repo, registry)
    result = _specs(skill)["routines.create"].handler(
        {
            "name": "Nicht extern feuern",
            "metric": "battery.percent",
            "op": "lt",
            "value": 15,
            "tool": "external.notify",
            "arguments": {},
        }
    )
    assert result["ok"] is False
    assert "bei jedem Aufruf" in result["error"]
    assert repo.list() == []


def test_skill_create_rejects_mismatched_arguments(tmp_path) -> None:
    skill, _repo = _skill(tmp_path)
    result = _specs(skill)["routines.create"].handler(
        {
            "name": "Falsche Args",
            "metric": "battery.percent",
            "op": "lt",
            "value": 15,
            "tool": "media.play_pause",
            "arguments": {"bogus": 1},
        }
    )
    assert result["ok"] is False
    assert "Argumente" in result["error"]


def test_skill_list_toggle_delete(tmp_path) -> None:
    skill, repo = _skill(tmp_path)
    routine = build_routine(
        name="Liste",
        trigger=build_trigger("disk.used_percent", "gt", 90),
        tool_name="media.play_pause",
    )
    repo.add(routine)

    listed = _specs(skill)["routines.list"].handler({})
    assert listed["count"] == 1
    assert listed["routines"][0]["id"] == routine.id

    toggled = _specs(skill)["routines.toggle"].handler({"id": routine.id, "enabled": False})
    assert toggled["ok"] is True
    assert repo.get(routine.id).enabled is False

    deleted = _specs(skill)["routines.delete"].handler({"id": routine.id})
    assert deleted["ok"] is True
    assert _specs(skill)["routines.delete"].handler({"id": routine.id})["ok"] is False


def test_skill_write_tools_always_show_the_card(tmp_path) -> None:
    skill, _repo = _skill(tmp_path)
    specs = _specs(skill)
    valid_params = {
        "routines.create": {
            "name": "Akku",
            "metric": "battery.percent",
            "op": "lt",
            "value": 15,
            "tool": "media.play_pause",
            "arguments": {},
        },
        "routines.delete": {"id": "x"},
        "routines.toggle": {"id": "x", "enabled": True},
    }
    # Even jarvis must not create or delete routines without the recipe card.
    for name, params in valid_params.items():
        decision = ToolPolicy(AutonomyLevel.JARVIS.value).evaluate(
            name=name,
            params=params,
            spec=specs[name],
            panic=False,
            integrations_enabled=True,
            origin=Origin.MODEL,
        )
        assert decision.kind is DecisionKind.CONFIRM
