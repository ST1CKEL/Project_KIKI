"""Brightness tools: backend cascade, scaling, and clean unavailability."""

from __future__ import annotations

from kiki.tools.display_tools import (
    BrightnessController,
    DisplayControlSkill,
    clamp_percent,
)
from kiki.tools.policy import AutonomyLevel, DecisionKind, Origin, RiskLevel, ToolPolicy


class FakeBackend:
    def __init__(
        self,
        name: str,
        *,
        available: bool = True,
        current: int = 0,
        maximum: int = 10000,
        get_error: Exception | None = None,
        set_error: Exception | None = None,
    ) -> None:
        self.name = name
        self._available = available
        self.current = current
        self.maximum = maximum
        self.get_error = get_error
        self.set_error = set_error
        self.set_values: list[int] = []

    def available(self) -> bool:
        return self._available

    def get(self) -> int:
        if self.get_error:
            raise self.get_error
        return clamp_percent(round(self.current * 100 / max(1, self.maximum)))

    def set(self, percent: int) -> int:
        if self.set_error:
            raise self.set_error
        value = clamp_percent(percent)
        self.set_values.append(value)
        self.current = round(self.maximum * value / 100)
        return value


def test_clamp_percent_bounds_values() -> None:
    assert clamp_percent(0) == 0
    assert clamp_percent(100) == 100
    assert clamp_percent(150) == 100
    assert clamp_percent(-5) == 0


def test_controller_prefers_first_available_backend() -> None:
    gnome = FakeBackend("gnome", available=False)
    kde = FakeBackend("kde", current=5000, maximum=10000)
    controller = BrightnessController((gnome, kde))
    percent, backend = controller.get()
    assert percent == 50
    assert backend == "kde"


def test_controller_without_backend_reports_error() -> None:
    controller = BrightnessController((FakeBackend("gnome", available=False),))
    percent, error = controller.get()
    assert percent is None
    assert "Keine Helligkeits" in error


def test_kde_scaling_maps_percent_into_device_range() -> None:
    kde = FakeBackend("kde", current=2500, maximum=10000)
    BrightnessController((kde,)).set(60)
    assert kde.current == 6000


def test_set_clamps_and_reports_applied_value() -> None:
    backend = FakeBackend("kde")
    result = BrightnessController((backend,)).set(180)
    assert result == {"ok": True, "percent": 100, "backend": "kde"}


def test_set_error_is_clean() -> None:
    from kiki.tools.display_tools import BrightnessError

    backend = FakeBackend("kde", set_error=BrightnessError("D-Bus weg"))
    result = BrightnessController((backend,)).set(50)
    assert result["ok"] is False
    assert "D-Bus weg" in result["error"]


def _skill(controller: BrightnessController) -> dict[str, object]:
    return {spec.name: spec for spec in DisplayControlSkill(controller).tools()}


def test_brightness_get_reads_controller() -> None:
    controller = BrightnessController((FakeBackend("kde", current=7500, maximum=10000),))
    result = _skill(controller)["display.brightness_get"].handler({})
    assert result == {"ok": True, "percent": 75, "backend": "kde"}


def test_brightness_set_reads_controller() -> None:
    backend = FakeBackend("gnome")
    result = _skill(BrightnessController((backend,)))["display.brightness_set"].handler(
        {"percent": 30}
    )
    assert result["ok"] is True
    assert backend.set_values == [30]


def test_brightness_set_is_control_risk_and_unattended_in_balanced() -> None:
    spec = _skill(BrightnessController())["display.brightness_set"]
    assert spec.risk is RiskLevel.CONTROL
    decision = ToolPolicy(AutonomyLevel.BALANCED.value).evaluate(
        name=spec.name,
        params={"percent": 30},
        spec=spec,
        panic=False,
        integrations_enabled=True,
        origin=Origin.MODEL,
    )
    assert decision.kind is DecisionKind.ALLOW


def test_all_specs_are_model_callable() -> None:
    for spec in DisplayControlSkill(BrightnessController()).tools():
        assert spec.model_callable is True
        assert spec.auto_allow is True
