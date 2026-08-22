from __future__ import annotations

from pathlib import Path

from kiki.config.settings import default_mapping, load_settings, settings_from_mapping
from kiki.skills.desktop import DesktopPerceptionSkill
from kiki.tools.policy import DecisionKind, ToolPolicy
from kiki.tools.registry import ToolRegistry


def test_screenshot_requires_confirmation() -> None:
    registry = ToolRegistry()
    registry.register(DesktopPerceptionSkill().tools()[0])
    spec = registry.get("capture_screen")
    decision = ToolPolicy().evaluate(
        name="capture_screen",
        params={},
        spec=spec,
        panic=False,
        integrations_enabled=True,
    )
    assert decision.kind is DecisionKind.CONFIRM


def test_screenshot_denied_in_panic() -> None:
    spec = DesktopPerceptionSkill().tools()[0]
    decision = ToolPolicy().evaluate(
        name="capture_screen",
        params={},
        spec=spec,
        panic=True,
        integrations_enabled=True,
    )
    assert decision.kind is DecisionKind.DENY


def test_model_stub_does_not_capture() -> None:
    spec = DesktopPerceptionSkill().tools()[0]
    result = spec.handler({})
    assert result["ok"] is False


def test_voice_and_screenshot_defaults(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "missing.toml")
    assert settings.screenshot.enabled is True
    assert settings.voice.enabled is True
    assert settings.voice.auto_send is True
    assert settings.tts.enabled is True
    assert settings.tts.speaker == "Serena"
    assert settings.screenshot_allowed() is True
    settings.app.privacy_panic = True
    assert settings.screenshot_allowed() is False
    assert settings.voice_allowed() is False
    assert settings.tts_allowed() is False


def test_voice_can_be_disabled() -> None:
    data = default_mapping()
    data["voice"]["enabled"] = False
    settings = settings_from_mapping(data)
    assert settings.voice.enabled is False
    assert settings.voice_allowed() is False
