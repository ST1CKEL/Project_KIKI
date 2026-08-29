"""Volume tools: pactl parsing, clamping, and clean failures."""

from __future__ import annotations

import pytest

from kiki.tools.audio_tools import (
    AudioControlSkill,
    AudioError,
    parse_muted,
    parse_volume_percent,
)
from kiki.tools.policy import AutonomyLevel, DecisionKind, Origin, ToolPolicy


class FakeRunner:
    def __init__(self, outputs: dict[str, str] | None = None, *, error: Exception | None = None):
        self.outputs = outputs or {}
        self.error = error
        self.argv: list[list[str]] = []

    def __call__(self, argv: list[str]) -> str:
        self.argv.append(list(argv))
        if self.error:
            raise self.error
        key = " ".join(argv[:2])
        if key in self.outputs:
            return self.outputs[key]
        return ""

    def last(self, argv: list[str]) -> list[str] | None:
        for call in reversed(self.argv):
            if call[: len(argv)] == argv:
                return call
        return None


_VOLUME_OUT = (
    "Volume: front-left: 23593 /  36% / -26.62 dB,   front-right: 23593 /  36% / -26.62 dB\n"
)
_MUTE_OUT_YES = "Mute: yes\n"
_MUTE_OUT_NO = "Mute: no\n"


def _skill(runner: FakeRunner) -> dict[str, object]:
    return {spec.name: spec for spec in AudioControlSkill(runner).tools()}


def test_parse_volume_percent_takes_first_match() -> None:
    assert parse_volume_percent(_VOLUME_OUT) == 36
    assert parse_volume_percent("Volume: 0%") == 0
    assert parse_volume_percent("kaputt") is None


def test_parse_muted_understands_english_and_german_answers() -> None:
    assert parse_muted(_MUTE_OUT_YES) is True
    assert parse_muted(_MUTE_OUT_NO) is False
    assert parse_muted("Mute: ja\n") is True
    assert parse_muted("Mute: nein\n") is False
    assert parse_muted("nichts") is None


def test_volume_get_reports_percent_and_mute() -> None:
    runner = FakeRunner(
        {
            "get-sink-volume @DEFAULT_SINK@": _VOLUME_OUT,
            "get-sink-mute @DEFAULT_SINK@": _MUTE_OUT_NO,
        }
    )
    result = _skill(runner)["audio.volume_get"].handler({})
    assert result == {"ok": True, "percent": 36, "muted": False}


def test_volume_get_returns_error_on_runner_failure() -> None:
    runner = FakeRunner(error=AudioError("pactl fehlt"))
    result = _skill(runner)["audio.volume_get"].handler({})
    assert result == {"ok": False, "error": "pactl fehlt"}


def test_volume_get_rejects_unparsable_output() -> None:
    runner = FakeRunner({})
    result = _skill(runner)["audio.volume_get"].handler({})
    assert result["ok"] is False


def test_volume_set_passes_clamped_percent() -> None:
    runner = FakeRunner()
    specs = _skill(runner)
    assert specs["audio.volume_set"].handler({"percent": 55}) == {"ok": True, "percent": 55}
    assert specs["audio.volume_set"].handler({"percent": 140}) == {"ok": True, "percent": 100}
    assert specs["audio.volume_set"].handler({"percent": -3}) == {"ok": True, "percent": 0}
    assert runner.argv == [
        ["set-sink-volume", "@DEFAULT_SINK@", "55%"],
        ["set-sink-volume", "@DEFAULT_SINK@", "100%"],
        ["set-sink-volume", "@DEFAULT_SINK@", "0%"],
    ]


def test_mute_passes_flag() -> None:
    runner = FakeRunner()
    specs = _skill(runner)
    assert specs["audio.mute"].handler({"muted": True}) == {"ok": True, "muted": True}
    assert specs["audio.mute"].handler({"muted": False}) == {"ok": True, "muted": False}
    assert runner.last(["set-sink-mute", "@DEFAULT_SINK@"]) == [
        "set-sink-mute",
        "@DEFAULT_SINK@",
        "0",
    ]


def test_volume_set_error_is_clean() -> None:
    runner = FakeRunner(error=AudioError("Verbindung fehlgeschlagen"))
    result = _skill(runner)["audio.volume_set"].handler({"percent": 10})
    assert result == {"ok": False, "error": "Verbindung fehlgeschlagen"}


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("audio.volume_set", {"percent": 40}),
        ("audio.mute", {"muted": True}),
    ],
)
def test_control_tools_are_unattended_in_balanced(name: str, params: dict) -> None:
    spec = _skill(FakeRunner())[name]
    decision = ToolPolicy(AutonomyLevel.BALANCED.value).evaluate(
        name=name,
        params=params,
        spec=spec,
        panic=False,
        integrations_enabled=True,
        origin=Origin.MODEL,
    )
    assert decision.kind is DecisionKind.ALLOW


def test_all_specs_are_model_callable() -> None:
    for spec in AudioControlSkill(FakeRunner()).tools():
        assert spec.model_callable is True
        assert spec.auto_allow is True
