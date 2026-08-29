"""MPRIS tools: player selection, Can*-gates, and the shaped status."""

from __future__ import annotations

from typing import Any

import pytest

from kiki.tools.media_tools import (
    MediaControlSkill,
    shape_player_properties,
    short_player_name,
)
from kiki.tools.policy import AutonomyLevel, DecisionKind, Origin, ToolPolicy
from kiki.tools.registry import ToolSpec


class FakeMprisClient:
    def __init__(
        self,
        players: dict[str, dict[str, Any]] | None = None,
        *,
        list_error: Exception | None = None,
    ) -> None:
        self.players = players or {}
        self.list_error = list_error
        self.calls: list[tuple[str, str]] = []

    def list_players(self) -> list[str]:
        if self.list_error:
            raise self.list_error
        return sorted(self.players)

    def player_properties(self, player: str) -> dict[str, Any]:
        if player not in self.players:
            raise RuntimeError("unbekannter Player")
        return self.players[player]

    def call_player_method(self, player: str, method: str) -> None:
        if player not in self.players:
            raise RuntimeError("unbekannter Player")
        self.calls.append((player, method))


def _props(
    *,
    status: str = "Paused",
    title: str | None = "Song",
    artist: list[str] | None = None,
    can_next: bool = True,
    can_pause: bool = True,
) -> dict[str, Any]:
    return {
        "PlaybackStatus": status,
        "CanPlay": True,
        "CanPause": can_pause,
        "CanGoNext": can_next,
        "CanGoPrevious": True,
        "Metadata": {
            "xesam:title": title,
            "xesam:artist": artist if artist is not None else ["A", "B"],
            "xesam:album": "Album",
            "mpris:length": 1_500_000,
        },
    }


def _specs() -> dict[str, ToolSpec]:
    skill = MediaControlSkill(FakeMprisClient())
    return {spec.name: spec for spec in skill.tools()}


def test_short_player_name_strips_instance_suffix() -> None:
    assert short_player_name("org.mpris.MediaPlayer2.firefox.instance_1_123") == "firefox"
    assert short_player_name("org.mpris.MediaPlayer2.vlc") == "vlc"


def test_shape_properties_trims_metadata() -> None:
    shaped = shape_player_properties(_props())
    assert shaped["title"] == "Song"
    assert shaped["artist"] == "A, B"
    assert shaped["length_s"] == 2
    assert shaped["can_go_next"] is True


def test_shape_properties_tolerates_string_artist() -> None:
    shaped = shape_player_properties(_props(artist="Solo"))
    assert shaped["artist"] == "Solo"


def test_status_lists_players_with_metadata() -> None:
    client = FakeMprisClient(
        {
            "org.mpris.MediaPlayer2.firefox": _props(status="Playing"),
            "org.mpris.MediaPlayer2.vlc": _props(status="Paused", title=None),
        }
    )
    result = MediaControlSkill(client).tools()[0].handler({})
    assert result["ok"] is True
    assert result["count"] == 2
    names = [p["player"] for p in result["players"]]
    assert names == ["firefox", "vlc"]
    assert result["players"][0]["playback_status"] == "Playing"


def test_status_without_players_is_ok() -> None:
    result = MediaControlSkill(FakeMprisClient()).tools()[0].handler({})
    assert result == {"ok": True, "count": 0, "players": []}


def test_status_reports_bus_error() -> None:
    client = FakeMprisClient(list_error=RuntimeError("kein Bus"))
    result = MediaControlSkill(client).tools()[0].handler({})
    assert result["ok"] is False
    assert "kein Bus" in result["error"]


def _control(client: FakeMprisClient, slug: str, params: dict | None = None) -> dict:
    skill = MediaControlSkill(client)
    specs = {spec.name: spec for spec in skill.tools()}
    return specs[f"media.{slug}"].handler(params or {})


def test_control_prefers_the_playing_player() -> None:
    client = FakeMprisClient(
        {
            "org.mpris.MediaPlayer2.firefox": _props(status="Playing"),
            "org.mpris.MediaPlayer2.vlc": _props(status="Paused"),
        }
    )
    result = _control(client, "play_pause")
    assert result == {"ok": True, "player": "firefox", "action": "playpause"}
    assert client.calls == [("org.mpris.MediaPlayer2.firefox", "PlayPause")]


def test_control_hint_selects_named_player() -> None:
    client = FakeMprisClient(
        {
            "org.mpris.MediaPlayer2.firefox": _props(status="Playing"),
            "org.mpris.MediaPlayer2.vlc": _props(status="Paused"),
        }
    )
    result = _control(client, "next", {"player": "vlc"})
    assert result["ok"] is True
    assert result["player"] == "vlc"
    assert client.calls == [("org.mpris.MediaPlayer2.vlc", "Next")]


def test_control_unknown_hint_fails_cleanly() -> None:
    client = FakeMprisClient({"org.mpris.MediaPlayer2.vlc": _props()})
    result = _control(client, "next", {"player": "spotify"})
    assert result["ok"] is False
    assert client.calls == []


def test_control_respects_can_go_next_false() -> None:
    client = FakeMprisClient(
        {"org.mpris.MediaPlayer2.firefox": _props(can_next=False)}
    )
    result = _control(client, "next")
    assert result["ok"] is False
    assert "nicht möglich" in result["error"]
    assert client.calls == []


def test_control_without_players_fails_cleanly() -> None:
    result = _control(FakeMprisClient(), "stop")
    assert result["ok"] is False
    assert "Kein Medienplayer" in result["error"]


def test_specs_are_model_callable_and_allowlisted() -> None:
    specs = _specs()
    assert set(specs) == {
        "media.status",
        "media.play_pause",
        "media.next",
        "media.previous",
        "media.stop",
    }
    for spec in specs.values():
        assert spec.model_callable is True
        assert spec.auto_allow is True


@pytest.mark.parametrize("slug", ["play_pause", "next", "previous", "stop"])
def test_control_risk_is_unattended_in_balanced(slug: str) -> None:
    spec = _specs()[f"media.{slug}"]
    policy = ToolPolicy(AutonomyLevel.BALANCED.value)
    decision = policy.evaluate(
        name=spec.name,
        params={},
        spec=spec,
        panic=False,
        integrations_enabled=True,
        origin=Origin.MODEL,
    )
    assert decision.kind is DecisionKind.ALLOW
