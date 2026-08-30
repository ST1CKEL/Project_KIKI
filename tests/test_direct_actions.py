"""Explicit direct launches are bound locally before any model is consulted."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import kiki.tools.app_launch_tools as app_tools
import kiki.tools.steam_launch_tools as steam_tools
from kiki.ai.chat_service import ChatService
from kiki.runtime.event_bus import EventBus
from kiki.tools.app_launch_tools import AppLaunchSkill, DesktopIndex
from kiki.tools.direct_actions import (
    DirectActionService,
    LaunchAction,
    LaunchRoute,
    _best_fuzzy,
    parse_direct_launch,
)
from kiki.tools.gateway import ToolGateway
from kiki.tools.steam_launch_tools import SteamIndex, SteamLaunchSkill


def _desktop(directory: Path, app_id: str, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{app_id}.desktop").write_text(
        f"[Desktop Entry]\nType=Application\nName={name}\nExec=/bin/true\n",
        encoding="utf-8",
    )


def _manifest(directory: Path, app_id: str, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"appmanifest_{app_id}.acf").write_text(
        f'"AppState"\n{{\n"appid" "{app_id}"\n"name" "{name}"\n}}\n',
        encoding="utf-8",
    )


def test_parser_accepts_one_imperative_and_rejects_compound_or_urls() -> None:
    assert parse_direct_launch("Starte Firefox").target == "Firefox"
    steam = parse_direct_launch("KIKI, öffne Hades über Steam!")
    assert steam.target == "Hades"
    assert steam.route is LaunchRoute.STEAM
    assert parse_direct_launch("Starte Firefox und dann das Terminal") is None
    assert parse_direct_launch("Öffne https://example.com") is None
    assert parse_direct_launch("Kannst du Firefox starten?") is None


@pytest.fixture
def direct_environment(tmp_path, tools_env, settings, monkeypatch):
    app_dir = tmp_path / "applications"
    steam_dir = tmp_path / "steamapps"
    _desktop(app_dir, "firefox", "Firefox")
    _desktop(app_dir, "org.thunderbird.Thunderbird", "Thunderbird")
    _desktop(app_dir, "firefox-esr", "Firefox ESR")
    _manifest(steam_dir, "1145360", "Hades")
    applications = DesktopIndex([app_dir])
    games = SteamIndex([steam_dir])
    registry, executor = tools_env
    for skill in (AppLaunchSkill(applications), SteamLaunchSkill(games)):
        for spec in skill.tools():
            registry.register(spec)
    app_launches: list[list[str]] = []
    steam_launches: list[list[str]] = []
    binaries = {"gio": "/usr/bin/gio", "steam": "/usr/bin/steam"}
    monkeypatch.setattr(app_tools.shutil, "which", binaries.get)
    monkeypatch.setattr(app_tools, "_launch", lambda argv: app_launches.append(list(argv)))
    monkeypatch.setattr(steam_tools, "_launch", lambda argv: steam_launches.append(list(argv)))
    gateway = ToolGateway(
        executor,
        panic_check=lambda: settings.app.privacy_panic,
        integrations_check=settings.integrations_active,
    )
    return DirectActionService(gateway, applications, games), app_launches, steam_launches


def test_explicit_app_and_game_launch_exactly_the_resolved_ids(direct_environment) -> None:
    service, apps, games = direct_environment
    app_result = asyncio.run(service.execute(parse_direct_launch("Starte Firefox")))
    game_result = asyncio.run(service.execute(parse_direct_launch("Starte Hades")))
    assert app_result.ok is True and app_result.tool == "app.open"
    assert game_result.ok is True and game_result.tool == "steam.launch"
    assert apps[0][0:2] == ["/usr/bin/gio", "launch"]
    assert games == [["/usr/bin/steam", "-applaunch", "1145360"]]


def test_unknown_direct_target_does_not_launch_anything(direct_environment) -> None:
    service, apps, games = direct_environment
    result = asyncio.run(service.execute(parse_direct_launch("Starte NichtVorhanden")))
    assert result.ok is False
    assert apps == [] and games == []


def test_fuzzy_fallback_resolves_misheard_app_names(direct_environment) -> None:
    service, apps, _games = direct_environment
    for heard in ("öffne sander bord", "starte sander bird", "öffne sonderboard"):
        result = asyncio.run(service.execute(parse_direct_launch(heard)))
        assert result.ok is True, heard
        assert result.tool == "app.open"
        assert "Thunderbird" in result.answer
    assert len(apps) == 3
    assert all("Thunderbird" in argv[2] for argv in apps)


def test_fuzzy_searches_the_full_index_not_the_capped_list(
    direct_environment, monkeypatch
) -> None:
    import kiki.tools.app_launch_tools as app_tools

    monkeypatch.setattr(app_tools, "_MAX_RESULTS", 1)
    service, apps, _games = direct_environment
    result = asyncio.run(service.execute(parse_direct_launch("öffne sander bord")))
    assert result.ok is True
    assert "Thunderbird" in result.answer


def test_ambiguous_fuzzy_matches_stay_unresolved(direct_environment) -> None:
    service, apps, _games = direct_environment
    # Firefox and Firefox ESR are too similar to pick a clear winner.
    result = asyncio.run(service.execute(parse_direct_launch("starte feuerfuchs")))
    assert result.ok is False
    assert apps == []


def test_fuzzy_matcher_rules() -> None:
    candidates = [("tb", "Thunderbird"), ("ff", "Firefox")]
    assert _best_fuzzy(candidates, "sander bord") == "tb"
    assert _best_fuzzy(candidates, "thunderbird") == "tb"
    # No candidate is close enough.
    assert _best_fuzzy(candidates, "maschinenbau") is None
    # Two near-equal candidates never resolve.
    assert (
        _best_fuzzy([("a", "Firefox"), ("b", "Firefox ESR")], "feuerfuchs") is None
    )
    # A heard name below four letters stays deterministic-only.
    assert _best_fuzzy(candidates, "ok") is None


def test_direct_chat_bypasses_the_provider_and_keeps_history(
    direct_environment, settings, chats, secrets, tools_env
) -> None:
    direct, apps, _games = direct_environment

    class ProviderMustStayUnused:
        id = "unused"

        async def stream_chat(self, *_args, **_kwargs):
            raise AssertionError("direct action must not reach a model")
            yield ""  # pragma: no cover

    _registry, executor = tools_env
    service = ChatService(
        settings,
        chats,
        secrets,
        EventBus(),
        executor,
        direct_actions=direct,
    )
    service._provider = ProviderMustStayUnused()
    conversation = service.ensure_conversation(None)

    async def _send():
        return [event async for event in service.send(conversation.id, "Starte Firefox")]

    events = asyncio.run(_send())
    assert [event.kind for event in events] == ["delta", "done"]
    assert events[-1].text == "Ich starte Firefox."
    assert len(apps) == 1
    history = chats.history(conversation.id)
    assert [message.role for message in history] == ["user", "assistant"]
    assert "app.open" in history[-1].content


# --- closing apps -------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    ["beende thunderbird", "schließe den Firefox", "KIKI, beende bitte Thunderbird."],
)
def test_close_commands_parse_with_close_action(phrase) -> None:
    request = parse_direct_launch(phrase)
    assert request is not None
    assert request.action is LaunchAction.CLOSE
    assert request.target in {"thunderbird", "Firefox", "Thunderbird"}


def test_open_commands_keep_the_open_action() -> None:
    request = parse_direct_launch("starte thunderbird")
    assert request.action is LaunchAction.OPEN


def test_close_dispatch_sends_only_a_polite_signal(
    direct_environment, monkeypatch
) -> None:
    import signal as signal_module

    import kiki.tools.app_launch_tools as app_tools

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(app_tools.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(app_tools, "matching_pids", lambda binary, **_: [12345])
    monkeypatch.setattr(app_tools.time, "sleep", lambda _s: None)
    service, _apps, _games = direct_environment
    result = asyncio.run(service.execute(parse_direct_launch("beende firefox")))
    assert result.ok is True
    assert result.tool == "app.close"
    assert signals == [(12345, signal_module.SIGTERM)]


def test_close_of_unknown_app_launches_nothing(direct_environment) -> None:
    service, _apps, _games = direct_environment
    result = asyncio.run(service.execute(parse_direct_launch("beende nichtvorhanden")))
    assert result.ok is False
    assert result.tool == ""


def test_closing_steam_games_is_refused(direct_environment) -> None:
    service, _apps, games = direct_environment
    result = asyncio.run(service.execute(parse_direct_launch("beende hades über steam")))
    assert result.ok is False
    assert result.tool == ""
    assert games == []
