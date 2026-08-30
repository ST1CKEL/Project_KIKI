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
from kiki.tools.direct_actions import DirectActionService, LaunchRoute, parse_direct_launch
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
