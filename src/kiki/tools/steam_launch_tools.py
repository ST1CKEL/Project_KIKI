"""Discover and launch locally installed Steam games without URL dispatch.

Only local ``appmanifest_*.acf`` files supply game ids and names. Launching
uses Steam's fixed ``-applaunch <numeric-id>`` argv (native or Flatpak); no
game name, model text, shell fragment or ``steam://`` URL reaches a process.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiki.runners.process import desktop_env
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec

_CACHE_TTL_S = 300.0
_MAX_RESULTS = 50
_APP_ID = re.compile(r'"appid"\s*"([0-9]{1,12})"', re.IGNORECASE)
_NAME = re.compile(r'"name"\s*"((?:\\.|[^"\\])*)"', re.IGNORECASE)
_LIBRARY_PATH = re.compile(r'"path"\s*"((?:\\.|[^"\\])*)"', re.IGNORECASE)


def _unescape_vdf(value: str) -> str:
    return value.replace(r"\"", '"').replace(r"\\", "\\")


def default_steamapps_dirs(home: Path | None = None) -> list[Path]:
    """Return native, Flatpak and configured Steam library directories."""
    root = home or Path.home()
    primary = [
        root / ".local" / "share" / "Steam" / "steamapps",
        root / ".steam" / "steam" / "steamapps",
        root
        / ".var"
        / "app"
        / "com.valvesoftware.Steam"
        / ".local"
        / "share"
        / "Steam"
        / "steamapps",
    ]
    found: list[Path] = []

    def _add(path: Path) -> None:
        expanded = path.expanduser()
        if expanded not in found:
            found.append(expanded)

    for directory in primary:
        _add(directory)
        config = directory / "libraryfolders.vdf"
        try:
            text = config.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _LIBRARY_PATH.finditer(text):
            _add(Path(_unescape_vdf(match.group(1))) / "steamapps")
    return found


@dataclass(frozen=True)
class SteamGame:
    app_id: str
    name: str
    manifest: Path


def _parse_manifest(path: Path) -> SteamGame | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    app_id = _APP_ID.search(text)
    name = _NAME.search(text)
    if app_id is None or name is None:
        return None
    clean_name = _unescape_vdf(name.group(1)).strip()
    if not clean_name:
        return None
    return SteamGame(app_id=app_id.group(1), name=clean_name, manifest=path)


class SteamIndex:
    """Cached, local-only index of installed Steam manifests."""

    def __init__(self, dirs: list[Path] | None = None, *, clock=time.monotonic) -> None:
        self._dirs = dirs if dirs is not None else default_steamapps_dirs()
        self._clock = clock
        self._games: dict[str, SteamGame] = {}
        self._built_at = -_CACHE_TTL_S
        self._stamp: tuple[float | None, ...] | None = None

    def _directory_stamp(self) -> tuple[float | None, ...]:
        values: list[float | None] = []
        for directory in self._dirs:
            try:
                values.append(directory.stat().st_mtime)
            except OSError:
                values.append(None)
        return tuple(values)

    def _stale(self) -> bool:
        return (
            self._stamp is None
            or self._clock() - self._built_at >= _CACHE_TTL_S
            or self._directory_stamp() != self._stamp
        )

    def _rebuild(self) -> None:
        games: dict[str, SteamGame] = {}
        for directory in self._dirs:
            try:
                manifests = sorted(directory.glob("appmanifest_*.acf"))
            except OSError:
                continue
            for path in manifests:
                game = _parse_manifest(path)
                if game is not None:
                    games.setdefault(game.app_id, game)
        self._games = games
        self._stamp = self._directory_stamp()
        self._built_at = self._clock()

    def _current(self) -> dict[str, SteamGame]:
        if self._stale():
            self._rebuild()
        return self._games

    def list(self, query: str | None = None) -> list[SteamGame]:
        games = sorted(self._current().values(), key=lambda game: game.name.casefold())
        needle = (query or "").strip().casefold()
        if needle:
            games = [
                game
                for game in games
                if needle in game.name.casefold() or needle == game.app_id
            ]
        return games[:_MAX_RESULTS]

    def find(self, value: str) -> SteamGame | None:
        needle = value.strip().casefold()
        if not needle:
            return None
        exact_id = self._current().get(value.strip())
        if exact_id is not None:
            return exact_id
        exact_names = [game for game in self._current().values() if game.name.casefold() == needle]
        if len(exact_names) == 1:
            return exact_names[0]
        matches = [game for game in self._current().values() if needle in game.name.casefold()]
        return matches[0] if len(matches) == 1 else None


def _steam_argv(app_id: str, *, home: Path | None = None) -> list[str] | None:
    native = shutil.which("steam")
    if native:
        return [native, "-applaunch", app_id]
    flatpak = shutil.which("flatpak")
    root = home or Path.home()
    flatpak_data = root / ".var" / "app" / "com.valvesoftware.Steam"
    if flatpak and flatpak_data.is_dir():
        return [flatpak, "run", "com.valvesoftware.Steam", "-applaunch", app_id]
    return None


def _launch(argv: list[str]) -> None:
    env = desktop_env(home=os.environ.get("HOME") or "/tmp")
    subprocess.Popen(
        argv,
        cwd=str(Path.home()),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


class SteamLaunchSkill:
    id = "steam_launch"
    name = "Steam-Spiele"
    description = "Lokal installierte Steam-Spiele auflisten und über Steam starten."

    def __init__(self, index: SteamIndex | None = None) -> None:
        self._index = index or SteamIndex()

    def tools(self) -> list[ToolSpec]:
        return [self._list_spec(), self._launch_spec()]

    def _list_spec(self) -> ToolSpec:
        return ToolSpec(
            name="steam.list_installed",
            title="Installierte Steam-Spiele auflisten",
            description="Liest Namen und app_id aus lokalen Steam-Manifesten.",
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 128}
                },
                "additionalProperties": False,
            },
            handler=self._list,
            effect="Liest ausschließlich lokale appmanifest-Dateien.",
            target="lokale Steam-Bibliothek",
            auto_allow=True,
            requires_integration=False,
            model_callable=True,
        )

    def _launch_spec(self) -> ToolSpec:
        return ToolSpec(
            name="steam.launch",
            title="Steam-Spiel starten",
            description="Startet eine lokal installierte app_id über Steam -applaunch.",
            risk=RiskLevel.LAUNCH,
            parameters={
                "type": "object",
                "properties": {
                    "app_id": {"type": "string", "pattern": "^[0-9]{1,12}$"}
                },
                "required": ["app_id"],
                "additionalProperties": False,
            },
            handler=self._open,
            effect="Öffnet das lokal installierte Spiel über den Steam-Client.",
            target="Steam",
            auto_allow=True,
            requires_integration=False,
            model_callable=True,
        )

    def _list(self, params: dict[str, Any]) -> dict[str, Any]:
        query = str(params["query"]) if params.get("query") else None
        games = self._index.list(query)
        return {
            "ok": True,
            "count": len(games),
            "games": [{"app_id": game.app_id, "name": game.name} for game in games],
        }

    def _open(self, params: dict[str, Any]) -> dict[str, Any]:
        game = self._index.find(str(params["app_id"]))
        if game is None:
            return {
                "ok": False,
                "error": "Steam-Spiel ist nicht lokal installiert oder nicht eindeutig.",
            }
        argv = _steam_argv(game.app_id)
        if argv is None:
            return {"ok": False, "error": "Weder Steam noch die Steam-Flatpak-App gefunden."}
        try:
            _launch(argv)
        except OSError as exc:
            return {"ok": False, "error": f"Steam-Spiel konnte nicht gestartet werden: {exc}"}
        return {"ok": True, "app_id": game.app_id, "name": game.name}
