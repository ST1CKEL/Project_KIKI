"""Steam discovery and launch stay local, numeric and shell-free."""

from __future__ import annotations

from pathlib import Path

import pytest

import kiki.tools.steam_launch_tools as steam_tools
from kiki.tools.policy import DecisionKind, Origin, RiskLevel, ToolPolicy
from kiki.tools.steam_launch_tools import SteamIndex, SteamLaunchSkill


def _manifest(directory: Path, app_id: str, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"appmanifest_{app_id}.acf"
    path.write_text(
        f'"AppState"\n{{\n  "appid"  "{app_id}"\n  "name"  "{name}"\n}}\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def steamapps(tmp_path: Path) -> Path:
    directory = tmp_path / "SteamLibrary" / "steamapps"
    _manifest(directory, "1145360", "Hades")
    _manifest(directory, "620", "Portal 2")
    (directory / "appmanifest_broken.acf").write_text('"appid" "oops"', encoding="utf-8")
    return directory


def _specs(index: SteamIndex) -> dict[str, object]:
    return {spec.name: spec for spec in SteamLaunchSkill(index).tools()}


def test_index_reads_only_valid_local_manifests(steamapps: Path) -> None:
    index = SteamIndex([steamapps])
    assert [(game.app_id, game.name) for game in index.list()] == [
        ("1145360", "Hades"),
        ("620", "Portal 2"),
    ]
    assert index.find("hades").app_id == "1145360"
    assert index.find("Portal").app_id == "620"
    assert index.find("unbekannt") is None


def test_launch_uses_fixed_native_steam_argv(
    steamapps: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(steam_tools.shutil, "which", lambda name: "/usr/bin/steam" if name == "steam" else None)
    monkeypatch.setattr(steam_tools, "_launch", lambda argv: launched.append(list(argv)))
    result = _specs(SteamIndex([steamapps]))["steam.launch"].handler({"app_id": "1145360"})
    assert result == {"ok": True, "app_id": "1145360", "name": "Hades"}
    assert launched == [["/usr/bin/steam", "-applaunch", "1145360"]]


def test_unknown_id_never_starts_steam(
    steamapps: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(steam_tools, "_launch", lambda argv: launched.append(list(argv)))
    result = _specs(SteamIndex([steamapps]))["steam.launch"].handler({"app_id": "999"})
    assert result["ok"] is False
    assert launched == []


def test_steam_launch_is_local_launch_risk(steamapps: Path) -> None:
    spec = _specs(SteamIndex([steamapps]))["steam.launch"]
    assert spec.risk is RiskLevel.LAUNCH
    decision = ToolPolicy().evaluate(
        name=spec.name,
        params={"app_id": "1145360"},
        spec=spec,
        panic=False,
        integrations_enabled=True,
        origin=Origin.USER,
    )
    assert decision.kind is DecisionKind.ALLOW
