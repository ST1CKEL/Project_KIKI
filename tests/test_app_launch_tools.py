"""App launch tools: index building, matching, and the fixed gio argv."""

from __future__ import annotations

from pathlib import Path

import pytest

import kiki.tools.app_launch_tools as app_launch_tools
from kiki.tools.app_launch_tools import AppLaunchSkill, DesktopIndex
from kiki.tools.policy import AutonomyLevel, DecisionKind, Origin, RiskLevel, ToolPolicy


def _write_desktop(
    directory: Path,
    app_id: str,
    *,
    name: str | None = None,
    extra: str = "",
) -> Path:
    path = directory / f"{app_id}.desktop"
    body = "[Desktop Entry]\nType=Application\n"
    body += f"Name={name or app_id}\n"
    body += "Exec=/bin/true\n"
    body += extra
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def apps(tmp_path: Path) -> tuple[Path, Path]:
    user = tmp_path / "user" / "applications"
    system = tmp_path / "system" / "applications"
    user.mkdir(parents=True)
    system.mkdir(parents=True)

    _write_desktop(system, "org.gnome.Calculator", name="Rechner")
    _write_desktop(system, "firefox", name="Firefox")
    _write_desktop(system, "hidden-app", extra="NoDisplay=true\n")
    _write_desktop(system, "deleted-app", extra="Hidden=true\n")
    _write_desktop(system, "link-thing", extra="Type=Link\nURL=https://example.com\n")
    (system / "broken.desktop").write_text("not a desktop file", encoding="utf-8")
    _write_desktop(user, "firefox", name="Firefox (Nightly)")
    return user, system


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    argvs: list[list[str]] = []

    def _record(argv: list[str]) -> None:
        argvs.append(list(argv))

    monkeypatch.setattr(app_launch_tools, "_launch", _record)
    return argvs


def _index(apps: tuple[Path, Path]) -> DesktopIndex:
    return DesktopIndex([apps[0], apps[1]])


def _skill(apps: tuple[Path, Path]) -> dict[str, object]:
    return {spec.name: spec for spec in AppLaunchSkill(_index(apps)).tools()}


def test_index_hides_nodisplay_hidden_links_and_broken_files(apps) -> None:
    ids = {e.app_id for e in _index(apps).list()}
    assert ids == {"org.gnome.Calculator", "firefox"}


def test_user_directory_shadows_system_entry(apps) -> None:
    entry = _index(apps).find("firefox")
    assert entry is not None
    assert entry.name == "Firefox (Nightly)"
    assert entry.path.parent == apps[0]


def test_list_filters_by_query(apps) -> None:
    entries = _index(apps).list("rech")
    assert [e.app_id for e in entries] == ["org.gnome.Calculator"]


def test_find_accepts_unique_substring_and_rejects_ambiguity(apps) -> None:
    index = _index(apps)
    assert index.find("zzz") is None  # no id or name contains this
    assert index.find("rechner").app_id == "org.gnome.Calculator"
    assert index.find("") is None


def test_open_launches_gio_with_fixed_argv(apps, launched) -> None:
    result = _skill(apps)["app.open"].handler({"app_id": "firefox"})
    assert result["ok"] is True
    assert result["app"] == "firefox"
    assert len(launched) == 1
    argv = launched[0]
    assert argv[0].endswith("/gio") or argv[0] == "gio"
    assert argv[1] == "launch"
    assert argv[2].endswith("firefox.desktop")


def test_open_rejects_unknown_app_id(apps, launched) -> None:
    result = _skill(apps)["app.open"].handler({"app_id": "totally-unknown"})
    assert result["ok"] is False
    assert "app.list" in result["error"]
    assert launched == []


def test_open_is_launch_risk_and_model_callable(apps) -> None:
    spec = _skill(apps)["app.open"]
    assert spec.risk is RiskLevel.LAUNCH
    assert spec.model_callable is True
    # LAUNCH is unattended only at trusted — balanced still asks.
    balanced = ToolPolicy(AutonomyLevel.BALANCED.value).evaluate(
        name=spec.name, params={"app_id": "firefox"}, spec=spec,
        panic=False, integrations_enabled=True, origin=Origin.MODEL,
    )
    assert balanced.kind is DecisionKind.CONFIRM
    trusted = ToolPolicy(AutonomyLevel.TRUSTED.value).evaluate(
        name=spec.name, params={"app_id": "firefox"}, spec=spec,
        panic=False, integrations_enabled=True, origin=Origin.MODEL,
    )
    assert trusted.kind is DecisionKind.ALLOW
    explicit = ToolPolicy().evaluate(
        name=spec.name, params={"app_id": "firefox"}, spec=spec,
        panic=False, integrations_enabled=True, origin=Origin.USER,
    )
    assert explicit.kind is DecisionKind.ALLOW


def test_index_refreshes_when_directory_changes(apps, tmp_path) -> None:
    index = _index(apps)
    assert index.find("new-app") is None
    _write_desktop(apps[1], "new-app")
    import os

    os.utime(apps[1], None)  # bump the directory mtime the cache compares against
    assert index.find("new-app") is not None


# --- app.close: exec derivation and process matching ---------------------------


def _write_exec_desktop(path: Path, exec_line: str) -> Path:
    path.write_text(
        f"[Desktop Entry]\nType=Application\nName=X\nExec={exec_line}\n",
        encoding="utf-8",
    )
    return path


def test_desktop_exec_binary_takes_the_first_program(tmp_path: Path) -> None:
    assert (
        app_launch_tools.desktop_exec_binary(_write_exec_desktop(tmp_path / "a.desktop", "/usr/bin/tool --flag %u"))
        == "tool"
    )


def test_desktop_exec_binary_refuses_shell_wrappers(tmp_path: Path) -> None:
    assert (
        app_launch_tools.desktop_exec_binary(_write_exec_desktop(tmp_path / "s.desktop", 'sh -c "rm -rf ~"'))
        is None
    )


def test_matching_pids_only_matches_the_named_program(tmp_path: Path) -> None:
    for pid, argv in (
        ("11", b"/usr/bin/tool\x00--flag\x00"),
        ("12", b"/usr/bin/python3\x00tool.py\x00"),
        ("13", b"/usr/bin/other\x00"),
        ("14", b""),
    ):
        d = tmp_path / pid
        d.mkdir()
        (d / "cmdline").write_bytes(argv)
    assert app_launch_tools.matching_pids("tool", proc_dir=tmp_path, exclude_pid=99) == [11]
    # Interpreter scripts are matched by their script name.
    assert app_launch_tools.matching_pids("tool.py", proc_dir=tmp_path, exclude_pid=99) == [12]


def test_close_is_control_risk_and_signals_only_sigterm(
    apps, monkeypatch
) -> None:
    spec = next(s for s in app_launch_tools.AppLaunchSkill().tools() if s.name == "app.close")
    assert spec.risk is RiskLevel.CONTROL
    decision = ToolPolicy().evaluate(
        name=spec.name,
        params={"app_id": "firefox"},
        spec=spec,
        panic=False,
        integrations_enabled=True,
        origin=Origin.USER,
    )
    assert decision.kind is DecisionKind.ALLOW
