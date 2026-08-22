from __future__ import annotations

from pathlib import Path

import pytest

from kiki.agents.handoff import coding_task_from_chat, format_coding_briefing, session_summary_for_chat
from kiki.runners.podman import PodmanWorkspaceRunner
from kiki.runners.process import RunnerError
from kiki.tools.desktop_tools import (
    copy_text_to_clipboard,
    editor_argv,
    show_desktop_notification,
    terminal_argv,
    validate_clipboard_text,
    validate_http_url,
    validate_notification,
)
from kiki.tools.policy import DecisionKind, ToolPolicy
from kiki.tools.workspace_tools import (
    browser_open_spec,
    clipboard_copy_spec,
    desktop_notification_spec,
    terminal_open_spec,
    workspace_open_editor_spec,
    workspace_open_file_spec,
)
from kiki.workspaces.models import WorkspaceError
from kiki.workspaces.validator import resolve_inside_workspace


def test_https_url_ok() -> None:
    assert validate_http_url("https://example.com/docs") == "https://example.com/docs"


@pytest.mark.parametrize(
    "bad",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,hi",
        "https://user:secret@example.com/",
        "ftp://example.com",
        "https://example.com/a b",
        "",
    ],
)
def test_url_rejects_dangerous_schemes(bad: str) -> None:
    with pytest.raises(WorkspaceError) as exc:
        validate_http_url(bad)
    assert exc.value.code == "invalid_url"


def test_terminal_argv_is_fixed(tmp_path: Path) -> None:
    def which(name: str) -> str | None:
        if name == "kgx":
            return "/usr/bin/kgx"
        return None

    argv = terminal_argv(tmp_path, which=which)
    assert argv[0] == "/usr/bin/kgx"
    assert "-c" not in argv
    assert any(str(tmp_path.resolve()) in part for part in argv)


def test_path_must_stay_in_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "app"
    workspace.mkdir()
    inside = workspace / "readme.txt"
    inside.write_text("hi\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("no\n", encoding="utf-8")
    assert resolve_inside_workspace("readme.txt", workspace) == inside.resolve()
    with pytest.raises(WorkspaceError) as exc:
        resolve_inside_workspace(str(tmp_path / "secret.txt"), workspace)
    assert exc.value.code == "outside_root"
    escape = workspace / "link"
    escape.symlink_to(tmp_path / "secret.txt")
    with pytest.raises(WorkspaceError):
        resolve_inside_workspace(str(escape), workspace)


def test_handoff_prefers_draft() -> None:
    assert coding_task_from_chat(draft="  neue aufgabe  ", last_user="alt") == "neue aufgabe"
    assert coding_task_from_chat(draft="", last_user="alt") == "alt"
    with pytest.raises(ValueError):
        coding_task_from_chat(draft="  ", last_user="")
    brief = format_coding_briefing("Login bauen")
    assert brief.startswith("Aufgabe: Login bauen")
    assert "Akzeptanzkriterien" in brief
    summary = session_summary_for_chat(
        type("S", (), {"kind": type("K", (), {"value": "plan"})(), "status": type("T", (), {"value": "finished"})(), "git_branch_before": "main", "exit_code": 0, "summary": "ok"})(),
        dirty=True,
    )
    assert "nicht vom Chat-Modell erzeugt" in summary
    assert "uncommitted" in summary


def test_editor_argv_allowlist(tmp_path: Path) -> None:
    def which(name: str) -> str | None:
        if name == "code":
            return "/usr/bin/code"
        return None

    argv = editor_argv(tmp_path, which=which)
    assert argv[0] == "/usr/bin/code"
    assert "-c" not in argv
    assert any(str(tmp_path.resolve()) in part for part in argv)

    def none(_name: str) -> str | None:
        return None

    with pytest.raises(WorkspaceError) as exc:
        editor_argv(tmp_path, which=none)
    assert exc.value.code == "no_handler"


def test_podman_runner_fail_closed(tmp_path: Path) -> None:
    from kiki.workspaces.models import Workspace

    dummy = Workspace(
        id="w",
        display_name="w",
        canonical_path=str(tmp_path),
        remote_url=None,
        active_branch="main",
        git_head=None,
        risk_profile="observe",
        created_at="",
        last_used_at="",
    )

    async def _run() -> None:
        with pytest.raises(RunnerError) as extra:
            await PodmanWorkspaceRunner().run_argv(["echo"], workspace=dummy)
        assert extra.value.code == "unavailable"

    import asyncio

    asyncio.run(_run())


def test_desktop_tools_need_confirm() -> None:
    policy = ToolPolicy()
    for spec, params in (
        (terminal_open_spec(), {"workspace_id": "w"}),
        (workspace_open_file_spec(), {"workspace_id": "w", "path": "a.py"}),
        (browser_open_spec(), {"url": "https://example.com"}),
        (workspace_open_editor_spec(), {"workspace_id": "w"}),
        (clipboard_copy_spec(), {"text": "sichtbarer Text"}),
        (desktop_notification_spec(), {"title": "KIKI", "body": "Fertig"}),
    ):
        decision = policy.evaluate(
            name=spec.name,
            params=params,
            spec=spec,
            panic=False,
            integrations_enabled=True,
            profile="observe",
        )
        assert decision.kind is DecisionKind.CONFIRM


def test_clipboard_action_is_bounded_and_uses_callback() -> None:
    copied: list[str] = []
    assert validate_clipboard_text("  Text bleibt exakt  ") == "  Text bleibt exakt  "
    assert copy_text_to_clipboard("Hallo", setter=copied.append) == 5
    assert copied == ["Hallo"]
    for bad in ("", "   ", "nul\x00byte", "x" * 8193):
        with pytest.raises(WorkspaceError) as exc:
            validate_clipboard_text(bad)
        assert exc.value.code == "invalid_text"


def test_notification_action_is_bounded_and_local() -> None:
    sent: list[tuple[str, str]] = []
    assert validate_notification("  KIKI  ", "  Aufgabe   fertig  ") == (
        "KIKI",
        "Aufgabe fertig",
    )
    rendered = show_desktop_notification(
        "KIKI",
        "Fertig",
        sender=lambda title, body: sent.append((title, body)),
    )
    assert rendered == ("KIKI", "Fertig")
    assert sent == [("KIKI", "Fertig")]
    with pytest.raises(WorkspaceError):
        validate_notification("", "Text")
    with pytest.raises(WorkspaceError):
        validate_notification("KIKI", "x" * 401)
