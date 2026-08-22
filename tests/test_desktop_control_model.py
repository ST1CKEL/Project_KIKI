from __future__ import annotations

from pathlib import Path

import pytest

from kiki.ui.desktop_control_model import (
    DESKTOP_CONTROL_TOOLS,
    is_desktop_control_intent,
    prepare_browser_url,
    prepare_clipboard_text,
    prepare_editor,
    prepare_notification,
    prepare_terminal,
    prepare_workspace_file,
    prepare_workspace_folder,
)
from kiki.workspaces.models import Workspace, WorkspaceError


def _workspace(root: Path) -> Workspace:
    return Workspace(
        id="workspace-1",
        display_name="KIKI",
        canonical_path=str(root),
        remote_url=None,
        active_branch="main",
        git_head=None,
        risk_profile="observe",
        created_at="",
        last_used_at="",
    )


def test_control_surface_has_exactly_the_seven_bounded_tools(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    source = tmp_path / "src" / "main.py"
    source.parent.mkdir()
    source.write_text("print('ok')\n", encoding="utf-8")

    prepared = (
        prepare_workspace_folder(workspace),
        prepare_workspace_file(workspace, "src/main.py"),
        prepare_terminal(workspace),
        prepare_editor(workspace),
        prepare_browser_url("https://example.com/docs"),
        prepare_clipboard_text("sichtbarer Text"),
        prepare_notification("KIKI", "Fertig"),
    )

    assert tuple(action.preview.tool for action in prepared) == DESKTOP_CONTROL_TOOLS
    assert all(action.preview.params == action.params for action in prepared)
    assert all(action.preview.reason for action in prepared)


def test_file_preview_canonicalizes_inside_workspace_and_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "README.md"
    target.write_text("KIKI\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    workspace = _workspace(root)

    action = prepare_workspace_file(workspace, str(target))
    assert action.params == {"workspace_id": "workspace-1", "path": "README.md"}
    assert action.preview.target == str(target.resolve())

    with pytest.raises(WorkspaceError) as exc:
        prepare_workspace_file(workspace, str(outside))
    assert exc.value.code == "outside_root"

    link = root / "escape"
    link.symlink_to(outside)
    with pytest.raises(WorkspaceError):
        prepare_workspace_file(workspace, "escape")


def test_text_url_and_notification_are_validated_before_approval() -> None:
    clipboard = prepare_clipboard_text("  bleibt sichtbar  ")
    assert clipboard.params["text"] == "  bleibt sichtbar  "
    assert f"{len(clipboard.params['text'])} Zeichen" in clipboard.preview.target

    notification = prepare_notification("  KIKI  ", "  Test   fertig  ")
    assert notification.params == {"title": "KIKI", "body": "Test fertig"}

    with pytest.raises(WorkspaceError):
        prepare_browser_url("file:///etc/passwd")
    with pytest.raises(WorkspaceError):
        prepare_clipboard_text("   ")
    with pytest.raises(WorkspaceError):
        prepare_notification("", "Text")


@pytest.mark.parametrize(
    "text",
    (
        "Öffne die PC-Steuerung",
        "KIKI, PC-Steuerung öffnen!",
        "Computersteuerung öffnen",
        "Hallo KIKI, öffne bitte die PC Steuerung",
        "hallo kiki der die pc steuer lohnen",
    ),
)
def test_voice_intent_only_opens_control_surface(text: str) -> None:
    assert is_desktop_control_intent(text)


@pytest.mark.parametrize(
    "text",
    (
        "Öffne ein Terminal",
        "Öffne https://example.com",
        "Kopiere das in die Zwischenablage",
        "Zeige eine Benachrichtigung",
        "PC-Steuerung öffnen und Terminal starten",
        "PC-Steuerung öffnen und Datei kopieren",
    ),
)
def test_voice_intent_does_not_recognize_actions(text: str) -> None:
    assert not is_desktop_control_intent(text)
