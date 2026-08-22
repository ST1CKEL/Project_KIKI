"""Pure preparation helpers for the bounded desktop-control surface.

This module deliberately has no GTK dependency.  It turns explicit user input
into the exact parameters and preview shown by the confirmation dialog; it
never executes an action.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiki.tools.desktop_tools import (
    validate_clipboard_text,
    validate_http_url,
    validate_notification,
)
from kiki.tools.registry import ActionPreview, ToolSpec
from kiki.tools.workspace_tools import (
    browser_open_spec,
    clipboard_copy_spec,
    desktop_notification_spec,
    terminal_open_spec,
    workspace_open_editor_spec,
    workspace_open_file_spec,
    workspace_open_spec,
)
from kiki.workspaces.models import Workspace
from kiki.workspaces.validator import resolve_inside_workspace

DESKTOP_CONTROL_TOOLS = (
    "workspace.open_in_file_manager",
    "workspace.open_file",
    "terminal.open_workspace",
    "workspace.open_in_editor",
    "browser.open_url",
    "desktop.copy_text",
    "desktop.show_notification",
)


@dataclass(frozen=True)
class PreparedDesktopAction:
    """A validated action ready to be bound to a one-time approval."""

    params: dict[str, Any]
    preview: ActionPreview


def _prepared(
    spec: ToolSpec,
    params: dict[str, Any],
    *,
    target: str,
    reason: str,
) -> PreparedDesktopAction:
    if spec.name not in DESKTOP_CONTROL_TOOLS:
        raise ValueError(f"desktop control tool not allowlisted: {spec.name}")
    return PreparedDesktopAction(
        params=params,
        preview=ActionPreview(
            tool=spec.name,
            title=spec.title,
            params=dict(params),
            target=target,
            effect=spec.effect,
            risk=spec.risk,
            reason=reason,
        ),
    )


def prepare_workspace_folder(workspace: Workspace) -> PreparedDesktopAction:
    return _workspace_action(
        workspace_open_spec(),
        workspace,
        reason="Manuell gewählter, registrierter Workspace; kein Pfad aus Modelltext.",
    )


def prepare_terminal(workspace: Workspace) -> PreparedDesktopAction:
    return _workspace_action(
        terminal_open_spec(),
        workspace,
        reason="Manueller Klick; feste Terminal-Vorlage ohne Kommando- oder Shellstring.",
    )


def prepare_editor(workspace: Workspace) -> PreparedDesktopAction:
    return _workspace_action(
        workspace_open_editor_spec(),
        workspace,
        reason="Manueller Klick; Allowlist-Editor erhält ausschließlich den Workspace-Pfad.",
    )


def _workspace_action(
    spec: ToolSpec,
    workspace: Workspace,
    *,
    reason: str,
) -> PreparedDesktopAction:
    root = Path(workspace.canonical_path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"workspace is not a directory: {root}")
    return _prepared(
        spec,
        {"workspace_id": workspace.id},
        target=str(root),
        reason=reason,
    )


def prepare_workspace_file(workspace: Workspace, raw_path: str) -> PreparedDesktopAction:
    root = Path(workspace.canonical_path).resolve(strict=True)
    target = resolve_inside_workspace(raw_path, root)
    relative = target.relative_to(root).as_posix()
    params = {"workspace_id": workspace.id, "path": relative}
    return _prepared(
        workspace_open_file_spec(),
        params,
        target=str(target),
        reason="Explizit gewählte Datei; kanonisch geprüft und innerhalb des Workspace gebunden.",
    )


def prepare_browser_url(raw_url: str) -> PreparedDesktopAction:
    url = validate_http_url(raw_url)
    return _prepared(
        browser_open_spec(),
        {"url": url},
        target=url,
        reason="Explizit eingegebene URL; nur http/https ohne eingebettete Zugangsdaten.",
    )


def prepare_clipboard_text(raw_text: str) -> PreparedDesktopAction:
    text = validate_clipboard_text(raw_text)
    return _prepared(
        clipboard_copy_spec(),
        {"text": text},
        target=f"Desktop-Zwischenablage ({len(text)} Zeichen)",
        reason="Sichtbarer, manuell eingegebener Text; keine Tastatur- oder Fenstersimulation.",
    )


def prepare_notification(title: str, body: str) -> PreparedDesktopAction:
    clean_title, clean_body = validate_notification(title, body)
    return _prepared(
        desktop_notification_spec(),
        {"title": clean_title, "body": clean_body},
        target="Lokale Desktop-Sitzung",
        reason="Manuell formulierter Inhalt; genau eine lokale Gio-Benachrichtigung.",
    )


_CONTROL_INTENTS = frozenset(
    {
        "öffne die pc steuerung",
        "öffne pc steuerung",
        "pc steuerung öffnen",
        "öffne die computersteuerung",
        "computersteuerung öffnen",
    }
)

# Small German Vosk models occasionally split ``PC-Steuerung`` or hear
# ``öffnen`` as ``lohnen``. Tolerating those bounded variants is safe because
# the intent can only expose the control surface; it can never run an action.
_CONTROL_OPEN_WORDS = frozenset(
    {"öffne", "öffnen", "oeffne", "oeffnen", "aufmachen", "zeige", "zeig", "lohnen"}
)
_CONTROL_ACTION_WORDS = frozenset(
    {
        "und",
        "terminal",
        "browser",
        "url",
        "datei",
        "ordner",
        "editor",
        "zwischenablage",
        "benachrichtigung",
        "start",
        "starte",
        "starten",
        "ausführen",
        "ausfuehren",
    }
)


def is_desktop_control_intent(text: str) -> bool:
    """Recognise only a bounded request to open the control window.

    This intentionally does not recognise action verbs such as opening a
    terminal or copying text.  Spoken input can expose the UI, never execute.
    """

    normalized = " ".join(re.sub(r"[^\wäöüß]+", " ", (text or "").casefold()).split())
    if normalized.startswith("kiki "):
        normalized = normalized.removeprefix("kiki ")
    if normalized in _CONTROL_INTENTS:
        return True

    tokens = normalized.split()
    if tokens[:2] in (["hallo", "kiki"], ["hey", "kiki"]):
        tokens = tokens[2:]
    elif tokens[:1] == ["kiki"]:
        tokens = tokens[1:]
    if not tokens or len(tokens) > 10 or _CONTROL_ACTION_WORDS.intersection(tokens):
        return False
    if any(token.startswith(("kopier", "benachrichtig")) for token in tokens):
        return False
    has_open = bool(_CONTROL_OPEN_WORDS.intersection(tokens))
    has_pc = "pc" in tokens or any(
        token.startswith(("pcsteuer", "computersteuer")) for token in tokens
    )
    has_control = any(
        token.startswith(("steuer", "pcsteuer", "computersteuer")) for token in tokens
    )
    return has_open and has_pc and has_control
