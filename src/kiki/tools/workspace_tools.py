"""Declared workspace tools. Paths come from the registry, never from free text as cwd."""

from __future__ import annotations

from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec

_EMPTY = lambda params: {"ok": False, "error": "handler not bound"}  # noqa: E731


def workspace_open_spec() -> ToolSpec:
    return ToolSpec(
        name="workspace.open_in_file_manager",
        title="Projektordner öffnen",
        description="Öffnet den registrierten Workspace im Dateimanager.",
        risk=RiskLevel.EXTERNAL,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["workspace_id"],
            "properties": {"workspace_id": {"type": "string"}},
        },
        handler=_EMPTY,
        effect="Startet xdg-open auf dem canonical Path. Kein Terminalbefehl aus Modelltext.",
        auto_allow=False,
        requires_integration=False,
        allowed_profiles=("observe", "develop"),
    )


def git_diff_spec() -> ToolSpec:
    return ToolSpec(
        name="git.diff",
        title="Git-Diff",
        description="Liest git diff HEAD im registrierten Workspace.",
        risk=RiskLevel.READ,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["workspace_id"],
            "properties": {"workspace_id": {"type": "string"}},
        },
        handler=_EMPTY,
        effect="Nur lesen, keine Staging-/Commit-Aktion.",
        auto_allow=True,
        requires_integration=False,
        allowed_profiles=("observe", "develop"),
    )


def terminal_open_spec() -> ToolSpec:
    return ToolSpec(
        name="terminal.open_workspace",
        title="Terminal im Workspace",
        description="Öffnet ein Systemterminal mit cwd = registrierter Workspace.",
        risk=RiskLevel.EXTERNAL,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["workspace_id"],
            "properties": {"workspace_id": {"type": "string"}},
        },
        handler=_EMPTY,
        effect="Feste Launcher-Vorlage (kgx/ptyxis/gnome-terminal/xdg-terminal-exec). Kein -c, kein Nutzerkommando.",
        auto_allow=False,
        requires_integration=False,
        allowed_profiles=("observe", "develop"),
    )


def workspace_open_file_spec() -> ToolSpec:
    return ToolSpec(
        name="workspace.open_file",
        title="Datei öffnen",
        description="Öffnet eine Datei innerhalb des registrierten Workspace.",
        risk=RiskLevel.EXTERNAL,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["workspace_id", "path"],
            "properties": {
                "workspace_id": {"type": "string"},
                "path": {"type": "string"},
            },
        },
        handler=_EMPTY,
        effect="xdg-open auf eine Datei, die nach Auflösen im Workspace bleiben muss.",
        auto_allow=False,
        requires_integration=False,
        allowed_profiles=("observe", "develop"),
    )


def workspace_open_editor_spec() -> ToolSpec:
    return ToolSpec(
        name="workspace.open_in_editor",
        title="Im Editor öffnen",
        description="Öffnet den registrierten Workspace in einem Allowlist-Editor.",
        risk=RiskLevel.EXTERNAL,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["workspace_id"],
            "properties": {"workspace_id": {"type": "string"}},
        },
        handler=_EMPTY,
        effect="Feste Vorlage (code/codium/gnome-text-editor/gedit), nur der Workspace-Pfad. Kein -c.",
        auto_allow=False,
        requires_integration=False,
        allowed_profiles=("observe", "develop"),
    )


def browser_open_spec() -> ToolSpec:
    return ToolSpec(
        name="browser.open_url",
        title="URL öffnen",
        description="Öffnet eine explizit freigegebene http(s)-URL im Standardbrowser.",
        risk=RiskLevel.EXTERNAL,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["url"],
            "properties": {"url": {"type": "string"}},
        },
        handler=_EMPTY,
        effect="Nur http/https, keine file:/javascript:/data:-URLs, keine Zugangsdaten in der URL.",
        auto_allow=False,
        requires_integration=False,
        allowed_profiles=("observe", "develop"),
    )


def clipboard_copy_spec() -> ToolSpec:
    return ToolSpec(
        name="desktop.copy_text",
        title="Text kopieren",
        description="Kopiert den sichtbaren, bestätigten Text in die Desktop-Zwischenablage.",
        risk=RiskLevel.WRITE,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["text"],
            "properties": {
                "text": {"type": "string", "minLength": 1, "maxLength": 8192},
            },
        },
        handler=_EMPTY,
        effect="Ersetzt den aktuellen Inhalt der Zwischenablage. Keine Tastatur- oder Fenstersimulation.",
        auto_allow=False,
        requires_integration=False,
        allowed_profiles=("observe", "develop"),
        sensitive_parameters=("text",),
    )


def desktop_notification_spec() -> ToolSpec:
    return ToolSpec(
        name="desktop.show_notification",
        title="Benachrichtigung anzeigen",
        description="Zeigt genau eine lokale KIKI-Desktop-Benachrichtigung.",
        risk=RiskLevel.EXTERNAL,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["title", "body"],
            "properties": {
                "title": {"type": "string", "minLength": 1, "maxLength": 80},
                "body": {"type": "string", "minLength": 1, "maxLength": 400},
            },
        },
        handler=_EMPTY,
        effect="Sendet eine lokale Benachrichtigung über Gio. Kein Netzwerk, kein freier Systembefehl.",
        auto_allow=False,
        requires_integration=False,
        allowed_profiles=("observe", "develop"),
        sensitive_parameters=("title", "body"),
    )
