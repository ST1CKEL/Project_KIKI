"""Openers KIKI may run herself.

These are the same seven declared desktop actions the control window offers, but
bound to real handlers and reclassified as `RiskLevel.LAUNCH`: they open
something visible on the user's own desktop and change no data.

What stays unchanged, and is the reason this is defensible without an approval
card at the `trusted` level:

* Every path comes from `WorkspaceRegistry.require()`, which revalidates the
  stored path against the allowed roots, symlinks and the Git root **on every
  call**. A path from model text is never used as a target.
* Files must resolve inside that workspace (`resolve_inside_workspace`).
* Terminal and editor use fixed argv templates from an allowlist. There is no
  shell string and no `-c`, so nothing the model writes becomes a command.
* URLs are http/https only, with no credentials — `validate_http_url`.
* Clipboard and notification are **not** here. Replacing the clipboard changes
  data the user owns, so it keeps its card.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiki.tools.desktop_tools import (
    open_editor_at,
    open_file_with_default_app,
    open_http_url,
    open_path_in_file_manager,
    open_terminal_at,
)
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec
from kiki.tools.workspace_tools import (
    browser_open_spec,
    terminal_open_spec,
    workspace_open_editor_spec,
    workspace_open_file_spec,
    workspace_open_spec,
)
from kiki.workspaces.models import WorkspaceError
from kiki.workspaces.validator import resolve_inside_workspace

log = logging.getLogger(__name__)


def _launchable(spec: ToolSpec, handler: Callable[[dict[str, Any]], dict[str, Any]]) -> ToolSpec:
    """Bind a handler and mark the spec as a model-callable launch action.

    The declarations in `workspace_tools` stay untouched: the control window
    keeps using them for its previews, and this derives executable copies.
    """
    return dataclasses.replace(
        spec,
        risk=RiskLevel.LAUNCH,
        handler=handler,
        auto_allow=True,
        model_callable=True,
    )


class DesktopLaunchSkill:
    id = "desktop_launch"
    name = "Öffnen"
    description = "Ordner, Dateien, Terminal, Editor und Webseiten öffnen."

    def __init__(self, workspaces: Any) -> None:
        self._workspaces = workspaces

    def _workspace_path(self, params: dict[str, Any]) -> Path:
        # require() revalidates roots, symlinks and the Git root every time, so
        # a path redirected after registration is rejected here, not trusted.
        record = self._workspaces.require(str(params["workspace_id"]))
        return Path(record.canonical_path)

    def tools(self) -> list[ToolSpec]:
        return [
            self._list_spec(),
            _launchable(workspace_open_spec(), self._open_folder),
            _launchable(workspace_open_file_spec(), self._open_file),
            _launchable(terminal_open_spec(), self._open_terminal),
            _launchable(workspace_open_editor_spec(), self._open_editor),
            _launchable(browser_open_spec(), self._open_url),
        ]

    def _list_spec(self) -> ToolSpec:
        return ToolSpec(
            name="workspace.list",
            title="Workspaces auflisten",
            description=(
                "Nennt die registrierten Projektordner mit ihrer ID. Die IDs der "
                "Öffnen-Werkzeuge kommen von hier — niemals raten."
            ),
            risk=RiskLevel.READ,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=self._list_workspaces,
            effect="Liest die Workspace-Allowlist. Keine Änderung.",
            target="lokale Registry",
            auto_allow=True,
            requires_integration=False,
            model_callable=True,
        )

    def _list_workspaces(self, _params: dict[str, Any]) -> dict[str, Any]:
        try:
            found = self._workspaces.list()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "count": len(found),
            "workspaces": [
                {"id": w.id, "name": w.display_name, "path": w.canonical_path} for w in found
            ],
        }

    def _open_folder(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            path = self._workspace_path(params)
            open_path_in_file_manager(path)
        except (WorkspaceError, OSError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "opened": str(path)}

    def _open_file(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            root = self._workspace_path(params)
            target = resolve_inside_workspace(str(params["path"]), root)
            open_file_with_default_app(target)
        except (WorkspaceError, OSError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "opened": str(target)}

    def _open_terminal(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            path = self._workspace_path(params)
            argv = open_terminal_at(path)
        except (WorkspaceError, OSError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "cwd": str(path), "argv": argv}

    def _open_editor(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            path = self._workspace_path(params)
            argv = open_editor_at(path)
        except (WorkspaceError, OSError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "cwd": str(path), "argv": argv}

    def _open_url(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            opened = open_http_url(str(params["url"]))
        except (WorkspaceError, OSError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "opened": opened}
