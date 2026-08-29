"""Register and forget Git workspaces. Never deletes repositories."""

from __future__ import annotations

import logging
from pathlib import Path

from kiki.agents.session_service import SessionService
from kiki.runtime.async_bridge import AsyncBridge
from kiki.ui.gi_bootstrap import Adw, Gio, Gtk
from kiki.workspaces.models import Workspace, WorkspaceError

log = logging.getLogger(__name__)


class WorkspaceManagerWindow(Adw.ApplicationWindow):
    def __init__(
        self,
        *,
        application: Adw.Application,
        service: SessionService,
        bridge: AsyncBridge,
        on_change=None,
    ) -> None:
        super().__init__(application=application, title="KIKI Workspaces")
        self.set_default_size(640, 480)
        self.set_hide_on_close(True)
        self._service = service
        self._bridge = bridge
        self._on_change = on_change
        self._toast = Adw.ToastOverlay()
        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        add = Gtk.Button(icon_name="list-add-symbolic", tooltip_text="Git-Ordner hinzufügen")
        add.connect("clicked", lambda *_: self._pick_folder())
        header.pack_start(add)
        toolbar.add_top_bar(header)
        self._list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE, vexpand=True)
        self._list.add_css_class("boxed-list")
        scroll = Gtk.ScrolledWindow(vexpand=True)
        scroll.set_child(self._list)
        hint = Gtk.Label(
            wrap=True,
            xalign=0,
            label="Nur Git-Repository-Wurzeln unter den erlaubten Roots. "
            "Entfernen löscht nicht das Repo auf der Platte.",
        )
        hint.add_css_class("dim-label")
        hint.set_margin_start(12)
        hint.set_margin_end(12)
        hint.set_margin_bottom(12)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_top(8)
        box.append(scroll)
        box.append(hint)
        toolbar.set_content(box)
        self._toast.set_child(toolbar)
        self.set_content(self._toast)
        self.reload()

    def reload(self) -> None:
        while (child := self._list.get_first_child()) is not None:
            self._list.remove(child)
        for workspace in self._service.list_workspaces():
            self._list.append(self._row(workspace))

    def _row(self, workspace: Workspace) -> Gtk.Widget:
        row = Adw.ActionRow(title=workspace.display_name, subtitle=workspace.canonical_path)
        branch = Gtk.Label(label=workspace.active_branch or "(detached)")
        branch.add_css_class("dim-label")
        row.add_suffix(branch)
        remove = Gtk.Button(icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER)
        remove.add_css_class("flat")
        remove.connect("clicked", lambda *_ , wid=workspace.id, name=workspace.display_name: self._confirm_remove(wid, name))
        row.add_suffix(remove)
        row.set_activatable(False)
        return row

    def _pick_folder(self) -> None:
        dialog = Gtk.FileDialog(title="Git-Workspace wählen")
        dialog.select_folder(self, None, self._on_folder)

    def _on_folder(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        try:
            gfile = dialog.select_folder_finish(result)
        except Exception:
            return
        if gfile is None:
            return
        path = gfile.get_path()
        if not path:
            return
        try:
            record = self._service.register_workspace(path, display_name=Path(path).name)
        except WorkspaceError as exc:
            self._toast.add_toast(Adw.Toast(title=str(exc)))
            return
        self._toast.add_toast(Adw.Toast(title=f"Registriert: {record.display_name}"))
        self.reload()
        if self._on_change:
            self._on_change()

    def _confirm_remove(self, workspace_id: str, name: str) -> None:
        dialog = Adw.AlertDialog(
            heading="Workspace entfernen?",
            body=f"„{name}“ verschwindet nur aus der Allowlist. Das Git-Repository bleibt unangetastet.",
        )
        dialog.add_response("cancel", "Abbrechen")
        dialog.add_response("remove", "Entfernen")
        dialog.set_response_appearance("remove", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_default_response("cancel")

        def _done(_d: Adw.AlertDialog, response: str) -> None:
            if response != "remove":
                return
            try:
                self._service.remove_workspace(workspace_id)
            except WorkspaceError as exc:
                self._toast.add_toast(Adw.Toast(title=str(exc)))
                return
            self.reload()
            if self._on_change:
                self._on_change()

        dialog.connect("response", _done)
        dialog.present(self)
