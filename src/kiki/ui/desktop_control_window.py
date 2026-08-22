"""Explicit, approval-bound desktop control UI.

The window is a user-operated control surface.  It does not accept model tool
calls and exposes no generic command, key, pointer, or window automation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gi.repository import Adw, Gio, Gtk

from kiki.agents.models import AgentError
from kiki.agents.session_service import SessionService
from kiki.config.settings import Settings
from kiki.ui.confirmation_dialog import present_confirmation
from kiki.ui.desktop_control_model import (
    PreparedDesktopAction,
    prepare_browser_url,
    prepare_clipboard_text,
    prepare_editor,
    prepare_notification,
    prepare_terminal,
    prepare_workspace_file,
    prepare_workspace_folder,
)
from kiki.workspaces.models import Workspace, WorkspaceError

log = logging.getLogger(__name__)

_PROFILE = "observe"


class DesktopControlWindow(Adw.ApplicationWindow):
    """A bounded control centre in which every OS effect needs confirmation."""

    def __init__(
        self,
        *,
        application: Adw.Application,
        service: SessionService,
        settings: Settings,
    ) -> None:
        super().__init__(application=application, title="KIKI PC-Steuerung")
        self.set_default_size(780, 720)
        self.set_hide_on_close(True)
        self._service = service
        self._settings = settings
        self._workspaces: list[Workspace] = []
        self._workspace: Workspace | None = None
        self._approval_pending = False
        self._reloading = False
        self._workspace_controls: list[Gtk.Widget] = []
        self._action_controls: list[Gtk.Widget] = []

        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        header.set_title_widget(
            Adw.WindowTitle(
                title="PC-Steuerung",
                subtitle="Observe-Profil · jede Aktion wird vorab gezeigt",
            )
        )
        self._manage_button = Gtk.Button(
            label="Workspaces",
            tooltip_text="Registrierte Projektordner verwalten",
        )
        self._manage_button.connect(
            "clicked",
            lambda *_: self.get_application().activate_action("workspaces", None),
        )
        header.pack_start(self._manage_button)
        toolbar.add_top_bar(header)

        self._panic_banner = Adw.Banner(
            title="Privacy-/Panic-Modus ist aktiv. PC-Aktionen sind gesperrt."
        )
        self._panic_banner.set_revealed(False)
        page = self._build_page()
        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        body.append(self._panic_banner)
        body.append(page)
        toolbar.set_content(body)

        self._toast = Adw.ToastOverlay(child=toolbar)
        self.set_content(self._toast)
        self.reload_workspaces()

    def present_control(self) -> None:
        self.reload_workspaces()
        self._sync_controls()
        self.present()

    def update_settings(self, settings: Settings) -> None:
        self._settings = settings
        self._sync_controls()

    def _build_page(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage()
        page.set_vexpand(True)

        workspace_group = Adw.PreferencesGroup(
            title="Workspace",
            description=(
                "KIKI verwendet ausschließlich registrierte Git-Workspaces. "
                "Freie Arbeitsverzeichnisse sind nicht möglich."
            ),
        )
        self._workspace_row = Adw.ComboRow(
            title="Aktiver Workspace",
            model=Gtk.StringList.new(["Keine registrierten Workspaces"]),
        )
        self._workspace_row.connect("notify::selected", self._on_workspace_selected)
        workspace_group.add(self._workspace_row)

        folder = self._action_row(
            "Projektordner öffnen",
            "Öffnet genau den registrierten Ordner im Standard-Dateimanager.",
            "Prüfen & öffnen",
            self._open_folder,
        )
        terminal = self._action_row(
            "Terminal öffnen",
            "Startet einen erlaubten Terminal-Launcher im Workspace – ohne Kommando.",
            "Prüfen & öffnen",
            self._open_terminal,
        )
        editor = self._action_row(
            "Editor öffnen",
            "Übergibt nur den Workspace-Pfad an einen erlaubten Editor.",
            "Prüfen & öffnen",
            self._open_editor,
        )
        for row, button in (folder, terminal, editor):
            workspace_group.add(row)
            self._workspace_controls.append(button)

        self._file_entry = Adw.EntryRow(title="Datei relativ zum Workspace")
        self._file_entry.set_tooltip_text("Zum Beispiel src/kiki/main.py")
        self._pick_file_button = Gtk.Button(
            icon_name="document-open-symbolic",
            tooltip_text="Datei im Workspace auswählen",
            valign=Gtk.Align.CENTER,
        )
        self._pick_file_button.add_css_class("flat")
        self._pick_file_button.connect("clicked", lambda *_: self._pick_file())
        self._file_entry.add_suffix(self._pick_file_button)
        workspace_group.add(self._file_entry)
        file_action = self._action_row(
            "Datei mit Standard-App öffnen",
            "Kanonische Prüfung blockiert absolute Pfade und Symlink-Ausbrüche.",
            "Prüfen & öffnen",
            self._open_file,
        )
        workspace_group.add(file_action[0])
        self._workspace_controls.extend(
            [self._file_entry, self._pick_file_button, file_action[1]]
        )
        page.add(workspace_group)

        web_group = Adw.PreferencesGroup(
            title="Browser",
            description="Nur http(s), ohne Zugangsdaten in der URL.",
        )
        self._url_entry = Adw.EntryRow(title="https://…")
        self._url_entry.set_input_purpose(Gtk.InputPurpose.URL)
        web_group.add(self._url_entry)
        url_action = self._action_row(
            "URL im Standardbrowser öffnen",
            "file:, javascript:, data: und andere Schemes werden abgewiesen.",
            "Prüfen & öffnen",
            self._open_url,
        )
        web_group.add(url_action[0])
        self._action_controls.extend([self._url_entry, url_action[1]])
        page.add(web_group)

        clipboard_group = Adw.PreferencesGroup(
            title="Zwischenablage",
            description="Der sichtbare Text ersetzt nach Bestätigung den aktuellen Inhalt.",
        )
        clipboard_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        clipboard_card.add_css_class("card")
        clipboard_card.set_margin_top(2)
        clipboard_card.set_margin_bottom(2)
        clipboard_card.set_margin_start(2)
        clipboard_card.set_margin_end(2)
        inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        inner.set_margin_top(12)
        inner.set_margin_bottom(12)
        inner.set_margin_start(12)
        inner.set_margin_end(12)
        heading = Gtk.Label(label="Text kopieren", xalign=0)
        heading.add_css_class("heading")
        hint = Gtk.Label(
            label="Maximal 8.192 Zeichen; keine simulierten Tastatureingaben.",
            xalign=0,
            wrap=True,
        )
        hint.add_css_class("dim-label")
        hint.add_css_class("caption")
        self._clipboard_text = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR,
            top_margin=8,
            bottom_margin=8,
            left_margin=8,
            right_margin=8,
        )
        clipboard_scroll = Gtk.ScrolledWindow(
            min_content_height=96,
            max_content_height=180,
            propagate_natural_height=True,
        )
        clipboard_scroll.add_css_class("view")
        clipboard_scroll.set_child(self._clipboard_text)
        self._clipboard_button = Gtk.Button(label="Prüfen & kopieren", halign=Gtk.Align.END)
        self._clipboard_button.connect("clicked", lambda *_: self._copy_text())
        inner.append(heading)
        inner.append(hint)
        inner.append(clipboard_scroll)
        inner.append(self._clipboard_button)
        clipboard_card.append(inner)
        clipboard_group.add(clipboard_card)
        self._action_controls.extend([self._clipboard_text, self._clipboard_button])
        page.add(clipboard_group)

        notification_group = Adw.PreferencesGroup(
            title="Lokale Benachrichtigung",
            description="Genau eine Gio-Benachrichtigung, ohne Netzwerk oder Systembefehl.",
        )
        self._notification_title = Adw.EntryRow(title="Titel")
        self._notification_body = Adw.EntryRow(title="Nachricht")
        notification_group.add(self._notification_title)
        notification_group.add(self._notification_body)
        notification_action = self._action_row(
            "Benachrichtigung anzeigen",
            "Titel und Nachricht bleiben lokal und werden nicht im Audit protokolliert.",
            "Prüfen & anzeigen",
            self._show_notification,
        )
        notification_group.add(notification_action[0])
        self._action_controls.extend(
            [self._notification_title, self._notification_body, notification_action[1]]
        )
        page.add(notification_group)

        safety_group = Adw.PreferencesGroup(title="Sicherheitsgrenze")
        safety_group.add(
            Adw.ActionRow(
                title="Keine Fernsteuerungs-Automation",
                subtitle=(
                    "Keine Shellstrings, keine Maus-/Tastatursimulation und keine Aktionen "
                    "direkt aus Chat- oder Sprachtext."
                ),
            )
        )
        page.add(safety_group)
        return page

    @staticmethod
    def _action_row(
        title: str,
        subtitle: str,
        button_label: str,
        callback: Callable[[], None],
    ) -> tuple[Adw.ActionRow, Gtk.Button]:
        row = Adw.ActionRow(title=title, subtitle=subtitle)
        button = Gtk.Button(label=button_label, valign=Gtk.Align.CENTER)
        button.connect("clicked", lambda *_: callback())
        row.add_suffix(button)
        row.set_activatable_widget(button)
        return row, button

    def reload_workspaces(self) -> None:
        selected_id = self._workspace.id if self._workspace else None
        self._workspaces = self._service.list_workspaces()
        labels = [workspace.display_name for workspace in self._workspaces]
        if not labels:
            labels = ["Keine registrierten Workspaces"]
        self._reloading = True
        try:
            self._workspace_row.set_model(Gtk.StringList.new(labels))
            selected = next(
                (
                    index
                    for index, workspace in enumerate(self._workspaces)
                    if workspace.id == selected_id
                ),
                0,
            )
            self._workspace_row.set_selected(selected)
        finally:
            self._reloading = False
        self._select_workspace_at(int(self._workspace_row.get_selected()))

    def _on_workspace_selected(self, row: Adw.ComboRow, _param: Any) -> None:
        if self._reloading:
            return
        self._select_workspace_at(int(row.get_selected()))

    def _select_workspace_at(self, index: int) -> None:
        previous_id = self._workspace.id if self._workspace else None
        self._workspace = self._workspaces[index] if 0 <= index < len(self._workspaces) else None
        if self._workspace is None:
            self._workspace_row.set_subtitle("Unter Workspaces zuerst einen Git-Ordner registrieren.")
            self._file_entry.set_text("")
        else:
            self._workspace_row.set_subtitle(self._workspace.canonical_path)
            if self._workspace.id != previous_id:
                self._file_entry.set_text("")
        self._sync_controls()

    def _panic(self) -> bool:
        return bool(self._settings.app.privacy_panic)

    def _sync_controls(self) -> None:
        panic = self._panic()
        locked = self._approval_pending or panic
        self._panic_banner.set_revealed(panic)
        self._workspace_row.set_sensitive(not self._approval_pending)
        self._manage_button.set_sensitive(not self._approval_pending)
        for widget in self._action_controls:
            widget.set_sensitive(not locked)
        workspace_enabled = not locked and self._workspace is not None
        for widget in self._workspace_controls:
            widget.set_sensitive(workspace_enabled)

    def _workspace_or_warn(self) -> Workspace | None:
        if self._workspace is None:
            self._toast_msg("Bitte zuerst einen registrierten Workspace wählen.")
            return None
        return self._workspace

    def _pick_file(self) -> None:
        workspace = self._workspace_or_warn()
        if workspace is None:
            return
        dialog = Gtk.FileDialog(title="Datei im Workspace wählen")
        dialog.set_initial_folder(Gio.File.new_for_path(workspace.canonical_path))
        dialog.open(self, None, self._on_file_chosen)

    def _on_file_chosen(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        try:
            chosen = dialog.open_finish(result)
        except Exception:
            return
        workspace = self._workspace
        if chosen is None or workspace is None:
            return
        path = chosen.get_path()
        if not path:
            self._toast_msg("Nur lokale Dateien im Workspace sind erlaubt.")
            return
        try:
            action = prepare_workspace_file(workspace, path)
        except (OSError, ValueError, WorkspaceError) as exc:
            self._toast_msg(str(exc))
            return
        self._file_entry.set_text(str(action.params["path"]))

    def _open_folder(self) -> None:
        workspace = self._workspace_or_warn()
        if workspace is None:
            return
        self._prepare_and_confirm(
            lambda: prepare_workspace_folder(workspace),
            lambda approval_id: self._service.open_workspace_folder(
                workspace.id,
                profile=_PROFILE,
                panic=self._panic(),
                approval_id=approval_id,
            ),
            lambda result: f"Ordner geöffnet: {result}",
        )

    def _open_terminal(self) -> None:
        workspace = self._workspace_or_warn()
        if workspace is None:
            return
        self._prepare_and_confirm(
            lambda: prepare_terminal(workspace),
            lambda approval_id: self._service.open_terminal(
                workspace.id,
                profile=_PROFILE,
                panic=self._panic(),
                approval_id=approval_id,
            ),
            lambda _result: "Terminal im Workspace geöffnet.",
        )

    def _open_editor(self) -> None:
        workspace = self._workspace_or_warn()
        if workspace is None:
            return
        self._prepare_and_confirm(
            lambda: prepare_editor(workspace),
            lambda approval_id: self._service.open_editor(
                workspace.id,
                profile=_PROFILE,
                panic=self._panic(),
                approval_id=approval_id,
            ),
            lambda _result: "Workspace im Editor geöffnet.",
        )

    def _open_file(self) -> None:
        workspace = self._workspace_or_warn()
        if workspace is None:
            return
        raw_path = self._file_entry.get_text()

        def _execute(approval_id: str, *, action: PreparedDesktopAction):
            return self._service.open_workspace_file(
                workspace.id,
                str(action.params["path"]),
                profile=_PROFILE,
                panic=self._panic(),
                approval_id=approval_id,
            )

        self._prepare_and_confirm_dynamic(
            lambda: prepare_workspace_file(workspace, raw_path),
            _execute,
            lambda result: f"Datei geöffnet: {Path(str(result)).name}",
        )

    def _open_url(self) -> None:
        raw_url = self._url_entry.get_text()

        def _execute(approval_id: str, *, action: PreparedDesktopAction):
            return self._service.open_browser_url(
                str(action.params["url"]),
                profile=_PROFILE,
                panic=self._panic(),
                approval_id=approval_id,
            )

        self._prepare_and_confirm_dynamic(
            lambda: prepare_browser_url(raw_url),
            _execute,
            lambda result: f"URL geöffnet: {result}",
        )

    def _copy_text(self) -> None:
        buffer = self._clipboard_text.get_buffer()
        raw_text = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)

        def _execute(approval_id: str, *, action: PreparedDesktopAction):
            return self._service.copy_desktop_text(
                str(action.params["text"]),
                profile=_PROFILE,
                panic=self._panic(),
                approval_id=approval_id,
            )

        self._prepare_and_confirm_dynamic(
            lambda: prepare_clipboard_text(raw_text),
            _execute,
            lambda result: f"{result} Zeichen in die Zwischenablage kopiert.",
        )

    def _show_notification(self) -> None:
        title = self._notification_title.get_text()
        body = self._notification_body.get_text()

        def _execute(approval_id: str, *, action: PreparedDesktopAction):
            return self._service.show_notification(
                str(action.params["title"]),
                str(action.params["body"]),
                profile=_PROFILE,
                panic=self._panic(),
                approval_id=approval_id,
            )

        self._prepare_and_confirm_dynamic(
            lambda: prepare_notification(title, body),
            _execute,
            lambda _result: "Lokale Benachrichtigung angezeigt.",
        )

    def _prepare_and_confirm(
        self,
        prepare: Callable[[], PreparedDesktopAction],
        execute: Callable[[str], object],
        success: Callable[[object], str],
    ) -> None:
        self._prepare_and_confirm_dynamic(
            prepare,
            lambda approval_id, *, action: execute(approval_id),
            success,
        )

    def _prepare_and_confirm_dynamic(
        self,
        prepare: Callable[[], PreparedDesktopAction],
        execute: Callable[..., object],
        success: Callable[[object], str],
    ) -> None:
        if self._approval_pending:
            self._toast_msg("Bitte zuerst die offene Freigabe entscheiden.")
            return
        if self._panic():
            self._toast_msg("Privacy-/Panic-Modus blockiert PC-Aktionen.")
            return
        try:
            action = prepare()
            request = self._service.request_approval(
                action.preview.tool,
                action.params,
                profile=_PROFILE,
                panic=self._panic(),
            )
        except (AgentError, OSError, ValueError, WorkspaceError) as exc:
            self._toast_msg(str(exc))
            return
        self._approval_pending = True
        self._sync_controls()

        def _decide(approved: bool) -> None:
            try:
                self._service.decide_approval(request.id, approved=approved)
                if not approved:
                    self._toast_msg("Aktion abgebrochen.")
                    return
                result = execute(request.id, action=action)
                self._toast_msg(success(result))
            except (AgentError, OSError, ValueError, WorkspaceError) as exc:
                self._toast_msg(str(exc))
            except Exception as exc:  # keep a desktop launcher failure inside the UI boundary
                log.exception("desktop control action failed")
                self._toast_msg(f"Aktion fehlgeschlagen: {exc}")
            finally:
                self._approval_pending = False
                self._sync_controls()

        try:
            present_confirmation(self, action.preview, _decide)
        except Exception:
            try:
                self._service.decide_approval(request.id, approved=False)
            except AgentError:
                log.exception("could not cancel unpresented desktop approval")
            self._approval_pending = False
            self._sync_controls()
            log.exception("could not present desktop action confirmation")
            self._toast_msg("Freigabedialog konnte nicht geöffnet werden.")

    def _toast_msg(self, text: str) -> None:
        self._toast.add_toast(Adw.Toast(title=text, timeout=4))
