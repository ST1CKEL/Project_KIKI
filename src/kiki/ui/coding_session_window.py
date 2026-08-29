"""Plan-first coding session UI. Approvals stay on screen as text."""

from __future__ import annotations

import logging

from kiki.agents.models import AgentError, AgentEventType, AgentSession
from kiki.agents.session_service import SessionService
from kiki.config.settings import Settings
from kiki.runners.local import TEST_PROFILES
from kiki.runtime.async_bridge import AsyncBridge
from kiki.tools.agent_tools import agent_start_spec
from kiki.tools.registry import ActionPreview
from kiki.tools.test_tools import tests_run_profile_spec as run_profile_tool
from kiki.tools.workspace_tools import (
    browser_open_spec,
    terminal_open_spec,
    workspace_open_editor_spec,
    workspace_open_file_spec,
    workspace_open_spec,
)
from kiki.ui.gi_bootstrap import Adw, Gdk, Gio, GLib, Gtk, Pango
from kiki.ui.widgets.agent_output_view import AgentOutputView, MonoText
from kiki.ui.widgets.approval_card import ApprovalCard
from kiki.ui.widgets.diff_view import DiffView
from kiki.ui.widgets.session_status import SessionStatus
from kiki.workspaces.models import Workspace

log = logging.getLogger(__name__)

_PROFILES = ("observe", "develop")


class CodingSessionWindow(Adw.ApplicationWindow):
    def __init__(
        self,
        *,
        application: Adw.Application,
        service: SessionService,
        bridge: AsyncBridge,
        settings: Settings,
    ) -> None:
        super().__init__(application=application, title="KIKI Coding")
        self.set_default_size(980, 720)
        self.set_hide_on_close(True)
        self._service = service
        self._bridge = bridge
        self._settings = settings
        self._workspace: Workspace | None = None
        self._session: AgentSession | None = None
        self._busy = False
        self._approval_pending = False
        self._poll_id = 0
        self._pending_plan_id: str | None = None
        self._planned_task = ""
        self._running_session_id: str | None = None
        self._running_workspace_id: str | None = None
        self._running_task = ""
        self._diff_tick = 0

        self._toast = Adw.ToastOverlay()
        split = Adw.NavigationSplitView(min_sidebar_width=240, sidebar_width_fraction=0.26)
        split.set_sidebar(Adw.NavigationPage(title="Workspaces", child=self._build_sidebar()))
        split.set_content(Adw.NavigationPage(title="Session", child=self._build_main()))
        self._toast.set_child(split)
        self.set_content(self._toast)
        self.connect("close-request", self._on_close)
        self.reload_workspaces()

    def update_settings(self, settings: Settings) -> None:
        self._settings = settings

    def reload_workspaces(self) -> None:
        selected = self._workspace.id if self._workspace else None
        while (child := self._list.get_first_child()) is not None:
            self._list.remove(child)
        rows = self._service.list_workspaces()
        for workspace in rows:
            row = Gtk.ListBoxRow()
            row.set_name(workspace.id)
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            box.set_margin_start(8)
            box.set_margin_end(8)
            box.set_margin_top(6)
            box.set_margin_bottom(6)
            title = Gtk.Label(label=workspace.display_name, xalign=0, ellipsize=Pango.EllipsizeMode.END)
            sub = Gtk.Label(
                label=f"{workspace.active_branch or '?'} · {workspace.canonical_path}",
                xalign=0,
                ellipsize=Pango.EllipsizeMode.MIDDLE,
            )
            sub.add_css_class("dim-label")
            sub.add_css_class("caption")
            box.append(title)
            box.append(sub)
            row.set_child(box)
            self._list.append(row)
            if workspace.id == selected:
                self._list.select_row(row)
        if selected is None and rows:
            first = self._list.get_row_at_index(0)
            if first:
                self._list.select_row(first)

    def _build_sidebar(self) -> Gtk.Widget:
        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        self._manage_btn = Gtk.Button(icon_name="folder-symbolic", tooltip_text="Workspaces verwalten")
        self._manage_btn.connect("clicked", lambda *_: self.get_application().activate_action("workspaces", None))
        header.pack_start(self._manage_btn)
        toolbar.add_top_bar(header)
        self._list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE, vexpand=True)
        self._list.add_css_class("navigation-sidebar")
        self._list.connect("row-selected", self._on_workspace_row)
        scroll = Gtk.ScrolledWindow(vexpand=True, hscrollbar_policy=Gtk.PolicyType.NEVER)
        scroll.set_child(self._list)
        toolbar.set_content(scroll)
        return toolbar

    def _build_main(self) -> Gtk.Widget:
        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        toolbar.add_top_bar(header)

        self._banner = Adw.Banner(revealed=False)
        self._approval = ApprovalCard()

        agents = Gtk.StringList.new(["OpenCode"])
        self._agent_row = Adw.ComboRow(title="Agent", model=agents)
        profiles = Gtk.StringList.new(list(_PROFILES))
        self._profile_row = Adw.ComboRow(title="Profil", model=profiles)
        self._profile_row.set_subtitle("observe = Plan. develop = Umsetzung nach Freigabe.")
        ping = Gtk.Button(label="OpenCode prüfen")
        ping.connect("clicked", lambda *_: self._ping())
        self._avail = Gtk.Label(xalign=0, wrap=True)
        self._avail.add_css_class("dim-label")

        self._task = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self._task.set_size_request(-1, 90)
        self._task.get_buffer().connect("changed", lambda *_: self._on_task_changed())
        task_frame = Gtk.Frame()
        task_frame.set_child(self._task)

        self._plan_btn = Gtk.Button(label="Plan erstellen")
        self._plan_btn.add_css_class("suggested-action")
        self._plan_btn.connect("clicked", lambda *_: self._start_plan())
        self._start_btn = Gtk.Button(label="Agent starten")
        self._start_btn.connect("clicked", lambda *_: self._start_impl())
        self._stop_btn = Gtk.Button(label="Stoppen")
        self._stop_btn.connect("clicked", lambda *_: self._stop())
        self._stop_btn.set_sensitive(False)
        self._folder_btn = Gtk.Button(label="Ordner öffnen")
        self._folder_btn.connect("clicked", lambda *_: self._open_folder())
        self._diff_btn = Gtk.Button(label="Diff laden")
        self._diff_btn.connect("clicked", lambda *_: self._load_diff())
        self._copy_btn = Gtk.Button(label="Diff kopieren")
        self._copy_btn.connect("clicked", lambda *_: self._copy_diff())
        self._term_btn = Gtk.Button(label="Terminal")
        self._term_btn.connect("clicked", lambda *_: self._open_terminal())
        self._editor_btn = Gtk.Button(label="Editor")
        self._editor_btn.connect("clicked", lambda *_: self._open_editor())
        self._brief_btn = Gtk.Button(label="Briefing")
        self._brief_btn.connect("clicked", lambda *_: self._apply_briefing())
        self._summary_btn = Gtk.Button(label="In den Chat")
        self._summary_btn.connect("clicked", lambda *_: self._send_summary_to_chat())
        self._file_btn = Gtk.Button(label="Datei öffnen")
        self._file_btn.connect("clicked", lambda *_: self._pick_file())
        self._url = Gtk.Entry(placeholder_text="https://…")
        self._url.set_hexpand(True)
        self._url_btn = Gtk.Button(label="URL öffnen")
        self._url_btn.connect("clicked", lambda *_: self._open_url())

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        for widget in (
            self._plan_btn,
            self._start_btn,
            self._stop_btn,
            self._brief_btn,
            self._summary_btn,
            self._folder_btn,
            self._diff_btn,
            self._copy_btn,
        ):
            actions.append(widget)
        desktop = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        desktop.append(self._term_btn)
        desktop.append(self._editor_btn)
        desktop.append(self._file_btn)
        desktop.append(self._url)
        desktop.append(self._url_btn)

        self._stack = Adw.ViewStack()
        self._plan_view = MonoText()
        self._output = AgentOutputView()
        self._diff = DiffView()
        self._tests_page = self._build_tests()
        self._audit_view = MonoText()
        self._stack.add_titled(self._plan_view, "plan", "Plan")
        self._stack.add_titled(self._output, "output", "Agent")
        self._stack.add_titled(self._diff, "diff", "Diff")
        self._stack.add_titled(self._tests_page, "tests", "Tests")
        self._stack.add_titled(self._audit_view, "audit", "Audit")
        switcher = Adw.ViewSwitcher(stack=self._stack, policy=Adw.ViewSwitcherPolicy.WIDE)

        self._status = SessionStatus()

        inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        inner.set_margin_start(12)
        inner.set_margin_end(12)
        inner.set_margin_top(8)
        inner.set_margin_bottom(8)
        inner.append(self._banner)
        inner.append(self._agent_row)
        inner.append(self._profile_row)
        inner.append(ping)
        inner.append(self._avail)
        inner.append(Gtk.Label(label="Aufgabe", xalign=0))
        inner.append(task_frame)
        inner.append(actions)
        inner.append(desktop)
        inner.append(self._approval)
        inner.append(switcher)
        inner.append(self._stack)
        inner.append(self._status)
        toolbar.set_content(inner)
        return toolbar

    def _build_tests(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        names = Gtk.StringList.new(sorted(TEST_PROFILES))
        self._test_row = Adw.ComboRow(title="Testprofil", model=names)
        self._test_row.set_subtitle("Kein freies Kommando — nur der Profilname.")
        self._test_btn = Gtk.Button(label="Tests starten")
        self._test_btn.connect("clicked", lambda *_: self._start_tests())
        self._test_out = MonoText()
        box.append(self._test_row)
        box.append(self._test_btn)
        box.append(self._test_out)
        return box

    def _profile(self) -> str:
        idx = int(self._profile_row.get_selected())
        return _PROFILES[idx] if 0 <= idx < len(_PROFILES) else "observe"

    def _task_text(self) -> str:
        buf = self._task.get_buffer()
        return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).strip()

    def _panic(self) -> bool:
        return bool(self._settings.app.privacy_panic)

    def _on_workspace_row(self, _list: Gtk.ListBox, row: Gtk.ListBoxRow | None) -> None:
        if self._busy or self._approval_pending:
            if row is not None and self._workspace is not None and row.get_name() != self._workspace.id:
                candidate = self._list.get_first_child()
                while candidate is not None:
                    if isinstance(candidate, Gtk.ListBoxRow) and candidate.get_name() == self._workspace.id:
                        self._list.select_row(candidate)
                        break
                    candidate = candidate.get_next_sibling()
            return
        if row is None:
            self._workspace = None
            self._clear_session_views()
            return
        found = self._service.workspaces.get(row.get_name())
        changed = found is None or self._workspace is None or found.id != self._workspace.id
        self._workspace = found
        if found:
            if changed:
                self._pending_plan_id = None
                self._planned_task = ""
                self._hydrate_workspace(found)
            self._refresh_git()

    def _on_task_changed(self) -> None:
        if self._pending_plan_id and self._task_text() != self._planned_task:
            self._pending_plan_id = None
            self._planned_task = ""

    def _clear_session_views(self) -> None:
        self._session = None
        self._running_session_id = None
        self._running_workspace_id = None
        self._running_task = ""
        self._output.clear()
        self._plan_view.set_text("")
        self._diff.set_text("")
        self._test_out.set_text("")
        self._status.set_text("Keine Session.")
        self._stop_btn.set_sensitive(False)

    def _hydrate_workspace(self, workspace: Workspace) -> None:
        self._clear_session_views()
        sessions = self._service.list_sessions(workspace.id)
        if not sessions:
            return
        latest = sessions[0]
        self._session = latest
        self._fill_session(latest)
        valid_plans = [
            session
            for session in sessions
            if session.kind.value == "plan" and session.status.value == "finished"
        ]
        current_task = self._task_text()
        valid_plan = next(
            (session for session in valid_plans if not current_task or session.task_text == current_task),
            None,
        )
        if valid_plan is not None:
            if not current_task:
                self._task.get_buffer().set_text(valid_plan.task_text)
            self._pending_plan_id = valid_plan.id
            self._planned_task = valid_plan.task_text
        if latest.status.value in {"starting", "running", "stopping"}:
            self._running_session_id = latest.id
            self._running_workspace_id = workspace.id
            self._running_task = latest.task_text
            self._set_busy(True)
            self._start_poll()

    def _toast_msg(self, text: str) -> None:
        self._toast.add_toast(Adw.Toast(title=text))

    def _speak(self, text: str) -> None:
        app = self.get_application()
        speak = getattr(app, "speak_status", None)
        if callable(speak):
            speak(text)

    def _notify(self, title: str, body: str) -> None:
        app = self.get_application()
        notify = getattr(app, "notify_status", None)
        if callable(notify):
            notify(title, body)

    def set_task(self, text: str) -> bool:
        if self._busy or self._approval_pending:
            self._toast_msg("Die Aufgabe ist während eines Laufs oder einer Freigabe gesperrt.")
            return False
        self._task.get_buffer().set_text(text or "")
        self._task.grab_focus()
        return True

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._sync_controls()

    def _sync_controls(self) -> None:
        locked = self._busy or self._approval_pending
        for widget in (
            self._plan_btn,
            self._start_btn,
            self._folder_btn,
            self._diff_btn,
            self._copy_btn,
            self._term_btn,
            self._editor_btn,
            self._brief_btn,
            self._summary_btn,
            self._file_btn,
            self._url_btn,
            self._test_btn,
            self._manage_btn,
            self._list,
            self._agent_row,
            self._profile_row,
            self._task,
            self._test_row,
            self._url,
        ):
            widget.set_sensitive(not locked)
        self._stop_btn.set_sensitive(bool(self._busy and self._running_session_id))

    def _present_approval(self, preview: ActionPreview, callback) -> None:
        if self._approval_pending:
            callback(False)
            self._toast_msg("Bitte zuerst die offene Freigabe entscheiden.")
            return
        self._approval_pending = True
        self._sync_controls()

        def _decide(approved: bool) -> None:
            self._approval_pending = False
            self._sync_controls()
            callback(approved)

        self._approval.present(preview, _decide)

    def _refresh_git(self) -> None:
        if self._workspace is None:
            return
        wid = self._workspace.id
        profile = self._profile()

        def _ok(snap) -> None:
            if self._workspace and self._workspace.id != wid:
                return
            branch = snap.branch or "detached"
            msg = f"{self._workspace.display_name if self._workspace else ''} · {branch}"
            if snap.dirty:
                extra = "Uncommitted Änderungen — Agent kann darüber schreiben."
                self._banner.set_title(extra)
                self._banner.set_revealed(True)
                msg += " · dirty"
            else:
                self._banner.set_revealed(False)
            self._status.set_text(msg)

        def _err(exc: BaseException) -> None:
            self._toast_msg(str(exc))

        self._bridge.submit(
            self._service.git_status(wid, profile=profile, panic=self._panic()),
            on_success=_ok,
            on_error=_err,
        )

    def _ping(self) -> None:
        self._avail.set_text("Prüfe OpenCode …")

        def _ok(health) -> None:
            prefix = "OK · " if health.ok else "Fehlt · "
            self._avail.set_text(prefix + health.detail + (f" ({health.version})" if health.version else ""))

        self._bridge.submit(
            self._service.availability(panic=self._panic()),
            on_success=_ok,
            on_error=lambda exc: self._avail.set_text(str(exc)),
        )

    def _require_workspace(self) -> Workspace | None:
        if self._workspace is None:
            self._toast_msg("Bitte einen Workspace wählen oder unter Workspaces registrieren.")
            return None
        return self._workspace

    def _start_plan(self) -> None:
        workspace = self._require_workspace()
        task = self._task_text()
        if workspace is None or not task or self._busy:
            if workspace is not None and not task:
                self._toast_msg("Bitte eine Aufgabe formulieren.")
            return
        self._pending_plan_id = None
        self._planned_task = ""
        self._running_session_id = None
        self._running_workspace_id = workspace.id
        self._running_task = task
        self._set_busy(True)
        self._output.clear()
        self._plan_view.set_text("Plane …")
        self._status.set_text("Plan-Session läuft …")
        self._speak("Ich erstelle einen Plan.")
        self._stack.set_visible_child_name("output")

        def _ok(session: AgentSession) -> None:
            self._session = session
            plan_matches_view = (
                self._workspace is not None
                and self._workspace.id == workspace.id
                and self._task_text() == session.task_text
            )
            if (
                session.kind.value == "plan"
                and session.status.value == "finished"
                and plan_matches_view
            ):
                self._pending_plan_id = session.id
                self._planned_task = session.task_text
            else:
                self._pending_plan_id = None
                self._planned_task = ""
            self._fill_session(session)
            self._running_session_id = None
            self._running_workspace_id = None
            self._running_task = ""
            self._set_busy(False)
            self._stop_poll()
            self._speak("Plan fertig." if session.status.value == "finished" else "Plan beendet.")
            self._notify("KIKI", "Plan-Session beendet.")
            self._refresh_git()
            self._load_audit()

        def _err(exc: BaseException) -> None:
            self._running_session_id = None
            self._running_workspace_id = None
            self._running_task = ""
            self._set_busy(False)
            self._stop_poll()
            self._toast_msg(str(exc))
            self._speak("Plan fehlgeschlagen.")

        self._bridge.submit(
            self._service.run_plan(
                workspace.id,
                task,
                profile=self._profile(),
                model=self._settings.agents.default_model,
                panic=self._panic(),
                agent_timeout=180.0,
            ),
            on_success=_ok,
            on_error=_err,
        )
        self._start_poll()

    def _start_impl(self) -> None:
        workspace = self._require_workspace()
        task = self._task_text()
        if workspace is None or not task or self._busy:
            if workspace is not None and not task:
                self._toast_msg("Bitte eine Aufgabe formulieren.")
            return
        if self._profile() != "develop":
            self._toast_msg("Umsetzung braucht das Profil „develop“.")
            return
        if self._settings.agents.plan_first and not self._pending_plan_id:
            self._toast_msg("Plan-First ist aktiv. Erstelle zuerst einen Plan für genau diese Aufgabe.")
            return
        params = {
            "workspace_id": workspace.id,
            "task": task,
            "model": self._settings.agents.default_model,
            "profile": "develop",
        }
        if self._pending_plan_id:
            params["plan_session_id"] = self._pending_plan_id
        try:
            request = self._service.request_approval("agent.start_implementation", params, profile="develop", panic=self._panic())
        except AgentError as exc:
            self._toast_msg(str(exc))
            return
        spec = agent_start_spec()
        preview = ActionPreview(
            tool=spec.name,
            title=spec.title,
            params=params,
            target=workspace.canonical_path,
            effect=spec.effect,
            risk=spec.risk,
            reason="Getrennte Umsetzungs-Session. Kein sudo, kein Push, nur dieser Workspace.",
        )
        self._stack.set_visible_child_name("output")
        self._present_approval(preview, lambda ok, req=request, p=params: self._on_impl_decision(ok, req.id, p))

    def _on_impl_decision(self, approved: bool, approval_id: str, params: dict) -> None:
        try:
            self._service.decide_approval(approval_id, approved=approved)
        except AgentError as exc:
            self._toast_msg(str(exc))
            return
        if not approved:
            self._toast_msg("Umsetzung abgelehnt.")
            return
        self._running_session_id = None
        self._running_workspace_id = str(params["workspace_id"])
        self._running_task = str(params["task"])
        self._set_busy(True)
        self._output.clear()
        self._status.set_text("Umsetzung läuft …")
        self._speak("Ich starte die Umsetzung.")

        def _ok(session: AgentSession) -> None:
            self._session = session
            self._fill_session(session)
            self._running_session_id = None
            self._running_workspace_id = None
            self._running_task = ""
            self._set_busy(False)
            self._stop_poll()
            self._speak("Umsetzung beendet.")
            self._notify("KIKI", "Umsetzungs-Session beendet.")
            self._refresh_git()
            self._load_diff()
            self._load_audit()

        def _err(exc: BaseException) -> None:
            self._running_session_id = None
            self._running_workspace_id = None
            self._running_task = ""
            self._set_busy(False)
            self._stop_poll()
            self._toast_msg(str(exc))

        self._bridge.submit(
            self._service.start_implementation(
                params["workspace_id"],
                params["task"],
                profile="develop",
                model=params.get("model") or "",
                plan_session_id=params.get("plan_session_id") or None,
                approval_id=approval_id,
                panic=self._panic(),
                agent_timeout=300.0,
            ),
            on_success=_ok,
            on_error=_err,
        )
        self._start_poll()

    def _stop(self) -> None:
        if not self._running_session_id:
            self._toast_msg("Keine laufende Session.")
            return
        sid = self._running_session_id

        def _ok(_result=None) -> None:
            self._toast_msg("Session gestoppt.")
            self._running_session_id = None
            self._running_workspace_id = None
            self._running_task = ""
            self._set_busy(False)
            self._stop_poll()
            self._speak("Gestoppt.")

        self._bridge.submit(
            self._service.stop(sid, profile=self._profile(), panic=self._panic()),
            on_success=_ok,
            on_error=lambda exc: self._toast_msg(str(exc)),
        )

    def _start_tests(self) -> None:
        workspace = self._require_workspace()
        if workspace is None or self._busy:
            return
        idx = int(self._test_row.get_selected())
        names = sorted(TEST_PROFILES)
        if not (0 <= idx < len(names)):
            return
        profile_name = names[idx]
        argv = " ".join(TEST_PROFILES[profile_name])
        params = {"workspace_id": workspace.id, "profile": profile_name}
        try:
            request = self._service.request_approval("tests.run_profile", params, profile="develop", panic=self._panic())
        except AgentError as exc:
            self._toast_msg(str(exc))
            return
        spec = run_profile_tool()
        preview = ActionPreview(
            tool=spec.name,
            title=spec.title,
            params=params,
            target=workspace.canonical_path,
            effect=f"Effektives Kommando: {argv}  · Timeout 300s  · kein Netzwerk-Secret-Env",
            risk=spec.risk,
            reason="Nur das fest verdrahtete Profil, kein Shell-String.",
        )
        parent_id = (
            self._session.id
            if self._session is not None and self._session.workspace_id == workspace.id
            else None
        )
        self._present_approval(
            preview,
            lambda ok, req=request, p=params, cmd=argv, sid=parent_id: self._on_test_decision(
                ok, req.id, p, cmd, sid
            ),
        )

    def _on_test_decision(
        self,
        approved: bool,
        approval_id: str,
        params: dict,
        argv: str,
        session_id: str | None,
    ) -> None:
        try:
            self._service.decide_approval(approval_id, approved=approved)
        except AgentError as exc:
            self._toast_msg(str(exc))
            return
        if not approved:
            return
        self._test_out.set_text(f"Starte {argv} …")
        self._set_busy(True)

        def _ok(result: dict) -> None:
            self._set_busy(False)
            header = (
                f"status={result.get('status')} exit={result.get('exit_code')} "
                f"timeout={result.get('timeout')}\nargv={result.get('argv')}"
            )
            output = str(result.get("output") or "").rstrip()
            truncated = "\n\n[Ausgabe bei 64 KiB gekürzt]" if result.get("output_truncated") else ""
            self._test_out.set_text(header + (f"\n\n{output}" if output else "") + truncated)
            self._load_audit()

        def _err(exc: BaseException) -> None:
            self._set_busy(False)
            self._test_out.set_text(f"Teststart fehlgeschlagen:\n{exc}")
            self._toast_msg(str(exc))
            self._load_audit()

        self._bridge.submit(
            self._service.run_tests(
                params["workspace_id"],
                params["profile"],
                profile="develop",
                approval_id=approval_id,
                panic=self._panic(),
                session_id=session_id,
            ),
            on_success=_ok,
            on_error=_err,
        )

    def _open_folder(self) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return
        params = {"workspace_id": workspace.id}
        try:
            request = self._service.request_approval(
                "workspace.open_in_file_manager", params, profile=self._profile(), panic=self._panic()
            )
        except AgentError as exc:
            self._toast_msg(str(exc))
            return
        spec = workspace_open_spec()
        preview = ActionPreview(
            tool=spec.name,
            title=spec.title,
            params=params,
            target=workspace.canonical_path,
            effect=spec.effect,
            risk=spec.risk,
            reason="Nur der registrierte Pfad, ausgelöst durch diesen Klick.",
        )
        self._present_approval(
            preview,
            lambda ok, req=request: self._on_folder_decision(ok, req.id, workspace.id),
        )

    def _on_folder_decision(self, approved: bool, approval_id: str, workspace_id: str) -> None:
        try:
            self._service.decide_approval(approval_id, approved=approved)
        except AgentError as exc:
            self._toast_msg(str(exc))
            return
        if not approved:
            return
        try:
            path = self._service.open_workspace_folder(
                workspace_id, profile=self._profile(), panic=self._panic(), approval_id=approval_id
            )
        except AgentError as exc:
            self._toast_msg(str(exc))
            return
        self._toast_msg(f"Geöffnet: {path}")

    def _load_diff(self, *, silent: bool = False) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return

        def _ok(diff) -> None:
            self._diff.show(diff.stat, diff.patch, truncated=diff.truncated)
            if not silent:
                self._stack.set_visible_child_name("diff")

        self._bridge.submit(
            self._service.git_diff(workspace.id, profile=self._profile(), panic=self._panic()),
            on_success=_ok,
            on_error=(None if silent else (lambda exc: self._toast_msg(str(exc)))),
        )

    def _open_terminal(self) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return
        params = {"workspace_id": workspace.id}
        try:
            request = self._service.request_approval(
                "terminal.open_workspace", params, profile=self._profile(), panic=self._panic()
            )
        except AgentError as exc:
            self._toast_msg(str(exc))
            return
        spec = terminal_open_spec()
        preview = ActionPreview(
            tool=spec.name,
            title=spec.title,
            params=params,
            target=workspace.canonical_path,
            effect=spec.effect,
            risk=spec.risk,
            reason="cwd ist der registrierte Workspace. Kein Kommando aus dem Chat.",
        )
        self._present_approval(
            preview,
            lambda ok, req=request: self._on_term_decision(ok, req.id, workspace.id),
        )

    def _on_term_decision(self, approved: bool, approval_id: str, workspace_id: str) -> None:
        try:
            self._service.decide_approval(approval_id, approved=approved)
        except AgentError as exc:
            self._toast_msg(str(exc))
            return
        if not approved:
            return
        try:
            argv = self._service.open_terminal(
                workspace_id, profile=self._profile(), panic=self._panic(), approval_id=approval_id
            )
        except AgentError as exc:
            self._toast_msg(str(exc))
            return
        self._toast_msg("Terminal: " + " ".join(argv))

    def _open_editor(self) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return
        params = {"workspace_id": workspace.id}
        try:
            request = self._service.request_approval(
                "workspace.open_in_editor", params, profile=self._profile(), panic=self._panic()
            )
        except AgentError as exc:
            self._toast_msg(str(exc))
            return
        spec = workspace_open_editor_spec()
        preview = ActionPreview(
            tool=spec.name,
            title=spec.title,
            params=params,
            target=workspace.canonical_path,
            effect=spec.effect,
            risk=spec.risk,
            reason="Nur Allowlist-Editor, nur dieser Workspace-Pfad.",
        )
        self._present_approval(
            preview,
            lambda ok, req=request: self._on_editor_decision(ok, req.id, workspace.id),
        )

    def _on_editor_decision(self, approved: bool, approval_id: str, workspace_id: str) -> None:
        try:
            self._service.decide_approval(approval_id, approved=approved)
        except AgentError as extra:
            self._toast_msg(str(extra))
            return
        if not approved:
            return
        try:
            argv = self._service.open_editor(
                workspace_id, profile=self._profile(), panic=self._panic(), approval_id=approval_id
            )
        except AgentError as extra:
            self._toast_msg(str(extra))
            return
        self._toast_msg("Editor: " + " ".join(argv))

    def _apply_briefing(self) -> None:
        from kiki.agents.handoff import format_coding_briefing

        try:
            text = format_coding_briefing(self._task_text())
        except ValueError as extra:
            self._toast_msg(str(extra))
            return
        if self.set_task(text):
            self._toast_msg("Briefing in das Aufgabenfeld geschrieben. Noch kein Agent gestartet.")

    def _send_summary_to_chat(self) -> None:
        from kiki.agents.handoff import session_summary_for_chat

        if self._session is None:
            self._toast_msg("Keine Session zum Zusammenfassen.")
            return
        dirty = bool(self._banner.get_revealed())
        summary = session_summary_for_chat(
            self._session,
            plan_excerpt=self._plan_view.text(),
            dirty=dirty,
        )
        app = self.get_application()
        post = getattr(app, "post_coding_summary_to_chat", None)
        if not callable(post):
            self._toast_msg("Chat ist nicht bereit.")
            return
        post(summary)
        self._speak("Zusammenfassung im Chat.")

    def _pick_file(self) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return
        dialog = Gtk.FileDialog(title="Datei im Workspace öffnen")
        folder = Gio.File.new_for_path(workspace.canonical_path)
        dialog.set_initial_folder(folder)
        dialog.open(self, None, lambda d, res, ws=workspace: self._on_file_chosen(d, res, ws))

    def _on_file_chosen(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult, workspace: Workspace) -> None:
        try:
            gfile = dialog.open_finish(result)
        except Exception:
            return
        if gfile is None:
            return
        path = gfile.get_path()
        if not path:
            return
        if (
            self._busy
            or self._approval_pending
            or self._workspace is None
            or self._workspace.id != workspace.id
        ):
            self._toast_msg("Workspace wurde während der Dateiauswahl gewechselt. Bitte erneut wählen.")
            return
        params = {"workspace_id": workspace.id, "path": path}
        try:
            request = self._service.request_approval(
                "workspace.open_file", params, profile=self._profile(), panic=self._panic()
            )
        except AgentError as extra:
            self._toast_msg(str(extra))
            return
        spec = workspace_open_file_spec()
        preview = ActionPreview(
            tool=spec.name,
            title=spec.title,
            params=params,
            target=path,
            effect=spec.effect,
            risk=spec.risk,
            reason="Pfad muss nach Symlink-Auflösung im Workspace bleiben.",
        )
        self._present_approval(
            preview,
            lambda ok, req=request, p=params: self._on_file_decision(ok, req.id, p),
        )

    def _on_file_decision(self, approved: bool, approval_id: str, params: dict) -> None:
        try:
            self._service.decide_approval(approval_id, approved=approved)
        except AgentError as extra:
            self._toast_msg(str(extra))
            return
        if not approved:
            return
        try:
            opened = self._service.open_workspace_file(
                params["workspace_id"],
                params["path"],
                profile=self._profile(),
                panic=self._panic(),
                approval_id=approval_id,
            )
        except AgentError as extra:
            self._toast_msg(str(extra))
            return
        self._toast_msg(f"Datei: {opened}")

    def _open_url(self) -> None:
        from kiki.tools.desktop_tools import validate_http_url
        from kiki.workspaces.models import WorkspaceError

        raw = self._url.get_text().strip()
        try:
            url = validate_http_url(raw)
        except WorkspaceError as extra:
            self._toast_msg(str(extra))
            return
        params = {"url": url}
        try:
            request = self._service.request_approval(
                "browser.open_url", params, profile=self._profile(), panic=self._panic()
            )
        except AgentError as extra:
            self._toast_msg(str(extra))
            return
        spec = browser_open_spec()
        preview = ActionPreview(
            tool=spec.name,
            title=spec.title,
            params=params,
            target=url,
            effect=spec.effect,
            risk=spec.risk,
            reason="Nur diese eine URL. Kein JavaScript, kein file://.",
        )
        self._present_approval(
            preview,
            lambda ok, req=request, p=params: self._on_url_decision(ok, req.id, p),
        )

    def _on_url_decision(self, approved: bool, approval_id: str, params: dict) -> None:
        try:
            self._service.decide_approval(approval_id, approved=approved)
        except AgentError as extra:
            self._toast_msg(str(extra))
            return
        if not approved:
            return
        try:
            opened = self._service.open_browser_url(
                params["url"], profile=self._profile(), panic=self._panic(), approval_id=approval_id
            )
        except AgentError as extra:
            self._toast_msg(str(extra))
            return
        self._toast_msg(f"Browser: {opened}")

    def _copy_diff(self) -> None:
        text = self._diff.text()
        display = Gdk.Display.get_default()
        if display is None or not text.strip():
            self._toast_msg("Kein Diff zum Kopieren.")
            return
        display.get_clipboard().set(text)
        self._toast_msg("Diff in der Zwischenablage.")

    def _fill_session(self, session: AgentSession) -> None:
        events = self._service.list_events(session.id)
        self._output.set_events(events)
        plans = [event.text for event in events if event.type is AgentEventType.PLAN and event.text]
        if plans:
            self._plan_view.set_text(plans[-1])
        elif session.summary:
            self._plan_view.set_text(session.summary)
        self._status.set_text(
            f"{session.kind.value} · {session.status.value} · "
            f"branch {session.git_branch_before or '?'} · exit {session.exit_code}"
        )

    def _load_audit(self) -> None:
        rows = self._service.recent_audit(60)
        lines = [
            f"{row.ts[11:19]} {row.policy_decision} {row.requested_action} {row.result_status or ''} {row.result_summary or ''}"
            for row in reversed(rows)
        ]
        self._audit_view.set_text("\n".join(lines) or "(leer)")

    def _start_poll(self) -> None:
        if self._poll_id:
            return
        self._diff_tick = 0
        self._poll_id = GLib.timeout_add(400, self._on_poll)

    def _stop_poll(self) -> None:
        if self._poll_id:
            GLib.source_remove(self._poll_id)
            self._poll_id = 0

    def _on_poll(self) -> bool:
        workspace_id = self._running_workspace_id
        if not workspace_id:
            return True
        latest = (
            self._service.get_session(self._running_session_id)
            if self._running_session_id
            else None
        )
        if latest is None:
            active = {"starting", "running", "stopping"}
            candidates = [
                session
                for session in self._service.list_sessions(workspace_id)
                if session.status.value in active and session.task_text == self._running_task
            ]
            if not candidates:
                return True
            latest = candidates[0]
            self._running_session_id = latest.id
            self._stop_btn.set_sensitive(True)
        self._session = latest
        self._fill_session(latest)
        if self._busy:
            self._diff_tick += 1
            if self._diff_tick % 5 == 0:
                self._load_diff(silent=True)
        return True

    def present_coding(self) -> None:
        self.present()
        self.reload_workspaces()
        self._load_audit()
        self._ping()
        if self._busy:
            self._start_poll()

    def _on_close(self, *_args: object) -> bool:
        self._stop_poll()
        return False
