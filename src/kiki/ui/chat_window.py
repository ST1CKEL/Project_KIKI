from __future__ import annotations

import logging
from pathlib import Path

from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango

from kiki.ai.chat_service import ChatService, StreamEvent
from kiki.ai.vision import VisionEncodeError, encode_image_file
from kiki.config.settings import Settings
from kiki.runtime.async_bridge import AsyncBridge, StreamHandle
from kiki.storage.chat_repository import ChatRepository, Conversation
from kiki.ui.widgets.chat_bubble import ChatBubble

log = logging.getLogger(__name__)

MAX_ATTACHMENTS = 2


class ChatWindow(Adw.ApplicationWindow):
    def __init__(
        self,
        *,
        application: Adw.Application,
        chats: ChatRepository,
        service: ChatService,
        bridge: AsyncBridge,
        settings: Settings,
    ) -> None:
        super().__init__(application=application, title="KIKI Chat")
        self.set_default_size(860, 620)
        self.set_hide_on_close(True)
        self._chats = chats
        self._service = service
        self._bridge = bridge
        self._settings = settings
        self._conversation: Conversation | None = None
        self._stream: StreamHandle | None = None
        self._stream_bubble: ChatBubble | None = None
        self._stream_text = ""
        self._pending_images: list[dict[str, str]] = []

        self._toast = Adw.ToastOverlay()
        split = Adw.NavigationSplitView(min_sidebar_width=220, sidebar_width_fraction=0.28)
        split.set_sidebar(Adw.NavigationPage(title="Chats", child=self._build_sidebar()))
        split.set_content(Adw.NavigationPage(title="Unterhaltung", child=self._build_chat()))
        self._toast.set_child(split)
        self.set_content(self._toast)
        self._reload_sidebar()
        self._new_conversation()

    def update_settings(self, settings: Settings) -> None:
        self._settings = settings

    def _build_sidebar(self) -> Gtk.Widget:
        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        new_btn = Gtk.Button(icon_name="document-new-symbolic", tooltip_text="Neuer Chat")
        new_btn.connect("clicked", lambda *_: self._new_conversation())
        header.pack_start(new_btn)
        toolbar.add_top_bar(header)
        self._list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE, vexpand=True)
        self._list.add_css_class("navigation-sidebar")
        self._list.connect("row-selected", self._on_row_selected)
        scroll = Gtk.ScrolledWindow(vexpand=True, hscrollbar_policy=Gtk.PolicyType.NEVER)
        scroll.set_child(self._list)
        toolbar.set_content(scroll)
        return toolbar

    def _build_chat(self) -> Gtk.Widget:
        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        delete = Gtk.Button(icon_name="user-trash-symbolic", tooltip_text="Chat löschen")
        delete.connect("clicked", lambda *_: self._confirm_delete())
        prefs = Gtk.Button(icon_name="emblem-system-symbolic", tooltip_text="Einstellungen")
        prefs.connect("clicked", lambda *_: self.get_application().activate_action("preferences", None))
        control = Gtk.Button(
            label="PC-Steuerung",
            tooltip_text="Sichere Desktop-Aktionen mit Vorschau und Freigabe",
        )
        control.connect(
            "clicked",
            lambda *_: self.get_application().activate_action("desktop-control", None),
        )
        header.pack_start(control)
        header.pack_end(delete)
        header.pack_end(prefs)
        toolbar.add_top_bar(header)

        self._messages = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._messages.set_margin_top(8)
        self._messages.set_margin_bottom(8)
        self._messages.set_margin_start(8)
        self._messages.set_margin_end(8)
        self._scroller = Gtk.ScrolledWindow(vexpand=True, hscrollbar_policy=Gtk.PolicyType.NEVER)
        self._scroller.set_child(self._messages)
        self._scroller.set_propagate_natural_width(False)

        self._input = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self._input.set_size_request(-1, 72)
        key = Gtk.EventControllerKey()
        key.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        key.connect("key-pressed", self._on_key)
        self._input.add_controller(key)

        send = Gtk.Button(label="Senden")
        send.add_css_class("suggested-action")
        send.connect("clicked", lambda *_: self._send())
        status = Gtk.Button(label="Status anhängen")
        status.add_css_class("flat")
        status.connect("clicked", lambda *_: self._send(attach_status=True))
        status.set_tooltip_text("Hängt Uhrzeit, Akku, Netzwerk und Speicher an — nur auf Klick, nie automatisch.")
        attach = Gtk.Button(icon_name="image-x-generic-symbolic", tooltip_text="Bilddatei anhängen")
        attach.add_css_class("flat")
        attach.connect("clicked", lambda *_: self._pick_image())
        screen = Gtk.Button(icon_name="video-display-symbolic", tooltip_text="Bildschirm zeigen (mit Freigabe)")
        screen.add_css_class("flat")
        screen.connect("clicked", lambda *_: self.get_application().activate_action("screenshot", None))
        self._mic_button = Gtk.Button(icon_name="audio-input-microphone-symbolic", tooltip_text="Zuhören (Push-to-talk)")
        self._mic_button.add_css_class("flat")
        self._mic_button.connect("clicked", lambda *_: self.get_application().activate_action("voice-toggle", None))
        coding = Gtk.Button(label="Coding-Session")
        coding.add_css_class("flat")
        coding.set_tooltip_text("Übernimmt den Entwurf oder die letzte Nutzerzeile in die Coding-Session. Startet keinen Agenten.")
        coding.connect("clicked", lambda *_: self._handoff_coding())

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        actions.set_margin_start(8)
        actions.set_margin_end(8)
        actions.set_margin_bottom(8)
        actions.append(status)
        actions.append(attach)
        actions.append(screen)
        actions.append(self._mic_button)
        actions.append(coding)
        spacer = Gtk.Box(hexpand=True)
        actions.append(spacer)
        actions.append(send)

        frame = Gtk.Frame()
        frame.add_css_class("kiki-input")
        frame.set_child(self._input)
        frame.set_margin_start(8)
        frame.set_margin_end(8)

        self._attach_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._attach_bar.set_margin_start(8)
        self._attach_bar.set_margin_end(8)
        self._attach_bar.set_visible(False)

        drop = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop.connect("drop", self._on_drop_files)
        self.add_controller(drop)

        bottom = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        bottom.append(self._attach_bar)
        bottom.append(frame)
        bottom.append(actions)

        inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        inner.append(self._scroller)
        inner.append(bottom)
        toolbar.set_content(inner)
        return toolbar

    def _on_key(self, _controller: Gtk.EventControllerKey, keyval: int, _keycode: int, state: Gdk.ModifierType) -> bool:
        if keyval in {Gdk.KEY_Return, Gdk.KEY_KP_Enter}:
            if state & Gdk.ModifierType.SHIFT_MASK:
                return False
            self._send()
            return True
        return False

    def _reload_sidebar(self, select_id: str | None = None) -> None:
        while (child := self._list.get_first_child()) is not None:
            self._list.remove(child)
        target = select_id or (self._conversation.id if self._conversation else None)
        for conv in self._chats.list_conversations():
            row = Gtk.ListBoxRow()
            row.set_name(conv.id)
            label = Gtk.Label(label=conv.title or "Chat", xalign=0, ellipsize=Pango.EllipsizeMode.END)
            label.set_margin_start(8)
            label.set_margin_end(8)
            label.set_margin_top(8)
            label.set_margin_bottom(8)
            row.set_child(label)
            self._list.append(row)
            if conv.id == target:
                self._list.select_row(row)

    def _on_row_selected(self, _list: Gtk.ListBox, row: Gtk.ListBoxRow | None) -> None:
        if row is None:
            return
        cid = row.get_name()
        if self._conversation and cid == self._conversation.id:
            return
        conv = self._chats.get_conversation(cid)
        if conv:
            self._open_conversation(conv)

    def _new_conversation(self) -> None:
        conv = self._chats.create_conversation()
        self._open_conversation(conv)
        self._reload_sidebar(conv.id)

    def _open_conversation(self, conv: Conversation) -> None:
        self._cancel_stream()
        self._conversation = conv
        self._clear_messages()
        for msg in self._chats.list_messages(conv.id):
            self._messages.append(ChatBubble(msg.role, msg.content))
        self._scroll_to_end()

    def _clear_messages(self) -> None:
        while (child := self._messages.get_first_child()) is not None:
            self._messages.remove(child)

    def _confirm_delete(self) -> None:
        if self._conversation is None:
            return
        dialog = Adw.AlertDialog(
            heading="Chat löschen?",
            body="Der Verlauf wird lokal entfernt. Das lässt sich nicht rückgängig machen.",
        )
        dialog.add_response("cancel", "Abbrechen")
        dialog.add_response("delete", "Löschen")
        dialog.set_response_appearance("delete", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_default_response("cancel")

        def _done(_d: Adw.AlertDialog, response: str) -> None:
            if response == "delete" and self._conversation:
                self._chats.delete_conversation(self._conversation.id)
                remaining = self._chats.list_conversations()
                if remaining:
                    self._open_conversation(remaining[0])
                else:
                    self._new_conversation()
                self._reload_sidebar()

        dialog.connect("response", _done)
        dialog.present(self)

    def _pick_image(self) -> None:
        dialog = Gtk.FileDialog(title="Bild anhängen")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        image_filter = Gtk.FileFilter(name="Bilder")
        for mime in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            image_filter.add_mime_type(mime)
        filters.append(image_filter)
        dialog.set_filters(filters)
        dialog.open(self, None, self._on_image_chosen)

    def _on_image_chosen(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        try:
            gfile = dialog.open_finish(result)
        except Exception:
            return
        if gfile is None:
            return
        path = gfile.get_path()
        if path:
            self._add_image(Path(path))

    def _on_drop_files(self, _target: Gtk.DropTarget, value: Gdk.FileList, _x: float, _y: float) -> bool:
        for gfile in value.get_files():
            path = gfile.get_path()
            if path:
                self._add_image(Path(path))
        return True

    def _add_image(self, path: Path) -> None:
        if len(self._pending_images) >= MAX_ATTACHMENTS:
            self._toast.add_toast(Adw.Toast(title="Höchstens zwei Bilder pro Nachricht."))
            return
        try:
            encoded = encode_image_file(path)
        except VisionEncodeError as exc:
            self._toast.add_toast(Adw.Toast(title=str(exc)))
            return
        self._pending_images.append({"name": path.name, "b64": encoded})
        self._refresh_attach_bar()

    def _refresh_attach_bar(self) -> None:
        while (child := self._attach_bar.get_first_child()) is not None:
            self._attach_bar.remove(child)
        if not self._pending_images:
            self._attach_bar.set_visible(False)
            return
        self._attach_bar.set_visible(True)
        for idx, item in enumerate(self._pending_images):
            chip = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
            lab = Gtk.Label(label=item["name"], ellipsize=Pango.EllipsizeMode.END)
            clear = Gtk.Button(icon_name="window-close-symbolic")
            clear.add_css_class("flat")
            clear.connect("clicked", lambda _b, i=idx: self._remove_image(i))
            chip.append(lab)
            chip.append(clear)
            self._attach_bar.append(chip)

    def _remove_image(self, index: int) -> None:
        if 0 <= index < len(self._pending_images):
            del self._pending_images[index]
            self._refresh_attach_bar()

    def _send(self, *, attach_status: bool = False) -> None:
        if self._conversation is None or self._stream is not None:
            return
        buf = self._input.get_buffer()
        text = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).strip()
        pending = list(self._pending_images)
        if not text and not pending:
            return
        buf.set_text("", -1)
        self._pending_images.clear()
        self._refresh_attach_bar()
        conv_id = self._conversation.id
        display = text or "Was siehst du auf dem Bild?"
        if pending:
            display = f"{display}\n\n_[Bild: {', '.join(p['name'] for p in pending)}]_"
        self._messages.append(ChatBubble("user", display))
        bubble = ChatBubble("assistant", "", streaming=True)
        self._messages.append(bubble)
        self._stream_bubble = bubble
        self._stream_text = ""
        self._scroll_to_end()
        images = tuple(p["b64"] for p in pending)
        names = tuple(p["name"] for p in pending)

        async def _gen():
            snapshot = None
            if attach_status:
                snapshot = await self._service.collect_status()
            async for event in self._service.send(
                conv_id,
                text,
                status_snapshot=snapshot,
                images=images,
                image_names=names,
            ):
                yield event

        self._stream = self._bridge.stream(
            _gen(),
            on_item=self._on_stream_item,
            on_error=self._on_stream_error,
            on_complete=self._on_stream_complete,
        )
        self._reload_sidebar(conv_id)

    def _on_stream_item(self, event: StreamEvent) -> None:
        if event.kind == "delta" and self._stream_bubble is not None:
            self._stream_text += event.text
            self._stream_bubble.append_delta(event.text)
            self._scroll_to_end()
        elif event.kind == "tool_start" and self._stream_bubble is not None:
            self._stream_bubble.note_activity(f"⚙ {event.text or event.tool} …")
            self._scroll_to_end()
        elif event.kind == "tool_end" and self._stream_bubble is not None:
            label = event.tool or "Werkzeug"
            if event.ok:
                self._stream_bubble.replace_last_activity(f"✓ {label}")
            else:
                detail = (event.text or "abgelehnt").strip()
                self._stream_bubble.replace_last_activity(f"✗ {label}: {detail}")
            self._scroll_to_end()
        elif event.kind == "error":
            self._on_stream_error(RuntimeError(event.text))
        elif event.kind == "done" and self._stream_bubble is not None:
            full = event.text or self._stream_text
            self._stream_bubble.finish_markdown(full or "_Leere Antwort._")
            self._stream_bubble = None

    def _on_stream_error(self, exc: BaseException) -> None:
        if self._stream_bubble is not None:
            self._messages.remove(self._stream_bubble)
            self._stream_bubble = None
        self._messages.append(ChatBubble("error", str(exc)))
        self._toast.add_toast(Adw.Toast(title="Modell nicht erreichbar"))
        self._scroll_to_end()

    def _on_stream_complete(self) -> None:
        self._stream = None
        self._reload_sidebar(self._conversation.id if self._conversation else None)

    def _cancel_stream(self) -> None:
        if self._stream is not None:
            self._stream.cancel()
            self._stream = None
        self._stream_bubble = None

    def _scroll_to_end(self) -> None:
        def _go() -> bool:
            adj = self._scroller.get_vadjustment()
            adj.set_value(adj.get_upper())
            return False

        GLib.idle_add(_go)

    def present_chat(self) -> None:  # noqa: D401 — Gio-style
        self.present()
        self._input.grab_focus()

    def show_toast(self, title: str) -> None:
        self._toast.add_toast(Adw.Toast(title=title))

    def set_listening(self, active: bool) -> None:
        if not hasattr(self, "_mic_button"):
            return
        if active:
            self._mic_button.add_css_class("suggested-action")
            self._mic_button.set_tooltip_text("Aufnahme läuft — klicken zum Stoppen")
        else:
            self._mic_button.remove_css_class("suggested-action")
            self._mic_button.set_tooltip_text("Zuhören (Push-to-talk)")

    def _handoff_coding(self) -> None:
        from kiki.agents.handoff import coding_task_from_chat

        buf = self._input.get_buffer()
        draft = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        last_user = ""
        if self._conversation is not None:
            for msg in reversed(self._chats.list_messages(self._conversation.id)):
                if msg.role == "user":
                    last_user = msg.content.split("\n\n_[Bild")[0].strip()
                    break
        try:
            task = coding_task_from_chat(draft=draft, last_user=last_user)
        except ValueError as exc:
            self.show_toast(str(exc))
            return
        app = self.get_application()
        handoff = getattr(app, "open_coding_with_task", None)
        if not callable(handoff):
            self.show_toast("Coding-Session ist nicht bereit.")
            return
        handoff(task)

    def append_note(self, text: str) -> None:
        """Show a local note in the open chat and persist it as assistant text."""
        if self._conversation is None:
            self._new_conversation()
        assert self._conversation is not None
        body = (text or "").strip()
        if not body:
            return
        self._chats.add_message(self._conversation.id, "assistant", body)
        self._messages.append(ChatBubble("assistant", body))
        self._scroll_to_end()
        self.show_toast("Coding-Zusammenfassung im Chat.")

    def submit_transcript(self, text: str, *, send: bool) -> None:
        self._input.get_buffer().set_text(text)
        if send and text.strip():
            self._send()
        else:
            self._input.grab_focus()

    def submit_vision(self, text: str, image_path: Path, *, label: str = "Bildschirm") -> None:
        try:
            encoded = encode_image_file(image_path)
        except VisionEncodeError as exc:
            self.show_toast(str(exc))
            return
        self._pending_images = [{"name": label, "b64": encoded}]
        self._refresh_attach_bar()
        self._input.get_buffer().set_text(text)
        self._send()
