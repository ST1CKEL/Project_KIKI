from __future__ import annotations

from gi.repository import Gtk, Pango

from kiki.ui.widgets.markdown_view import MarkdownView


class ChatBubble(Gtk.Box):
    def __init__(self, role: str, text: str, *, streaming: bool = False) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.set_hexpand(True)
        self._role = role
        self.add_css_class("kiki-bubble")
        self.add_css_class("user" if role == "user" else "assistant")
        if role == "error":
            self.add_css_class("error")
        who = {"user": "Du", "assistant": "KIKI", "error": "Fehler"}.get(role, role)
        header = Gtk.Label(label=who, xalign=0)
        header.add_css_class("dim-label")
        header.add_css_class("caption")
        self.append(header)
        self._stream_label: Gtk.Label | None = None
        self._body: Gtk.Widget | None = None
        self._activity: Gtk.Label | None = None
        self._activity_lines: list[str] = []
        if streaming:
            self._stream_label = Gtk.Label(
                xalign=0,
                wrap=True,
                wrap_mode=Pango.WrapMode.WORD_CHAR,
                selectable=True,
                hexpand=True,
            )
            self._stream_label.set_text(text)
            self.append(self._stream_label)
            self._body = self._stream_label
        else:
            view = MarkdownView(text)
            self.append(view)
            self._body = view

    def note_activity(self, line: str) -> None:
        """Show which tool KIKI reached for. Stays visible after the turn ends."""
        if not line:
            return
        self._activity_lines.append(line)
        if self._activity is None:
            self._activity = Gtk.Label(xalign=0, wrap=True, selectable=True)
            self._activity.add_css_class("caption")
            self._activity.add_css_class("dim-label")
            # Directly under the "KIKI" header, above whatever body exists.
            self.insert_child_after(self._activity, self.get_first_child())
        self._activity.set_text("\n".join(self._activity_lines))

    def replace_last_activity(self, line: str) -> None:
        """Update the running line in place once the tool has finished."""
        if not self._activity_lines or self._activity is None:
            self.note_activity(line)
            return
        self._activity_lines[-1] = line
        self._activity.set_text("\n".join(self._activity_lines))

    def append_delta(self, text: str) -> None:
        if self._stream_label is None:
            return
        current = self._stream_label.get_text()
        self._stream_label.set_text(current + text)

    def finish_markdown(self, full_text: str) -> None:
        if self._body is not None:
            self.remove(self._body)
        view = MarkdownView(full_text)
        self.append(view)
        self._body = view
        self._stream_label = None
