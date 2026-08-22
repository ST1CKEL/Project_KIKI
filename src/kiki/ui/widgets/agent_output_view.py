"""Timestamped agent log. Monospace, append-only."""

from __future__ import annotations

from gi.repository import Gtk

from kiki.agents.models import AgentEvent


class AgentOutputView(Gtk.ScrolledWindow):
    def __init__(self) -> None:
        super().__init__(vexpand=True, hexpand=True)
        self.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self._view = Gtk.TextView(editable=False, wrap_mode=Gtk.WrapMode.WORD_CHAR, monospace=True)
        self._view.add_css_class("kiki-mono")
        self.set_child(self._view)
        self._seen = 0

    def clear(self) -> None:
        self._view.get_buffer().set_text("", -1)
        self._seen = 0

    def set_events(self, events: list[AgentEvent]) -> None:
        if len(events) < self._seen:
            self.clear()
        if len(events) == self._seen:
            return
        buf = self._view.get_buffer()
        for event in events[self._seen :]:
            stamp = (event.ts or "")[11:19]
            line = f"[{stamp}] {event.type.value}: {event.text}\n"
            buf.insert(buf.get_end_iter(), line)
        self._seen = len(events)
        mark = buf.create_mark(None, buf.get_end_iter(), False)
        self._view.scroll_to_mark(mark, 0.0, False, 0.0, 1.0)


class MonoText(Gtk.ScrolledWindow):
    def __init__(self) -> None:
        super().__init__(vexpand=True, hexpand=True)
        self._view = Gtk.TextView(editable=False, wrap_mode=Gtk.WrapMode.CHAR, monospace=True)
        self._view.add_css_class("kiki-mono")
        self.set_child(self._view)

    def set_text(self, text: str) -> None:
        self._view.get_buffer().set_text(text or "", -1)

    def text(self) -> str:
        buf = self._view.get_buffer()
        return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
