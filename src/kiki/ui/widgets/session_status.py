"""Status line for coding sessions."""

from __future__ import annotations

from gi.repository import Gtk


class SessionStatus(Gtk.Box):
    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._label = Gtk.Label(xalign=0, wrap=True)
        self._label.add_css_class("dim-label")
        self.append(self._label)
        self.set_text("Keine Session.")

    def set_text(self, text: str) -> None:
        self._label.set_text(text)
