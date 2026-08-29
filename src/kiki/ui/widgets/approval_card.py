"""Visible, text-based approval. Bound to one tool + params."""

from __future__ import annotations

import json
from collections.abc import Callable

from kiki.tools.registry import ActionPreview
from kiki.ui.gi_bootstrap import Gtk, Pango


class ApprovalCard(Gtk.Box):
    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.add_css_class("kiki-approval")
        self._callback: Callable[[bool], None] | None = None
        self._title = Gtk.Label(xalign=0, wrap=True)
        self._title.add_css_class("heading")
        self._body = Gtk.Label(xalign=0, wrap=True, selectable=True)
        self._body.set_ellipsize(Pango.EllipsizeMode.NONE)
        buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._cancel = Gtk.Button(label="Ablehnen")
        self._ok = Gtk.Button(label="Freigeben")
        self._ok.add_css_class("destructive-action")
        self._cancel.connect("clicked", lambda *_: self._decide(False))
        self._ok.connect("clicked", lambda *_: self._decide(True))
        buttons.append(self._cancel)
        buttons.append(self._ok)
        self.append(self._title)
        self.append(self._body)
        self.append(buttons)
        self.set_visible(False)

    def present(self, preview: ActionPreview, callback: Callable[[bool], None]) -> None:
        self._callback = callback
        self._title.set_text(f"Freigabe: {preview.title}")
        params = json.dumps(preview.params, ensure_ascii=False, indent=2)
        self._body.set_text(
            f"Tool: {preview.tool}\n"
            f"Risiko: {preview.risk.value}\n"
            f"Ziel: {preview.target}\n"
            f"Wirkung: {preview.effect}\n"
            f"Grund: {preview.reason}\n"
            f"Parameter:\n{params}"
        )
        self.set_visible(True)

    def dismiss(self) -> None:
        self._callback = None
        self.set_visible(False)

    def _decide(self, approved: bool) -> None:
        callback = self._callback
        self.dismiss()
        if callback is not None:
            callback(approved)
