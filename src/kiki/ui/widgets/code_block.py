from __future__ import annotations

from gi.repository import Gdk, Gtk

from kiki.text.markdown import CodeBlock


class CodeBlockWidget(Gtk.Box):
    def __init__(self, block: CodeBlock) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add_css_class("kiki-code-block")
        self.add_css_class("card")
        self._code = block.code
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header.add_css_class("kiki-code-header")
        lang = Gtk.Label(label=block.language or "code", xalign=0, hexpand=True)
        lang.add_css_class("dim-label")
        copy = Gtk.Button(label="Kopieren")
        copy.add_css_class("flat")
        copy.connect("clicked", self._on_copy)
        header.append(lang)
        header.append(copy)
        view = Gtk.TextView(
            editable=False,
            cursor_visible=False,
            wrap_mode=Gtk.WrapMode.NONE,
            monospace=True,
        )
        view.add_css_class("kiki-code-body")
        view.get_buffer().set_text(block.code)
        scroll = Gtk.ScrolledWindow(
            hexpand=True,
            vexpand=False,
            min_content_height=min(240, 24 + 18 * max(1, block.code.count("\n") + 1)),
            max_content_height=320,
            propagate_natural_height=True,
            hscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        scroll.set_child(view)
        self.append(header)
        self.append(scroll)
        self._toast_host: Gtk.Widget | None = None

    def _on_copy(self, _button: Gtk.Button) -> None:
        display = self.get_display()
        if display is None:
            return
        display.get_clipboard().set(self._code)
        _button.set_label("Kopiert")

        def _reset() -> bool:
            _button.set_label("Kopieren")
            return False

        from gi.repository import GLib

        GLib.timeout_add(1200, _reset)
        # Gdk imported so type checkers keep the clipboard path honest.
        _ = Gdk
