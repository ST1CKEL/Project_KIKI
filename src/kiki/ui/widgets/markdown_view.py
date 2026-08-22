from __future__ import annotations

from gi.repository import Gtk, Pango

from kiki.text.markdown import (
    CodeBlock,
    Heading,
    ListBlock,
    Paragraph,
    parse_markdown,
    spans_to_pango,
)
from kiki.ui.widgets.code_block import CodeBlockWidget


def _label(markup: str, *, css: str | None = None, wrap: bool = True) -> Gtk.Label:
    label = Gtk.Label(
        xalign=0,
        wrap=wrap,
        wrap_mode=Pango.WrapMode.WORD_CHAR,
        selectable=True,
        hexpand=True,
        use_markup=True,
    )
    label.set_markup(markup)
    if css:
        label.add_css_class(css)
    return label


class MarkdownView(Gtk.Box):
    def __init__(self, source: str) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.set_hexpand(True)
        for block in parse_markdown(source):
            if isinstance(block, CodeBlock):
                self.append(CodeBlockWidget(block))
            elif isinstance(block, Heading):
                markup = spans_to_pango(block.spans)
                self.append(_label(f"<b>{markup}</b>", css="heading"))
            elif isinstance(block, ListBlock):
                for idx, item in enumerate(block.items, start=1):
                    prefix = f"{idx}." if block.ordered else "•"
                    self.append(_label(f"{prefix} {spans_to_pango(item)}"))
            elif isinstance(block, Paragraph):
                self.append(_label(spans_to_pango(block.spans)))
