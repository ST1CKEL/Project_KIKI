from __future__ import annotations

import json
from collections.abc import Callable

from kiki.tools.registry import ActionPreview
from kiki.ui.gi_bootstrap import Adw, Gtk


def present_confirmation(
    parent: Gtk.Widget | None,
    preview: ActionPreview,
    callback: Callable[[bool], None],
) -> None:
    dialog = Adw.AlertDialog(
        heading="Aktion bestätigen",
        body=(
            "KIKI führt nichts allein aus. Prüfe Ziel und Wirkung, "
            "bevor du zustimmst."
        ),
    )
    dialog.add_response("cancel", "Abbrechen")
    dialog.add_response("confirm", "Ausführen")
    dialog.set_default_response("cancel")
    dialog.set_close_response("cancel")
    appearance = Adw.ResponseAppearance.DESTRUCTIVE
    if preview.risk.value == "read":
        appearance = Adw.ResponseAppearance.SUGGESTED
    dialog.set_response_appearance("confirm", appearance)

    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    for title, value in (
        ("Tool", f"{preview.title} (`{preview.tool}`)"),
        ("Risiko", preview.risk.value),
        ("Ziel", preview.target),
        ("Wirkung", preview.effect),
        ("Warum erscheint die Freigabe?", preview.reason),
        ("Parameter", json.dumps(preview.params, ensure_ascii=False, indent=2) or "{}"),
    ):
        row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        lab = Gtk.Label(label=title, xalign=0)
        lab.add_css_class("caption")
        lab.add_css_class("dim-label")
        val = Gtk.Label(label=value, xalign=0, wrap=True, selectable=True)
        row.append(lab)
        row.append(val)
        box.append(row)
    dialog.set_extra_child(box)

    def _on_response(_dialog: Adw.AlertDialog, response: str) -> None:
        callback(response == "confirm")

    dialog.connect("response", _on_response)
    dialog.present(parent)
