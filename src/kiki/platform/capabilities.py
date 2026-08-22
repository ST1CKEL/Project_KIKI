"""Runtime facts about the display server. No window manipulation here."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformCapabilities:
    display_backend: str  # wayland, x11, unknown
    can_position_window: bool
    can_keep_above: bool
    can_set_input_region: bool
    layer_shell: bool
    notes: tuple[str, ...]


def detect_capabilities() -> PlatformCapabilities:
    backend = (os.environ.get("GDK_BACKEND") or "").split(",")[0].strip().lower()
    session = (os.environ.get("XDG_SESSION_TYPE") or "").lower()
    if not backend:
        if os.environ.get("WAYLAND_DISPLAY"):
            backend = "wayland"
        elif os.environ.get("DISPLAY"):
            backend = "x11"
        elif session in {"wayland", "x11"}:
            backend = session
        else:
            backend = "unknown"

    notes: list[str] = []
    if backend == "wayland":
        notes.append(
            "GNOME/Wayland verbietet Apps, ihre Fensterposition zu setzen. "
            "Verschieben geht über gdk_toplevel_begin_move (interaktiv)."
        )
        notes.append(
            "Always-on-top kann unter Wayland nicht programmatisch gesetzt werden. "
            "Nutze das GNOME-Fenstermenü (Alt+Leertaste → Immer im Vordergrund)."
        )
        notes.append(
            "wlr-layer-shell wird von GNOME nicht unterstützt. KIKI nutzt ein normales Gtk.Window."
        )
        can_pos = False
        can_above = False
        can_input = True
        layer = False
    elif backend == "x11":
        notes.append("X11/XWayland: Position und _NET_WM_STATE_ABOVE werden best-effort gesetzt.")
        can_pos = True
        can_above = True
        can_input = True
        layer = False
    else:
        can_pos = False
        can_above = False
        can_input = False
        layer = False
        notes.append("Unbekanntes Display-Backend.")
    return PlatformCapabilities(
        display_backend=backend,
        can_position_window=can_pos,
        can_keep_above=can_above,
        can_set_input_region=can_input,
        layer_shell=layer,
        notes=tuple(notes),
    )
