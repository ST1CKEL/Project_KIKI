"""Must be imported before any gi.repository module."""

from __future__ import annotations

import warnings

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Gsk", "4.0")
gi.require_version("Graphene", "1.0")
gi.require_version("Adw", "1")
gi.require_version("GdkPixbuf", "2.0")
gi.require_version("Pango", "1.0")

# Importing through this module lets composition roots enforce the version
# contract without a deliberately unsorted "bootstrap, then gi.repository"
# import block of their own. PyGObject 3.54 on Fedora 44 inspects GLib's
# deprecated compatibility symbol while loading its overrides and warns even
# though KIKI never calls it. Force that lazy load inside a narrowly scoped
# filter; every other PyGI deprecation remains visible.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=(
            r"GLib\.unix_signal_add_full is deprecated; "
            r"use GLibUnix\.signal_add_full instead"
        ),
        category=gi.PyGIDeprecationWarning,
    )
    from gi.repository import (  # noqa: E402
        Adw,
        Gdk,
        GdkPixbuf,
        Gio,
        GLib,
        Graphene,
        Gtk,
        Pango,
    )

__all__ = [
    "Adw",
    "Gdk",
    "GdkPixbuf",
    "Gio",
    "GLib",
    "Graphene",
    "Gtk",
    "Pango",
    "gi",
]
