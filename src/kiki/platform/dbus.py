"""Synchronous session-bus helpers for the desktop control tools.

The integrations read cached properties through proxies; the control tools
instead call methods and read/write properties directly. Both styles stay on
the session bus and both fail with plain exceptions — callers translate them
into tool error dicts, never into UI crashes.

Gio is imported lazily so test runs without a bus never touch it.
"""

from __future__ import annotations

from typing import Any

DEFAULT_TIMEOUT_MS = 5000


class BusError(RuntimeError):
    """A D-Bus call failed. The message is safe to show in a tool result."""


def session_bus() -> Any:
    try:
        from gi.repository import Gio
    except Exception as exc:  # pragma: no cover - depends on the platform
        raise BusError(f"Gio fehlt: {exc}") from exc
    try:
        return Gio.bus_get_sync(Gio.BusType.SESSION, None)
    except Exception as exc:
        raise BusError(f"Sitzungsbus nicht erreichbar: {exc}") from exc


def system_bus() -> Any:
    try:
        from gi.repository import Gio
    except Exception as exc:  # pragma: no cover - depends on the platform
        raise BusError(f"Gio fehlt: {exc}") from exc
    try:
        return Gio.bus_get_sync(Gio.BusType.SYSTEM, None)
    except Exception as exc:
        raise BusError(f"Systembus nicht erreichbar: {exc}") from exc


def call(
    bus: Any,
    destination: str,
    path: str,
    interface: str,
    method: str,
    parameters: Any = None,
    reply_type: Any = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> Any:
    """One synchronous call. Raises BusError instead of leaking GLib errors."""
    try:
        from gi.repository import Gio
    except Exception as exc:  # pragma: no cover
        raise BusError(f"Gio fehlt: {exc}") from exc
    try:
        return bus.call_sync(
            destination,
            path,
            interface,
            method,
            parameters,
            reply_type,
            Gio.DBusCallFlags.NONE,
            timeout_ms,
            None,
        )
    except Exception as exc:
        raise BusError(str(exc) or f"{destination}.{method} fehlgeschlagen.") from exc


def list_names(bus: Any) -> list[str]:
    from gi.repository import GLib

    reply = call(
        bus,
        "org.freedesktop.DBus",
        "/org/freedesktop/DBus",
        "org.freedesktop.DBus",
        "ListNames",
        None,
        GLib.VariantType("(as)"),
    )
    return list(reply.unpack()[0])


def property_get(
    bus: Any,
    destination: str,
    path: str,
    interface: str,
    name: str,
) -> Any:
    from gi.repository import GLib

    reply = call(
        bus,
        destination,
        path,
        "org.freedesktop.DBus.Properties",
        "Get",
        GLib.Variant("(ss)", (interface, name)),
        GLib.VariantType("(v)"),
    )
    return reply.unpack()[0]


def property_set(
    bus: Any,
    destination: str,
    path: str,
    interface: str,
    name: str,
    value: Any,
) -> None:
    from gi.repository import GLib

    call(
        bus,
        destination,
        path,
        "org.freedesktop.DBus.Properties",
        "Set",
        GLib.Variant("(ssv)", (interface, name, value)),
        None,
    )
