"""User-initiated screen capture. Wayland: xdg-desktop-portal, then Spectacle."""

from __future__ import annotations

import logging
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path

from kiki.paths import cache_dir

log = logging.getLogger(__name__)

DoneFn = Callable[[Path | None, str | None], None]


class ScreenshotError(Exception):
    """Capture failed or was cancelled."""


def screenshot_dir() -> Path:
    path = cache_dir() / "screenshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_screenshot_path() -> Path:
    return screenshot_dir() / f"kiki-{uuid.uuid4().hex}.png"


def capture_screenshot(on_done: DoneFn, *, interactive: bool = True) -> None:
    """GTK-thread capture. on_done(path, error)."""
    try:
        _capture_portal(on_done, interactive=interactive)
    except Exception as exc:
        log.info("portal screenshot unavailable (%s), trying spectacle", exc)
        _capture_spectacle(on_done)


def _capture_portal(on_done: DoneFn, *, interactive: bool) -> None:
    from gi.repository import Gio, GLib

    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    unique = bus.get_unique_name() or ":0.0"
    sender = unique[1:].replace(".", "_")
    token = uuid.uuid4().hex[:16]
    request_path = f"/org/freedesktop/portal/desktop/request/{sender}/{token}"

    def on_response(
        _conn: Gio.DBusConnection,
        _sender: str,
        _path: str,
        _iface: str,
        _signal: str,
        params: GLib.Variant,
    ) -> None:
        try:
            status, results = params.unpack()
        except Exception as exc:
            on_done(None, f"Portal-Antwort unlesbar: {exc}")
            return
        if int(status) != 0:
            on_done(None, "cancelled")
            return
        uri = ""
        if isinstance(results, dict):
            uri = str(results.get("uri") or "")
        if not uri.startswith("file:"):
            on_done(None, "Portal lieferte keine Datei.")
            return
        from urllib.parse import unquote, urlparse

        path = Path(unquote(urlparse(uri).path))
        if not path.is_file():
            on_done(None, f"Screenshot fehlt: {path}")
            return
        on_done(path, None)

    bus.signal_subscribe(
        None,
        "org.freedesktop.portal.Request",
        "Response",
        request_path,
        None,
        Gio.DBusSignalFlags.NONE,
        on_response,
    )

    proxy = Gio.DBusProxy.new_sync(
        bus,
        Gio.DBusProxyFlags.NONE,
        None,
        "org.freedesktop.portal.Desktop",
        "/org/freedesktop/portal/desktop",
        "org.freedesktop.portal.Screenshot",
        None,
    )
    options = {
        "handle_token": GLib.Variant("s", token),
        "interactive": GLib.Variant("b", bool(interactive)),
        "modal": GLib.Variant("b", True),
    }
    def on_call(source: Gio.DBusProxy, result: Gio.AsyncResult, _data: object) -> None:
        try:
            source.call_finish(result)
        except Exception as exc:
            log.info("portal Screenshot call failed: %s", exc)
            _capture_spectacle(on_done)

    proxy.call(
        "Screenshot",
        GLib.Variant("(sa{sv})", ("", options)),
        Gio.DBusCallFlags.NONE,
        120_000,
        None,
        on_call,
        None,
    )


def _capture_spectacle(on_done: DoneFn) -> None:
    dest = new_screenshot_path()

    def _finish(path: Path | None, error: str | None) -> None:
        try:
            from gi.repository import GLib

            GLib.idle_add(lambda: (on_done(path, error), False)[1])
        except Exception:
            on_done(path, error)

    def _run() -> None:
        try:
            proc = subprocess.run(
                ["spectacle", "-b", "-n", "-f", "-o", str(dest)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            _finish(None, f"Spectacle fehlgeschlagen: {exc}")
            return
        if dest.is_file() and dest.stat().st_size > 0:
            _finish(dest, None)
            return
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        _finish(None, f"Kein Screenshot: {err}")

    import threading

    threading.Thread(target=_run, name="kiki-screenshot", daemon=True).start()
