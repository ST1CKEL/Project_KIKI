"""User-initiated desktop helpers. No global input simulation, no free shell."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from kiki.runners.process import desktop_env
from kiki.workspaces.models import WorkspaceError

WhichFn = Callable[[str], str | None]
ClipboardFn = Callable[[str], None]
NotificationFn = Callable[[str, str], None]

_MAX_URL = 2048
_ALLOWED_SCHEMES = frozenset({"http", "https"})
_MAX_CLIPBOARD = 8192
_MAX_NOTIFICATION_TITLE = 80
_MAX_NOTIFICATION_BODY = 400
_TERMINAL_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("xdg-terminal-exec", ("xdg-terminal-exec",)),
    ("kgx", ("kgx", "--working-directory={cwd}")),
    ("ptyxis", ("ptyxis", "--new-window", "--working-directory", "{cwd}")),
    ("gnome-terminal", ("gnome-terminal", "--working-directory={cwd}")),
)
_EDITOR_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("code", ("code", "--new-window", "{cwd}")),
    ("codium", ("codium", "--new-window", "{cwd}")),
    ("gnome-text-editor", ("gnome-text-editor", "{cwd}")),
    ("gedit", ("gedit", "{cwd}")),
)


def validate_http_url(raw: str) -> str:
    text = (raw or "").strip()
    if not text or len(text) > _MAX_URL:
        raise WorkspaceError("invalid_url", "URL fehlt oder ist zu lang.")
    if any(ch.isspace() for ch in text):
        raise WorkspaceError("invalid_url", "URL darf keine Leerzeichen enthalten.")
    parsed = urlparse(text)
    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise WorkspaceError("invalid_url", "Nur http(s)-URLs sind erlaubt.")
    if parsed.username or parsed.password:
        raise WorkspaceError("invalid_url", "URLs mit Zugangsdaten sind verboten.")
    if not parsed.netloc or parsed.netloc.startswith("."):
        raise WorkspaceError("invalid_url", "URL ohne gültigen Host.")
    return text


def validate_clipboard_text(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise WorkspaceError("invalid_text", "Text für die Zwischenablage fehlt.")
    if len(raw) > _MAX_CLIPBOARD:
        raise WorkspaceError(
            "invalid_text",
            f"Text für die Zwischenablage darf höchstens {_MAX_CLIPBOARD} Zeichen haben.",
        )
    if "\x00" in raw:
        raise WorkspaceError("invalid_text", "Text für die Zwischenablage enthält ein NUL-Zeichen.")
    return raw


def validate_notification(title: str, body: str) -> tuple[str, str]:
    clean_title = " ".join((title or "").split())
    clean_body = " ".join((body or "").split())
    if not clean_title or not clean_body:
        raise WorkspaceError("invalid_notification", "Titel und Nachricht dürfen nicht leer sein.")
    if len(clean_title) > _MAX_NOTIFICATION_TITLE:
        raise WorkspaceError(
            "invalid_notification",
            f"Der Titel darf höchstens {_MAX_NOTIFICATION_TITLE} Zeichen haben.",
        )
    if len(clean_body) > _MAX_NOTIFICATION_BODY:
        raise WorkspaceError(
            "invalid_notification",
            f"Die Nachricht darf höchstens {_MAX_NOTIFICATION_BODY} Zeichen haben.",
        )
    return clean_title, clean_body


def terminal_argv(cwd: Path, *, which: WhichFn | None = None) -> list[str]:
    """Fixed launcher templates. ``cwd`` is interpolated as one path argument."""
    lookup = which or shutil.which
    resolved = cwd.resolve(strict=True)
    if not resolved.is_dir():
        raise WorkspaceError("not_a_directory", f"Kein Verzeichnis: {resolved}")
    for name, template in _TERMINAL_CANDIDATES:
        binary = lookup(name)
        if not binary:
            continue
        argv = [binary if part == name else part.replace("{cwd}", str(resolved)) for part in template]
        argv[0] = binary
        if "-c" in argv:
            raise WorkspaceError("free_shell", "Terminal-Vorlage ungültig.")
        return argv
    raise WorkspaceError("no_handler", "Kein erlaubtes Terminal gefunden (kgx/ptyxis/gnome-terminal).")


def editor_argv(cwd: Path, *, which: WhichFn | None = None) -> list[str]:
    """Fixed IDE/editor templates. Only the workspace path is interpolated."""
    lookup = which or shutil.which
    resolved = cwd.resolve(strict=True)
    if not resolved.is_dir():
        raise WorkspaceError("not_a_directory", f"Kein Verzeichnis: {resolved}")
    for name, template in _EDITOR_CANDIDATES:
        binary = lookup(name)
        if not binary:
            continue
        argv = [part.replace("{cwd}", str(resolved)) for part in template]
        argv[0] = binary
        if "-c" in argv or "--command" in argv:
            raise WorkspaceError("free_shell", "Editor-Vorlage ungültig.")
        return argv
    raise WorkspaceError("no_handler", "Kein erlaubter Editor gefunden (code/codium/gnome-text-editor).")


def _launch(argv: list[str], *, cwd: Path) -> None:
    env = desktop_env(home=str(Path.home()))
    subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def open_path_in_file_manager(path: Path) -> None:
    """Open a directory with xdg-open. Caller must have already authorized it."""
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise WorkspaceError("not_a_directory", f"Kein Verzeichnis: {resolved}")
    binary = shutil.which("xdg-open")
    if binary is None:
        raise WorkspaceError("no_handler", "xdg-open fehlt.")
    _launch([binary, str(resolved)], cwd=resolved)


def open_file_with_default_app(path: Path) -> None:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise WorkspaceError("not_a_file", f"Keine Datei: {resolved}")
    binary = shutil.which("xdg-open")
    if binary is None:
        raise WorkspaceError("no_handler", "xdg-open fehlt.")
    _launch([binary, str(resolved)], cwd=resolved.parent)


def open_http_url(url: str) -> str:
    cleaned = validate_http_url(url)
    binary = shutil.which("xdg-open")
    if binary is None:
        raise WorkspaceError("no_handler", "xdg-open fehlt.")
    cwd = Path.home() if Path.home().is_dir() else Path("/")
    _launch([binary, cleaned], cwd=cwd)
    return cleaned


def open_terminal_at(cwd: Path, *, which: WhichFn | None = None) -> list[str]:
    argv = terminal_argv(cwd, which=which)
    _launch(argv, cwd=cwd)
    return argv


def open_editor_at(cwd: Path, *, which: WhichFn | None = None) -> list[str]:
    argv = editor_argv(cwd, which=which)
    _launch(argv, cwd=cwd)
    return argv


def copy_text_to_clipboard(text: str, *, setter: ClipboardFn | None = None) -> int:
    """Replace the desktop clipboard with one validated, user-confirmed string."""
    cleaned = validate_clipboard_text(text)
    if setter is not None:
        setter(cleaned)
        return len(cleaned)
    from gi.repository import Gdk

    display = Gdk.Display.get_default()
    if display is None:
        raise WorkspaceError("no_display", "Keine Desktop-Sitzung für die Zwischenablage gefunden.")
    display.get_clipboard().set(cleaned)
    return len(cleaned)


def show_desktop_notification(
    title: str,
    body: str,
    *,
    sender: NotificationFn | None = None,
) -> tuple[str, str]:
    """Show one local notification without invoking a shell or network client."""
    clean_title, clean_body = validate_notification(title, body)
    if sender is not None:
        sender(clean_title, clean_body)
        return clean_title, clean_body
    from gi.repository import Gio

    app = Gio.Application.get_default()
    if app is None:
        raise WorkspaceError("no_application", "KIKI-Anwendung ist für Benachrichtigungen nicht aktiv.")
    note = Gio.Notification.new(clean_title)
    note.set_body(clean_body)
    app.send_notification("kiki-desktop-control", note)
    return clean_title, clean_body
