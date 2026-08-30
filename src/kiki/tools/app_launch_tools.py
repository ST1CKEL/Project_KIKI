"""Launch installed applications by their desktop entry.

The index is built from the two XDG application directories and nothing else.
`app.open` only accepts an app_id the index knows and spawns
`gio launch <file.desktop>` — a fixed argv in which no model text appears, so
a hallucinated Exec line can never become a command.
"""

from __future__ import annotations

import configparser
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiki.runners.process import desktop_env
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec

log = logging.getLogger(__name__)

_MAX_RESULTS = 30
_CACHE_TTL_S = 300.0


def default_application_dirs() -> list[Path]:
    """User directory first — it shadows identically named system entries."""
    dirs: list[Path] = []
    data_home = os.environ.get("XDG_DATA_HOME")
    user_dir = Path(data_home) if data_home else Path.home() / ".local" / "share"
    dirs.append(user_dir / "applications")
    dirs.extend(
        Path(part) / "applications"
        for part in os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(":")
        if part.strip()
    )
    return [d for d in dirs if d.is_dir()]


@dataclass(frozen=True)
class DesktopEntry:
    app_id: str
    name: str
    path: Path


class DesktopIndex:
    """Cached view over the application directories. Refreshes on staleness.

    Directory mtimes catch installs and removals; the TTL catches in-place
    edits, which do not bump a directory mtime.
    """

    def __init__(self, dirs: list[Path] | None = None, *, clock=time.monotonic) -> None:
        self._dirs = dirs if dirs is not None else default_application_dirs()
        self._clock = clock
        self._entries: dict[str, DesktopEntry] = {}
        self._built_at = -_CACHE_TTL_S
        self._dir_stamp: tuple[float, ...] | None = None

    def _stale(self) -> bool:
        if self._dir_stamp is None:
            return True
        if self._clock() - self._built_at >= _CACHE_TTL_S:
            return True
        try:
            return tuple(d.stat().st_mtime for d in self._dirs) != self._dir_stamp
        except OSError:
            return True

    def _rebuild(self) -> None:
        entries: dict[str, DesktopEntry] = {}
        for directory in self._dirs:
            try:
                candidates = sorted(directory.glob("*.desktop"))
            except OSError:
                continue
            for file_path in candidates:
                entry = _parse_entry(file_path)
                if entry is not None:
                    # Earlier directories win: user entries shadow system ones.
                    entries.setdefault(entry.app_id, entry)
        try:
            self._dir_stamp = tuple(d.stat().st_mtime for d in self._dirs)
        except OSError:
            self._dir_stamp = None
        self._entries = entries
        self._built_at = self._clock()

    def _current(self) -> dict[str, DesktopEntry]:
        if self._stale():
            self._rebuild()
        return self._entries

    def entries(self) -> list[DesktopEntry]:
        """All indexed entries, alphabetical. `list` caps this for display."""
        return sorted(self._current().values(), key=lambda e: e.name.lower())

    def list(self, query: str | None = None) -> list[DesktopEntry]:
        entries = self.entries()
        needle = (query or "").strip().lower()
        if needle:
            entries = [
                e for e in entries if needle in e.name.lower() or needle in e.app_id.lower()
            ]
        return entries[:_MAX_RESULTS]

    def find(self, app_id: str) -> DesktopEntry | None:
        """Exact id first, then an unambiguous substring match."""
        needle = app_id.strip().lower()
        if not needle:
            return None
        entries = self._current()
        exact = entries.get(app_id.strip())
        if exact is not None:
            return exact
        matches = [
            e for e in entries.values() if needle in e.app_id.lower() or needle in e.name.lower()
        ]
        if len(matches) == 1:
            return matches[0]
        return None


def _parse_entry(file_path: Path) -> DesktopEntry | None:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        # .desktop keys use `=` and never continuation lines; one read is enough.
        parser.read(file_path, encoding="utf-8")
    except (configparser.Error, OSError, UnicodeDecodeError):
        return None
    if not parser.has_section("Desktop Entry"):
        return None
    section = parser["Desktop Entry"]
    if section.getboolean("NoDisplay", fallback=False) or section.getboolean(
        "Hidden", fallback=False
    ):
        return None
    if section.get("Type", "Application").strip() != "Application":
        return None
    name = (section.get("Name") or "").strip()
    if not name:
        return None
    return DesktopEntry(app_id=file_path.stem, name=name, path=file_path)


_INTERPRETERS = {"python", "python3", "sh", "bash", "node", "ruby", "perl"}
_EXEC_SKIP = {"env", "nohup", "stdbuf"}
_EXEC_FIELD_CODES = re.compile(r"%[fFuUdDnNickvm]")


def desktop_exec_binary(desktop_path: Path) -> str | None:
    """The program a .desktop entry runs: Exec's first real token, basenamed.

    Deliberately conservative: entries that run through a shell wrapper or a
    command string yield None, and app.close refuses to guess from there.
    """
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read(desktop_path, encoding="utf-8")
    except (configparser.Error, OSError):
        return None
    if not parser.has_section("Desktop Entry"):
        return None
    exec_line = parser["Desktop Entry"].get("Exec", "").strip()
    if not exec_line:
        return None
    cleaned = _EXEC_FIELD_CODES.sub("", exec_line)
    try:
        tokens = shlex.split(cleaned)
    except ValueError:
        return None
    for token in tokens:
        base = os.path.basename(token)
        if base in _EXEC_SKIP:
            continue
        if "=" in base and not token.startswith(("-", "/")):
            continue  # env-style assignment
        if token.startswith("-"):
            break
        if base in _EXEC_SKIP or base in _INTERPRETERS:
            # Interpreter without a visible script name: nothing to signal.
            break
        return base
    return None


def matching_pids(
    binary: str, *, proc_dir: Path = Path("/proc"), exclude_pid: int | None = None
) -> list[int]:
    """Running processes whose program is this one; KIKI is never included."""
    if exclude_pid is None:
        exclude_pid = os.getpid()
    found: list[int] = []
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == exclude_pid:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\x00")
        except OSError:
            continue
        if not argv or not argv[0]:
            continue
        name = os.path.basename(argv[0].decode("utf-8", "replace"))
        if name in _INTERPRETERS and len(argv) > 1 and argv[1]:
            name = os.path.basename(argv[1].decode("utf-8", "replace"))
        if name == binary:
            found.append(pid)
    return found


def _launch(argv: list[str]) -> None:
    env = desktop_env(home=os.environ.get("HOME") or "/tmp")
    subprocess.Popen(
        argv,
        cwd=str(Path.home()),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


class AppLaunchSkill:
    id = "app_launch"
    name = "Anwendungen"
    description = "Installierte Anwendungen über ihren Desktop-Eintrag starten."

    def __init__(self, index: DesktopIndex | None = None) -> None:
        self._index = index or DesktopIndex()

    def tools(self) -> list[ToolSpec]:
        return [self._list_spec(), self._open_spec(), self._close_spec()]

    def _list_spec(self) -> ToolSpec:
        return ToolSpec(
            name="app.list",
            title="Anwendungen auflisten",
            description=(
                "Findet installierte Anwendungen mit ihrer app_id. Optionale "
                "`query` filtert nach Namen. Die app_id für app.open kommt von hier."
            ),
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 64}
                },
                "additionalProperties": False,
            },
            handler=self._list,
            effect="Liest den Anwendungsindex aus den XDG-Verzeichnissen. Keine Änderung.",
            target="Anwendungsindex",
            auto_allow=True,
            model_callable=True,
        )

    def _open_spec(self) -> ToolSpec:
        return ToolSpec(
            name="app.open",
            title="Anwendung starten",
            description=(
                "Startet eine installierte Anwendung über ihre app_id aus app.list "
                "(z. B. `org.gnome.Calculator` oder `firefox`)."
            ),
            risk=RiskLevel.LAUNCH,
            parameters={
                "type": "object",
                "properties": {
                    "app_id": {"type": "string", "minLength": 1, "maxLength": 128}
                },
                "required": ["app_id"],
                "additionalProperties": False,
            },
            handler=self._open,
            effect="Öffnet die Anwendung auf dem Desktop. Ändert keine Daten.",
            target="Desktop",
            auto_allow=True,
            model_callable=True,
        )

    def _close_spec(self) -> ToolSpec:
        return ToolSpec(
            name="app.close",
            title="Anwendung beenden",
            description=(
                "Beendet eine laufende Anwendung über ihre app_id aus app.list "
                "(z. B. `firefox` oder `org.thunderbird.Thunderbird`)."
            ),
            risk=RiskLevel.CONTROL,
            parameters={
                "type": "object",
                "properties": {
                    "app_id": {"type": "string", "minLength": 1, "maxLength": 128}
                },
                "required": ["app_id"],
                "additionalProperties": False,
            },
            handler=self._close,
            effect=(
                "Sendet ein Beenden-Signal (SIGTERM) an die Prozesse der "
                "Anwendung. Kein erzwungenes Kill: Das Programm kann sich "
                "selbst schützen (z. B. nachfragen, ob gespeichert werden soll)."
            ),
            target="Laufende Anwendung",
            auto_allow=True,
            model_callable=True,
        )

    def _list(self, params: dict[str, Any]) -> dict[str, Any]:
        query = params.get("query")
        entries = self._index.list(str(query) if query else None)
        return {
            "ok": True,
            "count": len(entries),
            "apps": [{"app_id": e.app_id, "name": e.name} for e in entries],
        }

    def _open(self, params: dict[str, Any]) -> dict[str, Any]:
        entry = self._index.find(str(params["app_id"]))
        if entry is None:
            return {
                "ok": False,
                "error": (
                    f"Anwendung „{params['app_id']}“ nicht eindeutig gefunden. "
                    "app.list zeigt die verfügbaren app_ids."
                ),
            }
        gio = shutil.which("gio")
        if gio is None:
            return {"ok": False, "error": "gio nicht gefunden — Start nicht möglich."}
        try:
            resolved = entry.path.resolve(strict=True)
            if not resolved.is_file():
                raise OSError("Desktop-Datei ist verschwunden.")
            _launch([gio, "launch", str(resolved)])
        except OSError as exc:
            return {"ok": False, "error": f"Anwendung konnte nicht gestartet werden: {exc}"}
        return {"ok": True, "app": entry.app_id, "name": entry.name}

    def _close(self, params: dict[str, Any]) -> dict[str, Any]:
        entry = self._index.find(str(params["app_id"]))
        if entry is None:
            return {
                "ok": False,
                "error": (
                    f"Anwendung „{params['app_id']}“ nicht eindeutig gefunden. "
                    "app.list zeigt die verfügbaren app_ids."
                ),
            }
        binary = desktop_exec_binary(entry.path)
        if binary is None:
            return {
                "ok": False,
                "error": f"Aus dem Desktop-Eintrag von {entry.name} lässt sich kein "
                "Programm ableiten — Beenden nicht möglich.",
            }
        pids = matching_pids(binary)
        if not pids:
            return {"ok": False, "error": f"{entry.name} läuft gerade nicht."}
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                continue
        time.sleep(1.0)
        running = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
        result: dict[str, Any] = {"ok": True, "app": entry.app_id, "name": entry.name}
        if running:
            # Still alive after the polite signal: apps with unsaved work show
            # their own "save?" dialog. Never escalate to SIGKILL.
            result["note"] = (
                f"{entry.name} schließt sich noch (oder wartet auf eine Antwort)."
            )
        return result
