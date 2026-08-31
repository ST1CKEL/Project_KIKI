"""Runtime socket paths. Never hard-code a uid — $XDG_RUNTIME_DIR owns this."""

from __future__ import annotations

import os
from pathlib import Path


def runtime_dir(explicit: str | Path | None = None) -> Path:
    """Directory for Unix sockets and tmpfs turn audio.

    Resolution order:
    1. explicit argument (tests, CLI)
    2. ``KIKI_RUNTIME_DIR``
    3. ``$XDG_RUNTIME_DIR/kiki``
    4. ``/run/user/$UID/kiki`` — last resort, still per-user
    """
    if explicit:
        path = Path(explicit)
    else:
        env = os.environ.get("KIKI_RUNTIME_DIR", "").strip()
        if env:
            path = Path(env)
        else:
            xdg = os.environ.get("XDG_RUNTIME_DIR", "").strip()
            if xdg:
                path = Path(xdg) / "kiki"
            else:
                path = Path(f"/run/user/{os.getuid()}") / "kiki"
    path.mkdir(parents=True, exist_ok=True)
    return path


def socket_path(name: str, *, runtime: Path | None = None) -> Path:
    """``name`` is a stem like ``audio`` → ``<runtime>/audio.sock``."""
    stem = name[:-5] if name.endswith(".sock") else name
    return (runtime or runtime_dir()) / f"{stem}.sock"


def turns_dir(*, runtime: Path | None = None) -> Path:
    path = (runtime or runtime_dir()) / "turns"
    path.mkdir(parents=True, exist_ok=True)
    return path
