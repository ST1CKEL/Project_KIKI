"""XDG autostart desktop file in ~/.config/autostart."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from kiki.paths import bundled_data_dir, xdg_config_home

log = logging.getLogger(__name__)

DESKTOP_NAME = "io.github.projectkiki.Kiki.desktop"


def autostart_path() -> Path:
    return xdg_config_home() / "autostart" / DESKTOP_NAME


def is_enabled() -> bool:
    path = autostart_path()
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if "Hidden=true" in text:
        return False
    if "X-GNOME-Autostart-enabled=false" in text:
        return False
    return True


def set_enabled(enabled: bool, exec_line: str | None = None) -> None:
    path = autostart_path()
    if not enabled:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    command = exec_line or os.environ.get("KIKI_EXEC") or "kiki"
    template = bundled_data_dir() / DESKTOP_NAME
    if template.is_file():
        body = template.read_text(encoding="utf-8")
        lines = []
        for line in body.splitlines():
            if line.startswith("Exec="):
                lines.append(f"Exec={command}")
            elif line.startswith("TryExec="):
                lines.append(f"TryExec={command}")
            elif line.startswith("X-GNOME-Autostart-enabled="):
                continue
            else:
                lines.append(line)
        lines.append("X-GNOME-Autostart-enabled=true")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    path.write_text(
        "\n".join(
            [
                "[Desktop Entry]",
                "Type=Application",
                "Name=KIKI",
                "Comment=Freundliches 2D-KI-Desktop-Pet",
                f"Exec={command}",
                "Icon=io.github.projectkiki.Kiki",
                "Terminal=false",
                "Categories=Utility;",
                "X-GNOME-Autostart-enabled=true",
                "",
            ]
        ),
        encoding="utf-8",
    )
