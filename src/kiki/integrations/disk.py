"""Free space on the home directory via statvfs — no D-Bus required."""

from __future__ import annotations

import os
from pathlib import Path

from kiki.integrations.base import IntegrationSnapshot


def _fmt_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(number)} {unit}"
            return f"{number:.1f} {unit}"
        number /= 1024
    return f"{value} B"


class DiskIntegration:
    id = "disk"
    title = "Speicherplatz"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else Path.home()

    def snapshot(self) -> IntegrationSnapshot:
        try:
            st = os.statvfs(self.path)
        except OSError as exc:
            return IntegrationSnapshot(
                id=self.id,
                title=self.title,
                available=False,
                data={},
                error=str(exc),
            )
        total = int(st.f_blocks * st.f_frsize)
        free = int(st.f_bavail * st.f_frsize)
        used = total - int(st.f_bfree * st.f_frsize)
        percent = (used / total * 100.0) if total else 0.0
        return IntegrationSnapshot(
            id=self.id,
            title=self.title,
            available=True,
            data={
                "path": str(self.path),
                "total_bytes": total,
                "free_bytes": free,
                "used_percent": round(percent, 1),
                "free_human": _fmt_bytes(free),
                "total_human": _fmt_bytes(total),
            },
        )
