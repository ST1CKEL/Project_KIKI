from __future__ import annotations

from pathlib import Path

from kiki.integrations.disk import DiskIntegration


def test_disk_snapshot(tmp_path: Path) -> None:
    snap = DiskIntegration(tmp_path).snapshot()
    assert snap.available is True
    assert snap.data["free_bytes"] >= 0
    assert "GiB" in snap.data["free_human"] or "MiB" in snap.data["free_human"] or "B" in snap.data["free_human"]
