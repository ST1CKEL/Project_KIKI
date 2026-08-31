"""VRAM budget. Law 1: a missing GPU is not "16 GB free"."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass

log = logging.getLogger("kiki.gpu")


@dataclass(frozen=True)
class GpuMemoryStatus:
    total_mb: int
    used_mb: int
    free_mb: int
    utilization_pct: int
    available: bool
    source: str
    error: str = ""


class GpuResourceManager:
    def __init__(self, device_id: int = 0, safety_margin_mb: int = 2048) -> None:
        self.device_id = device_id
        self.safety_margin_mb = safety_margin_mb
        self._nvml = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            log.info("NVML ready for GPU %d", device_id)
        except Exception as exc:
            log.info("NVML unavailable (%s) — nvidia-smi fallback if present", exc)

    def get_memory_status(self) -> GpuMemoryStatus:
        if self._nvml is not None and self._handle is not None:
            try:
                info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                rates = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                return GpuMemoryStatus(
                    total_mb=int(info.total / (1024 * 1024)),
                    used_mb=int(info.used / (1024 * 1024)),
                    free_mb=int(info.free / (1024 * 1024)),
                    utilization_pct=int(rates.gpu),
                    available=True,
                    source="nvml",
                )
            except Exception as exc:
                log.debug("NVML query failed: %s", exc)
        smi = shutil.which("nvidia-smi")
        if smi is None:
            return GpuMemoryStatus(
                0, 0, 0, 0, False, "none", "nvidia-smi/NVML fehlen — VRAM unbekannt, Vision bleibt aus."
            )
        try:
            out = subprocess.check_output(
                [
                    smi,
                    f"--id={self.device_id}",
                    "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8",
                timeout=2,
            ).strip()
            total, used, free, util = [int(v.strip()) for v in out.split(",")]
            return GpuMemoryStatus(total, used, free, util, True, "nvidia-smi")
        except Exception as exc:
            return GpuMemoryStatus(
                0, 0, 0, 0, False, "none", f"GPU-Abfrage fehlgeschlagen: {exc}"
            )

    def can_allocate_vram(self, required_mb: int) -> tuple[bool, str]:
        status = self.get_memory_status()
        if not status.available:
            return False, status.error or "VRAM unbekannt — ich rate nicht."
        headroom = status.free_mb - self.safety_margin_mb
        if headroom < required_mb:
            return False, (
                f"Zu wenig VRAM: {required_mb} MB nötig, "
                f"{status.free_mb} MB frei, {self.safety_margin_mb} MB Reserve."
            )
        return True, ""
