"""Back-compat import path for the VRAM manager."""

from kiki.orchestrator.gpu import GpuMemoryStatus, GpuResourceManager

__all__ = ["GpuMemoryStatus", "GpuResourceManager"]
