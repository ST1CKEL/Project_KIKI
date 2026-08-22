"""VRAM budget for the KIKI harness.

The GPU is shared: the TTS service holds its model, the desktop and whatever
else the user runs take a slice, and the LLM has to fit in what is left. Rather
than assuming a fixed split, the harness keeps a budget and decides what stays
resident.

Priorities come from the context planner, so the same vocabulary applies:
`exclusive` may evict everything else, `high` may evict `low`, and `low` never
evicts anyone.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import StrEnum

log = logging.getLogger(__name__)

# Leave this much untouched. Filling VRAM to the brim makes the compositor
# stutter long before CUDA reports an error.
HEADROOM_BYTES = 768 * 1024 * 1024


class Priority(StrEnum):
    LOW = "low"
    HIGH = "high"
    EXCLUSIVE = "exclusive"


_RANK = {Priority.LOW: 0, Priority.HIGH: 1, Priority.EXCLUSIVE: 2}


@dataclass
class Resident:
    """Something currently holding VRAM that the harness itself manages."""

    name: str
    bytes_used: int
    priority: Priority = Priority.HIGH
    evictable: bool = True
    unload: object = None  # callable, invoked when evicted


@dataclass
class Decision:
    allowed: bool
    evict: list[str] = field(default_factory=list)
    reason: str = ""
    free_after: int = 0


class VramBudget:
    """Tracks what the harness has loaded and what it may still load."""

    def __init__(
        self,
        *,
        total_bytes: int,
        headroom_bytes: int = HEADROOM_BYTES,
        probe: object = None,
    ) -> None:
        self._total = int(total_bytes)
        self._headroom = int(headroom_bytes)
        # Reports free VRAM including memory held by other processes. Injected
        # so the decision logic is testable without a GPU.
        self._probe = probe
        self._residents: dict[str, Resident] = {}
        self._lock = threading.Lock()

    def free_bytes(self) -> int:
        if self._probe is not None:
            return int(self._probe())
        held = sum(r.bytes_used for r in self._residents.values())
        return max(0, self._total - held)

    def residents(self) -> list[Resident]:
        with self._lock:
            return list(self._residents.values())

    def register(self, resident: Resident) -> None:
        with self._lock:
            self._residents[resident.name] = resident

    def release(self, name: str) -> None:
        with self._lock:
            self._residents.pop(name, None)

    def plan(self, *, name: str, need_bytes: int, priority: Priority) -> Decision:
        """Decide whether `need_bytes` can be made available, and at what cost."""
        need = int(need_bytes)
        if need <= 0:
            return Decision(allowed=True, reason="nothing_to_load")
        if need + self._headroom > self._total:
            return Decision(
                allowed=False,
                reason=f"braucht {need / 1e9:.1f} GB, die Karte hat {self._total / 1e9:.1f} GB",
            )

        free = self.free_bytes()
        if free >= need + self._headroom:
            return Decision(allowed=True, reason="fits", free_after=free - need)

        with self._lock:
            # Evict the cheapest-ranked residents first, and never something
            # that outranks the request.
            candidates = [
                r
                for r in self._residents.values()
                if r.name != name and r.evictable and _RANK[r.priority] < _RANK[priority]
            ]
            candidates.sort(key=lambda r: (_RANK[r.priority], -r.bytes_used))

        evict: list[str] = []
        projected = free
        for resident in candidates:
            if projected >= need + self._headroom:
                break
            evict.append(resident.name)
            projected += resident.bytes_used

        if projected < need + self._headroom:
            return Decision(
                allowed=False,
                evict=evict,
                reason=(
                    f"nur {projected / 1e9:.1f} GB erreichbar, gebraucht werden "
                    f"{(need + self._headroom) / 1e9:.1f} GB"
                ),
            )
        return Decision(
            allowed=True, evict=evict, reason="evicted", free_after=projected - need
        )

    def apply(self, decision: Decision) -> list[str]:
        """Run the evictions a decision asked for. Returns what was unloaded."""
        done: list[str] = []
        for name in decision.evict:
            with self._lock:
                resident = self._residents.pop(name, None)
            if resident is None:
                continue
            if callable(resident.unload):
                try:
                    resident.unload()
                except Exception:
                    log.exception("could not unload %s", name)
                    continue
            log.info("evicted %s (%.1f GB)", name, resident.bytes_used / 1e9)
            done.append(name)
        return done
