"""VRAM budgeting for the KIKI harness. No GPU required."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "services" / "kiki-llm" / "vram.py"


def _load():
    spec = importlib.util.spec_from_file_location("kiki_vram", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules, so register first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vram = _load()
GB = 1_000_000_000


def _budget(total_gb=16, free_gb=None, headroom_gb=0.768):
    free = {"v": int((free_gb if free_gb is not None else total_gb) * GB)}
    b = vram.VramBudget(
        total_bytes=int(total_gb * GB),
        headroom_bytes=int(headroom_gb * GB),
        probe=lambda: free["v"],
    )
    return b, free


def _resident(name, gb, priority=vram.Priority.HIGH, evictable=True, unloaded=None):
    return vram.Resident(
        name=name,
        bytes_used=int(gb * GB),
        priority=priority,
        evictable=evictable,
        unload=(lambda: unloaded.append(name)) if unloaded is not None else None,
    )


def test_a_model_that_fits_is_simply_allowed() -> None:
    b, _ = _budget(total_gb=16, free_gb=11)
    d = b.plan(name="llm", need_bytes=2 * GB, priority=vram.Priority.HIGH)
    assert d.allowed is True
    assert d.evict == []


def test_a_model_larger_than_the_card_is_refused_outright() -> None:
    b, _ = _budget(total_gb=16, free_gb=16)
    d = b.plan(name="llm", need_bytes=20 * GB, priority=vram.Priority.EXCLUSIVE)
    assert d.allowed is False
    assert "die Karte hat" in d.reason


def test_headroom_is_not_spent() -> None:
    """Filling VRAM completely stutters the desktop before CUDA complains."""
    b, _ = _budget(total_gb=16, free_gb=2.0, headroom_gb=0.768)
    assert b.plan(name="llm", need_bytes=int(1.9 * GB), priority=vram.Priority.HIGH).allowed is False
    assert b.plan(name="llm", need_bytes=int(1.0 * GB), priority=vram.Priority.HIGH).allowed is True


def test_high_priority_evicts_low_but_not_the_other_way() -> None:
    b, _ = _budget(total_gb=16, free_gb=1)
    b.register(_resident("background", 4, vram.Priority.LOW))

    up = b.plan(name="llm", need_bytes=3 * GB, priority=vram.Priority.HIGH)
    assert up.allowed is True and up.evict == ["background"]

    b2, _ = _budget(total_gb=16, free_gb=1)
    b2.register(_resident("conversation", 4, vram.Priority.HIGH))
    down = b2.plan(name="bg", need_bytes=3 * GB, priority=vram.Priority.LOW)
    assert down.allowed is False
    assert down.evict == []


def test_exclusive_may_evict_high() -> None:
    b, _ = _budget(total_gb=16, free_gb=1)
    b.register(_resident("conversation", 5, vram.Priority.HIGH))
    d = b.plan(name="review", need_bytes=4 * GB, priority=vram.Priority.EXCLUSIVE)
    assert d.allowed is True
    assert d.evict == ["conversation"]


def test_equal_priority_never_evicts_a_peer() -> None:
    b, _ = _budget(total_gb=16, free_gb=1)
    b.register(_resident("tts", 5, vram.Priority.HIGH))
    d = b.plan(name="llm", need_bytes=4 * GB, priority=vram.Priority.HIGH)
    assert d.allowed is False


def test_a_pinned_resident_is_never_evicted() -> None:
    b, _ = _budget(total_gb=16, free_gb=1)
    b.register(_resident("tts", 6, vram.Priority.LOW, evictable=False))
    d = b.plan(name="llm", need_bytes=4 * GB, priority=vram.Priority.EXCLUSIVE)
    assert d.allowed is False
    assert d.evict == []


def test_only_as_many_are_evicted_as_needed() -> None:
    b, _ = _budget(total_gb=32, free_gb=1)
    for i in range(4):
        b.register(_resident(f"job{i}", 3, vram.Priority.LOW))
    d = b.plan(name="llm", need_bytes=5 * GB, priority=vram.Priority.HIGH)
    assert d.allowed is True
    assert len(d.evict) == 2, d.evict


def test_applying_a_decision_unloads_and_forgets() -> None:
    unloaded: list[str] = []
    b, _ = _budget(total_gb=16, free_gb=1)
    b.register(_resident("background", 5, vram.Priority.LOW, unloaded=unloaded))

    d = b.plan(name="llm", need_bytes=4 * GB, priority=vram.Priority.HIGH)
    assert b.apply(d) == ["background"]
    assert unloaded == ["background"]
    assert b.residents() == []


def test_a_failing_unload_does_not_crash_the_harness() -> None:
    b, _ = _budget(total_gb=16, free_gb=1)
    bad = vram.Resident(name="bad", bytes_used=5 * GB, priority=vram.Priority.LOW)
    bad.unload = lambda: (_ for _ in ()).throw(RuntimeError("CUDA hakt"))
    b.register(bad)
    d = b.plan(name="llm", need_bytes=4 * GB, priority=vram.Priority.HIGH)
    assert b.apply(d) == []


@pytest.mark.parametrize("need", [0, -1])
def test_loading_nothing_is_always_fine(need) -> None:
    b, _ = _budget(total_gb=16, free_gb=0)
    assert b.plan(name="x", need_bytes=need, priority=vram.Priority.LOW).allowed is True


def test_free_bytes_falls_back_to_accounting_without_a_probe() -> None:
    b = vram.VramBudget(total_bytes=16 * GB)
    assert b.free_bytes() == 16 * GB
    b.register(_resident("llm", 4))
    assert b.free_bytes() == 12 * GB
