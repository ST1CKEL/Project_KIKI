"""Continuous batching: sequences must actually share forward passes."""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "services" / "kiki-llm" / "batching.py"


def _load():
    spec = importlib.util.spec_from_file_location("kiki_batching", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bt = _load()


class FakeModel:
    """Records the batch size of every forward pass."""

    def __init__(self, script: dict[str, list[str]] | None = None) -> None:
        self.script = script or {}
        self.batch_sizes: list[int] = []
        self.prefilled: list[str] = []
        self.released: list[str] = []
        self._pos: dict[str, int] = {}

    def prefill(self, sequences) -> None:
        for s in sequences:
            self.prefilled.append(s.id)
            self._pos[s.id] = 0

    def decode(self, sequences):
        self.batch_sizes.append(len(sequences))
        out = []
        for s in sequences:
            tokens = self.script.get(s.id, ["a", "b", "c"])
            i = self._pos.get(s.id, 0)
            if i >= len(tokens):
                out.append((s, "", True))
                continue
            self._pos[s.id] = i + 1
            out.append((s, tokens[i], i + 1 >= len(tokens)))
        return out

    def release(self, sequences) -> None:
        self.released.extend(s.id for s in sequences)


def _seq(sid, **kw):
    return bt.Sequence(id=sid, messages=[{"role": "user", "content": "x"}], **kw)


def _collect(sequence) -> str:
    out = []
    while True:
        item = sequence.out.get(timeout=5)
        if item is bt.DONE:
            return "".join(out)
        out.append(item)


# --- the point of the exercise ---------------------------------------------


def test_two_sequences_share_one_forward_pass() -> None:
    """The whole reason for the harness: not one request after another."""
    model = FakeModel({"a": ["1", "2", "3"], "b": ["x", "y", "z"]})
    sched = bt.BatchScheduler(model, max_batch=4)
    sa, sb = sched.submit(_seq("a")), sched.submit(_seq("b"))

    while sa.state is not bt.State.FINISHED or sb.state is not bt.State.FINISHED:
        if sched.step_once() == 0:
            break

    assert _collect(sa) == "123"
    assert _collect(sb) == "xyz"
    # Three steps for six tokens: they were decoded together, not in turn.
    assert model.batch_sizes[:3] == [2, 2, 2]
    assert sched.steps == 3
    assert sched.decoded_tokens == 6


def test_a_sequence_finishing_early_leaves_the_batch() -> None:
    model = FakeModel({"short": ["1"], "long": ["a", "b", "c", "d"]})
    sched = bt.BatchScheduler(model, max_batch=4)
    short, long_ = sched.submit(_seq("short")), sched.submit(_seq("long"))

    for _ in range(10):
        if sched.step_once() == 0:
            break

    assert _collect(short) == "1"
    assert _collect(long_) == "abcd"
    # First step batched both, later steps carried only the survivor.
    assert model.batch_sizes[0] == 2
    assert model.batch_sizes[-1] == 1
    assert "short" in model.released


def test_a_late_request_joins_the_running_batch() -> None:
    model = FakeModel({"first": ["a", "b", "c", "d"], "late": ["x", "y"]})
    sched = bt.BatchScheduler(model, max_batch=4)
    first = sched.submit(_seq("first"))

    sched.step_once()
    assert model.batch_sizes == [1]

    late = sched.submit(_seq("late"))
    for _ in range(10):
        if sched.step_once() == 0:
            break

    assert _collect(first) == "abcd"
    assert _collect(late) == "xy"
    assert 2 in model.batch_sizes, model.batch_sizes


def test_the_batch_never_exceeds_its_limit() -> None:
    model = FakeModel()
    sched = bt.BatchScheduler(model, max_batch=2)
    for i in range(5):
        sched.submit(_seq(f"s{i}"))
    for _ in range(30):
        if sched.step_once() == 0:
            break
    assert max(model.batch_sizes) <= 2


# --- priorities -------------------------------------------------------------


def test_exclusive_waits_for_the_batch_to_drain() -> None:
    """`coding_review` claims the runtime; it must not share a pass."""
    model = FakeModel({"chat": ["a", "b"], "review": ["r"]})
    sched = bt.BatchScheduler(model, max_batch=4)
    sched.submit(_seq("chat"))
    sched.step_once()

    sched.submit(_seq("review", priority="exclusive"))
    sched.step_once()
    assert "review" not in model.prefilled  # chat still running

    for _ in range(10):
        if sched.step_once() == 0:
            break
    assert "review" in model.prefilled
    assert all(size == 1 for size in model.batch_sizes)


def test_high_priority_is_admitted_before_low() -> None:
    model = FakeModel()
    sched = bt.BatchScheduler(model, max_batch=1)
    sched.submit(_seq("background", priority="low"))
    sched.submit(_seq("conversation", priority="high"))
    sched.step_once()
    assert model.prefilled[0] == "conversation"


# --- failure paths ----------------------------------------------------------


def test_a_cancelled_sequence_is_dropped_and_its_cache_freed() -> None:
    model = FakeModel({"a": ["1"] * 50})
    sched = bt.BatchScheduler(model, max_batch=4)
    s = sched.submit(_seq("a"))
    sched.step_once()
    sched.cancel(s)
    sched.step_once()
    assert "a" in model.released
    assert s.state is bt.State.FINISHED


def test_a_failing_prefill_finishes_the_sequence_instead_of_hanging() -> None:
    class Broken(FakeModel):
        def prefill(self, sequences):
            raise RuntimeError("CUDA OOM")

    sched = bt.BatchScheduler(Broken(), max_batch=2)
    s = sched.submit(_seq("a"))
    sched.step_once()
    assert s.state is bt.State.FINISHED
    assert _collect(s) == ""


def test_a_failing_decode_retires_the_whole_batch() -> None:
    class Broken(FakeModel):
        def decode(self, sequences):
            raise RuntimeError("kaputt")

    sched = bt.BatchScheduler(Broken(), max_batch=2)
    a, b = sched.submit(_seq("a")), sched.submit(_seq("b"))
    sched.step_once()
    assert a.state is bt.State.FINISHED and b.state is bt.State.FINISHED


def test_max_new_tokens_stops_a_runaway_sequence() -> None:
    model = FakeModel({"a": ["x"] * 1000})
    sched = bt.BatchScheduler(model, max_batch=2)
    s = sched.submit(_seq("a", max_new_tokens=5))
    for _ in range(50):
        if sched.step_once() == 0:
            break
    assert s.produced == 5


def test_an_idle_scheduler_does_nothing() -> None:
    model = FakeModel()
    sched = bt.BatchScheduler(model, max_batch=2)
    assert sched.step_once() == 0
    assert model.batch_sizes == []


# --- the background thread --------------------------------------------------


def test_the_loop_serves_concurrent_submitters() -> None:
    model = FakeModel({f"s{i}": ["a", "b"] for i in range(4)})
    sched = bt.BatchScheduler(model, max_batch=4, idle_sleep_s=0.001)
    sched.start()
    try:
        results: dict[str, str] = {}

        def run(i):
            s = sched.submit(_seq(f"s{i}"))
            results[f"s{i}"] = _collect(s)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
        [t.start() for t in threads]
        [t.join(timeout=10) for t in threads]
        assert results == {f"s{i}": "ab" for i in range(4)}
        assert max(model.batch_sizes) > 1, "nothing was actually batched"
    finally:
        sched.stop()


@pytest.mark.parametrize("size", [1, 2, 4])
def test_stats_report_what_happened(size) -> None:
    model = FakeModel({f"s{i}": ["a"] for i in range(size)})
    sched = bt.BatchScheduler(model, max_batch=size)
    for i in range(size):
        sched.submit(_seq(f"s{i}"))
    sched.step_once()
    stats = sched.stats()
    assert stats["decoded_tokens"] == size
    assert stats["steps"] == 1


# --- cache surgery ----------------------------------------------------------


def _torch():
    return pytest.importorskip("torch")


def _load_torch_batch():
    spec = importlib.util.spec_from_file_location(
        "kiki_torch_batch", ROOT / "services" / "kiki-llm" / "torch_batch.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cache(torch, rows: int, length: int, layers: int = 2, fill: float = 1.0):
    from transformers import DynamicCache

    cache = DynamicCache()
    for layer in range(layers):
        shape = (rows, 2, length, 4)
        cache.update(torch.full(shape, fill), torch.full(shape, fill), layer)
    return cache


def test_selecting_rows_keeps_the_right_sequences() -> None:
    torch = _torch()
    pytest.importorskip("transformers")
    tb = _load_torch_batch()
    cache = _cache(torch, rows=3, length=5)
    for layer in range(2):
        cache[layer][0][1] = 7.0  # mark row 1

    kept = tb._select(cache, [1, 2], torch)
    assert kept[0][0].shape[0] == 2
    assert float(kept[0][0][0].flatten()[0]) == 7.0


def test_merging_left_pads_the_shorter_sequence() -> None:
    """Generation reads the last position, so padding must sit in front."""
    torch = _torch()
    pytest.importorskip("transformers")
    tb = _load_torch_batch()
    short = _cache(torch, rows=1, length=2, fill=3.0)
    long = _cache(torch, rows=1, length=5, fill=9.0)

    merged = tb._merge_caches([short, long], torch)
    k, _v = merged[0]
    assert k.shape[0] == 2 and k.shape[2] == 5
    # Row 0 is the short one: three zeros in front, then its own values.
    assert float(k[0, 0, 0, 0]) == 0.0
    assert float(k[0, 0, -1, 0]) == 3.0
    assert float(k[1, 0, 0, 0]) == 9.0


def test_merging_a_single_cache_is_a_no_op() -> None:
    torch = _torch()
    pytest.importorskip("transformers")
    tb = _load_torch_batch()
    only = _cache(torch, rows=1, length=3)
    assert tb._merge_caches([only], torch) is only


def test_multibyte_characters_survive_token_by_token_decoding() -> None:
    """Qwen uses byte-level BPE: one emoji is several tokens.

    Decoding each token on its own produced U+FFFD, and those replacement
    characters reached the TTS service — which, being a Chinese-first model,
    turned the garbage into Chinese speech.
    """
    pytest.importorskip("transformers")
    tb = _load_torch_batch()

    class FakeTok:
        """Byte-level: decoding a partial character yields the replacement char."""

        def __init__(self, text: str) -> None:
            self.raw = text.encode("utf-8")

        def decode(self, ids, skip_special_tokens=True):
            return bytes(self.raw[i] for i in ids).decode("utf-8", errors="replace")

    text = "Hi 😊!"
    tok = FakeTok(text)
    state = tb.SeqState()
    # One byte per "token" — the worst case for a multi-byte character.
    got = "".join(state.take(tok, i) for i in range(len(tok.raw)))

    assert got == text
    assert "�" not in got


def test_a_half_finished_character_is_held_back_not_emitted() -> None:
    pytest.importorskip("transformers")
    tb = _load_torch_batch()

    class FakeTok:
        def __init__(self) -> None:
            self.raw = "😊".encode()

        def decode(self, ids, skip_special_tokens=True):
            return bytes(self.raw[i] for i in ids).decode("utf-8", errors="replace")

    tok = FakeTok()
    state = tb.SeqState()
    # The first three bytes cannot complete the character.
    assert state.take(tok, 0) == ""
    assert state.take(tok, 1) == ""
    assert state.take(tok, 2) == ""
    assert state.take(tok, 3) == "😊"
