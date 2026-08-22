"""Continuous batching: one forward pass serves every active sequence.

Without this, "slots" only means admission control — requests are let in and
then run one after another, which is exactly what Ollama does and what the
harness was meant to improve on.

The scheduler is deliberately free of tensors. It decides *what* happens each
step — who joins, who is decoded together, who retires — and delegates the model
work to a `BatchedModel`. That split is what makes the awkward parts (a sequence
finishing mid-batch, a new request arriving while others decode, a client
disconnecting) testable without a GPU.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

log = logging.getLogger(__name__)

# End marker pushed onto a sequence's queue when it is done.
DONE = object()


class State(StrEnum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    FINISHED = "finished"


@dataclass
class Sequence:
    """One in-flight generation."""

    id: str
    messages: list[dict]
    tools: list | None = None
    temperature: float = 0.7
    max_new_tokens: int = 512
    suppress_reasoning: bool = True
    priority: str = "high"

    state: State = State.WAITING
    produced: int = 0
    cancelled: bool = False
    out: queue.Queue = field(default_factory=queue.Queue)
    # Opaque handle the model implementation uses for this sequence's KV cache.
    handle: object = None

    def emit(self, text: str) -> None:
        if text and not self.cancelled:
            self.out.put(text)

    def finish(self) -> None:
        if self.state is not State.FINISHED:
            self.state = State.FINISHED
            self.out.put(DONE)


class BatchedModel(Protocol):
    """What the scheduler needs from the model. Implemented with real tensors."""

    def prefill(self, sequences: list[Sequence]) -> None:
        """Run the prompts and set up each sequence's cache."""
        ...

    def decode(self, sequences: list[Sequence]) -> list[tuple[Sequence, str, bool]]:
        """One batched step. Returns (sequence, new text, finished) per entry."""
        ...

    def release(self, sequences: list[Sequence]) -> None:
        """Drop the caches of sequences that will not continue."""
        ...


class BatchScheduler:
    def __init__(
        self,
        model: BatchedModel,
        *,
        max_batch: int = 4,
        admit_window_s: float = 0.02,
        idle_sleep_s: float = 0.01,
    ) -> None:
        self._model = model
        self._max_batch = max(1, int(max_batch))
        self._admit_window = max(0.0, float(admit_window_s))
        self._idle = max(0.001, float(idle_sleep_s))
        self._pending: list[Sequence] = []
        self._active: list[Sequence] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Observability, so a speedup claim can be checked rather than asserted.
        self.steps = 0
        self.decoded_tokens = 0

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="kiki-batch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)

    # --- submission --------------------------------------------------------

    def submit(self, sequence: Sequence) -> Sequence:
        with self._lock:
            self._pending.append(sequence)
        self._wake.set()
        return sequence

    def cancel(self, sequence: Sequence) -> None:
        sequence.cancelled = True
        self._wake.set()

    def stats(self) -> dict:
        with self._lock:
            return {
                "active": len(self._active),
                "pending": len(self._pending),
                "steps": self.steps,
                "decoded_tokens": self.decoded_tokens,
            }

    # --- the loop ----------------------------------------------------------

    def _admit(self) -> None:
        """Move waiting requests into the batch, highest priority first."""
        with self._lock:
            room = self._max_batch - len(self._active)
            if room <= 0 or not self._pending:
                return
            forming = not self._active
            pending = len(self._pending)

        if forming and pending < self._max_batch and self._admit_window > 0:
            # Starting a fresh batch: give near-simultaneous requests a moment
            # to arrive. Without this the loop admits the first one and begins
            # decoding before the second is even submitted, so nothing ever
            # shares a forward pass. A lone request pays this window once —
            # milliseconds against a generation measured in seconds.
            deadline = time.monotonic() + self._admit_window
            while time.monotonic() < deadline:
                with self._lock:
                    if len(self._pending) >= self._max_batch:
                        break
                time.sleep(0.002)

        with self._lock:
            room = self._max_batch - len(self._active)
            if room <= 0 or not self._pending:
                return
            order = {"exclusive": 0, "high": 1, "low": 2}
            self._pending.sort(key=lambda s: order.get(s.priority, 1))
            # An exclusive request waits for the batch to drain rather than
            # sharing a forward pass — that is what "exclusive" is for.
            if self._pending[0].priority == "exclusive" and self._active:
                return
            take = self._pending[:room]
            self._pending = self._pending[room:]
        if not take:
            return
        for sequence in take:
            sequence.state = State.PREFILL
        try:
            self._model.prefill(take)
        except Exception:
            log.exception("prefill failed")
            for sequence in take:
                sequence.finish()
            return
        for sequence in take:
            sequence.state = State.DECODING
        with self._lock:
            self._active.extend(take)

    def _retire(self, sequences: list[Sequence]) -> None:
        if not sequences:
            return
        try:
            self._model.release(sequences)
        except Exception:
            log.exception("releasing caches failed")
        with self._lock:
            self._active = [s for s in self._active if s not in sequences]
        for sequence in sequences:
            sequence.finish()

    def step_once(self) -> int:
        """One scheduling step. Returns how many sequences were decoded."""
        self._admit()
        with self._lock:
            batch = list(self._active)
        if not batch:
            return 0

        gone = [s for s in batch if s.cancelled]
        if gone:
            self._retire(gone)
            batch = [s for s in batch if not s.cancelled]
        if not batch:
            return 0

        try:
            results = self._model.decode(batch)
        except Exception:
            log.exception("decode step failed")
            self._retire(batch)
            return 0

        self.steps += 1
        finished: list[Sequence] = []
        for sequence, text, done in results:
            if text:
                sequence.emit(text)
                sequence.produced += 1
                self.decoded_tokens += 1
            if done or sequence.produced >= sequence.max_new_tokens:
                finished.append(sequence)
        self._retire(finished)
        return len(batch)

    def _run(self) -> None:
        while not self._stop.is_set():
            worked = self.step_once()
            if worked:
                continue
            with self._lock:
                idle = not self._active and not self._pending
            if idle:
                # Wait for work rather than spinning on an empty GPU.
                self._wake.wait(timeout=0.5)
                self._wake.clear()
            else:
                time.sleep(self._idle)
