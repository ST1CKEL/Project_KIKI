"""Where audio goes. No device library is imported here.

The controller talks to this protocol only, so tests run without a sound card
and a real PipeWire adapter can be added later without touching the queue.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from kiki.voice.tts.models import AudioChunk


@runtime_checkable
class AudioSink(Protocol):
    """One chunk at a time, played to completion."""

    async def play(self, chunk: AudioChunk) -> None:
        """Return once the chunk finished playing.

        Must return promptly after `stop()`; raising is allowed and must not
        leave the sink unusable for the next chunk.
        """
        ...

    async def stop(self) -> None:
        """Interrupt whatever is playing. Idempotent."""
        ...

    async def close(self) -> None:
        """Release the device. Idempotent."""
        ...


class FakeAudioSink:
    """Records what was played instead of making sound.

    `realtime=False` (the default) returns immediately, so queue and
    cancellation behaviour can be tested without waiting for audio.
    """

    def __init__(
        self,
        *,
        realtime: bool = False,
        fail_on_sequence: set[int] | None = None,
    ) -> None:
        self._realtime = realtime
        self._fail_on = set(fail_on_sequence or ())
        self.played: list[AudioChunk] = []
        self.stops = 0
        self.closed = False
        self.max_concurrent = 0
        self._active = 0
        self._interrupt = asyncio.Event()

    @property
    def played_sequences(self) -> list[int]:
        return [chunk.sequence for chunk in self.played]

    async def play(self, chunk: AudioChunk) -> None:
        if self.closed:
            raise RuntimeError("sink ist geschlossen")
        self._active += 1
        # Proves the controller never overlaps playback, which no amount of
        # queue arithmetic would show on its own.
        self.max_concurrent = max(self.max_concurrent, self._active)
        try:
            if chunk.sequence in self._fail_on:
                raise RuntimeError(f"Wiedergabe von Chunk {chunk.sequence} fehlgeschlagen")
            if self._realtime:
                self._interrupt.clear()
                try:
                    await asyncio.wait_for(
                        self._interrupt.wait(), timeout=chunk.duration_s
                    )
                    return  # interrupted by stop()
                except TimeoutError:
                    pass
            self.played.append(chunk)
        finally:
            self._active -= 1

    async def stop(self) -> None:
        self.stops += 1
        self._interrupt.set()

    async def close(self) -> None:
        self.closed = True
        self._interrupt.set()
