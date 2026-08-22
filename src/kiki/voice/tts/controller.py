"""Queue and playback for one voice answer at a time.

Producer and consumer are separate asyncio tasks joined by a bounded queue:

    provider.synthesize() ──put──> Queue(maxsize=1) ──get──> sink.play()

The bound is the whole mechanism. With `maxsize=1`, one chunk is being played
while at most one more waits; the producer blocks on `put` until the player
takes one. No counter to keep in sync, and synthesis cannot run ahead and waste
GPU time on audio a cancellation is about to discard.

Only one generation is active. A new answer supersedes the old one, because a
reply to a superseded question must never keep talking.

Nothing here imports torch, CUDA, PipeWire, GTK or a network client.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from enum import StrEnum

from kiki.voice.tts.models import AudioChunk, TTSError, TTSGenerationResult, TTSRequest
from kiki.voice.tts.playback import AudioSink
from kiki.voice.tts.provider import TTSProvider

log = logging.getLogger(__name__)

# One chunk buffered ahead of the one playing.
DEFAULT_PREFETCH = 1


class PlaybackState(StrEnum):
    IDLE = "idle"
    SPEAKING = "speaking"
    CANCELLING = "cancelling"
    CLOSED = "closed"


class VoicePlaybackController:
    def __init__(
        self,
        provider: TTSProvider,
        sink: AudioSink,
        *,
        prefetch: int = DEFAULT_PREFETCH,
    ) -> None:
        if int(prefetch) < 1:
            # Not silently corrected: a caller asking for "no buffer" means
            # something the queue cannot express, and guessing hides the bug.
            raise ValueError("prefetch muss mindestens 1 sein")
        self._provider = provider
        self._sink = sink
        self._prefetch = int(prefetch)
        self._state = PlaybackState.IDLE
        self._current: TTSRequest | None = None
        self._task: asyncio.Task[TTSGenerationResult] | None = None
        self._cancelled: set[str] = set()
        self._lock = asyncio.Lock()

    # --- observation -------------------------------------------------------

    @property
    def state(self) -> PlaybackState:
        return self._state

    @property
    def current_request_id(self) -> str | None:
        return self._current.id if self._current else None

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    # --- driving -----------------------------------------------------------

    async def speak(self, request: TTSRequest) -> TTSGenerationResult:
        """Speak one answer, superseding whatever was running."""
        task = await self.submit(request)
        return await task

    async def submit(self, request: TTSRequest) -> asyncio.Task[TTSGenerationResult]:
        """Start speaking and return the task, without awaiting it."""
        if self._state is PlaybackState.CLOSED:
            raise TTSError("Controller ist beendet", code="closed")
        async with self._lock:
            await self._abort_current()
            self._cancelled.discard(request.id)
            self._current = request
            self._state = PlaybackState.SPEAKING
            self._task = asyncio.create_task(
                self._run(request), name=f"voice-{request.id}"
            )
            return self._task

    async def cancel(self, request_id: str) -> bool:
        """Stop one request by id. Idempotent; unknown ids return False."""
        self._cancelled.add(request_id)
        with contextlib.suppress(Exception):
            await self._provider.cancel(request_id)
        if self.current_request_id != request_id:
            return False
        async with self._lock:
            # Re-check: the request may have finished while we waited.
            if self.current_request_id != request_id:
                return False
            self._state = PlaybackState.CANCELLING
            await self._abort_current()
        return True

    async def interrupt(self) -> bool:
        """New user input: drop whatever is being said."""
        current = self.current_request_id
        if current is None:
            return False
        return await self.cancel(current)

    async def shutdown(self) -> None:
        """Stop playback, clear the queue, release the sink. Idempotent."""
        if self._state is PlaybackState.CLOSED:
            return
        async with self._lock:
            await self._abort_current()
            self._state = PlaybackState.CLOSED
        with contextlib.suppress(Exception):
            await self._sink.close()

    # --- internals ---------------------------------------------------------

    async def _abort_current(self) -> None:
        """Tear down the running generation. Must leave the controller usable."""
        task, self._task = self._task, None
        request, self._current = self._current, None
        if request is not None:
            self._cancelled.add(request.id)
            with contextlib.suppress(Exception):
                await self._provider.cancel(request.id)
        with contextlib.suppress(Exception):
            await self._sink.stop()
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self._state is not PlaybackState.CLOSED:
            self._state = PlaybackState.IDLE

    async def _run(self, request: TTSRequest) -> TTSGenerationResult:
        result = TTSGenerationResult(request_id=request.id)
        started = time.perf_counter()
        queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue(maxsize=self._prefetch)
        producer = asyncio.create_task(
            self._produce(request, queue, result), name=f"tts-{request.id}"
        )
        try:
            await self._consume(request, queue, result, started)
            await producer
        except asyncio.CancelledError:
            result.cancelled = True
            raise
        finally:
            if not producer.done():
                producer.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await producer
            _drain(queue)
            result.synthesis_seconds = time.perf_counter() - started
            if self._current is request:
                self._current = None
                if self._state is PlaybackState.SPEAKING:
                    self._state = PlaybackState.IDLE
        return result

    async def _produce(
        self,
        request: TTSRequest,
        queue: asyncio.Queue[AudioChunk | None],
        result: TTSGenerationResult,
    ) -> None:
        try:
            async for chunk in self._provider.synthesize(request):
                if request.id in self._cancelled:
                    break
                # Blocks while the buffer is full: that is the prefetch bound.
                await queue.put(chunk)
        except asyncio.CancelledError:
            raise
        except TTSError as exc:
            result.error = _safe_error(exc)
            log.warning("synthesis failed for %s: %s", request.id, result.error)
        except Exception as exc:
            result.error = _safe_error(exc)
            # No exc_info: a traceback would carry the unredacted message and
            # the prompt text through the frames.
            log.warning("synthesis crashed for %s: %s", request.id, result.error)
        finally:
            # The sentinel is what ends the consumer; without it a failed
            # producer would leave the queue waiting forever. The consumer
            # drains even after a cancel, so this put always finds room and
            # cannot block. A CancelledError here is deliberately *not*
            # swallowed: a task that eats its own cancellation reports success
            # it did not have.
            await queue.put(None)

    async def _consume(
        self,
        request: TTSRequest,
        queue: asyncio.Queue[AudioChunk | None],
        result: TTSGenerationResult,
        started: float,
    ) -> None:
        """Play until the sentinel arrives — and keep draining after a cancel.

        Leaving the loop early on the cancel flag looks tidier but deadlocks:
        the producer would still be holding a full queue and would block
        forever on its own sentinel, and `_run` would block on `await
        producer`. Draining costs one or two discarded chunks and removes the
        deadlock by construction instead of relying on an external
        `task.cancel()` arriving in time.
        """
        draining = False
        while True:
            chunk = await queue.get()
            if chunk is None:
                return
            if draining or request.id in self._cancelled:
                # Stop making sound immediately, but keep the queue moving.
                draining = True
                result.cancelled = True
                continue
            if chunk.request_id != request.id:
                # A chunk from a superseded answer must never be heard.
                continue
            try:
                await self._sink.play(chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # One bad chunk is skipped; the queue keeps moving.
                result.error = result.error or _safe_error(exc)
                log.warning("playback failed for chunk %s: %s", chunk.sequence, result.error)
                continue
            if result.time_to_first_audio is None:
                result.time_to_first_audio = time.perf_counter() - started
            result.chunks += 1
            result.audio_seconds += chunk.duration_s


# Provider exceptions quote whatever they choked on. That text reaches
# `TTSGenerationResult.error` and the log, so credentials, URLs and home paths
# are masked here rather than hoping every backend is careful.
_TOKENISH = re.compile(
    r"\b(?:sk-|ghp_|gho_|github_pat_|xox[baprs]-|AKIA|ASIA|eyJ[\w-]{8,})[\w\-./+=]{6,}"
    r"|\b(?:api[_-]?key|token|secret|passwor[dt]|bearer)s?\b\s*[:=]\s*\S+",
    re.IGNORECASE,
)
_URLISH = re.compile(r"\b(?:https?|ftp|ws)://\S+", re.IGNORECASE)
_HOMEISH = re.compile(r"(?:/home/[^\s:,)'\"]+|/Users/[^\s:,)'\"]+|/root/[^\s:,)'\"]*)")
MAX_ERROR_CHARS = 200


def _safe_error(exc: BaseException) -> str:
    """A short, redacted message fit for a result field and a log line."""
    text = str(exc) or exc.__class__.__name__
    text = _TOKENISH.sub("[entfernt]", text)
    text = _URLISH.sub("[url]", text)
    text = _HOMEISH.sub("[pfad]", text)
    text = " ".join(text.split())
    if len(text) > MAX_ERROR_CHARS:
        text = text[:MAX_ERROR_CHARS] + "…"
    return text


def _drain(queue: asyncio.Queue[AudioChunk | None]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
