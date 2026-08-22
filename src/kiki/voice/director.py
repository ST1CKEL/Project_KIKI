"""Sentence-chunk TTS: synthesize on the asyncio thread, play on GLib."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from kiki.voice.tts_client import TtsError
from kiki.voice.tts_text import flush_buffer, speakable, split_ready

log = logging.getLogger(__name__)


class CancelHandle(Protocol):
    def cancel(self) -> None: ...


SubmitFn = Callable[..., CancelHandle | None]
SynthFn = Callable[[str, Path], Awaitable[Path]]
SynthJob = tuple[str, int, Path]
PlayJob = tuple[Path, int]


class SpeechDirector:
    """Queue complete sentences, speak them in order, support barge-in."""

    def __init__(
        self,
        *,
        synthesize: SynthFn,
        player: object,
        submit: SubmitFn,
        wav_dir: Path,
        on_speaking: Callable[[], None] | None = None,
        on_idle: Callable[[], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._synthesize = synthesize
        self._player = player
        self._submit = submit
        self._wav_dir = wav_dir
        self._on_speaking = on_speaking
        self._on_idle = on_idle
        self._on_error = on_error
        self._lock = threading.Lock()
        self._generation = 0
        self._buffer = ""
        self._spoke_anything = False
        self._stream_open = False
        self._synth_queue: deque[str] = deque()
        self._play_queue: deque[Path] = deque()
        self._synth_busy = False
        self._playing = False
        self._active = False
        self._warned_down = False
        self._synth_handle: CancelHandle | None = None
        self._active_synth_path: Path | None = None
        self._active_play_path: Path | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active or self._playing or self._synth_busy or bool(self._play_queue)

    def begin(self) -> None:
        self.stop()
        with self._lock:
            self._stream_open = True
            self._spoke_anything = False
            self._active = True

    def feed(self, delta: str) -> None:
        if not delta:
            return
        with self._lock:
            if not self._stream_open and not self._active:
                self._stream_open = True
                self._active = True
            self._buffer += delta
            # The opening chunk may end at a clause so KIKI starts talking
            # sooner; later chunks keep whole sentences for natural prosody.
            chunks, self._buffer = split_ready(
                self._buffer, first=not self._spoke_anything
            )
            if chunks:
                self._spoke_anything = True
            for chunk in chunks:
                self._synth_queue.append(chunk)
            synth = self._arm_synth_locked()
            play = self._take_play_locked()
        self._dispatch_synth(synth)
        self._start_play(play)

    def say(self, text: str) -> None:
        """Speak a complete utterance (prefs test, greet)."""
        spoken = speakable(text)
        if not spoken:
            return
        self.stop()
        with self._lock:
            self._stream_open = False
            self._active = True
            self._synth_queue.append(spoken)
            synth = self._arm_synth_locked()
            play = self._take_play_locked()
        self._dispatch_synth(synth)
        self._start_play(play)

    def flush(self) -> None:
        with self._lock:
            leftover = flush_buffer(self._buffer)
            self._buffer = ""
            self._stream_open = False
            if leftover:
                self._synth_queue.append(leftover)
            synth = self._arm_synth_locked()
            play = self._take_play_locked()
            idle = self._maybe_idle_locked()
        self._dispatch_synth(synth)
        self._start_play(play)
        if idle:
            self._emit_idle()

    def stop(self) -> None:
        with self._lock:
            was_active = (
                self._active
                or self._playing
                or self._synth_busy
                or bool(self._synth_queue)
                or bool(self._play_queue)
            )
            handle, stale_paths = self._reset_locked()
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                log.debug("synth cancellation failed", exc_info=True)
        try:
            self._player.stop()
        except Exception:
            log.debug("player stop failed", exc_info=True)
        for path in stale_paths:
            _unlink(path)
        if was_active:
            self._emit_idle()

    def _reset_locked(self) -> tuple[CancelHandle | None, list[Path]]:
        self._generation += 1
        self._buffer = ""
        self._spoke_anything = False
        self._stream_open = False
        self._synth_queue.clear()
        stale_paths = list(self._play_queue)
        self._play_queue.clear()
        if self._active_synth_path is not None:
            stale_paths.append(self._active_synth_path)
        if self._active_play_path is not None:
            stale_paths.append(self._active_play_path)
        handle = self._synth_handle
        self._synth_handle = None
        self._active_synth_path = None
        self._active_play_path = None
        self._synth_busy = False
        self._playing = False
        self._active = False
        return handle, list(dict.fromkeys(stale_paths))

    def _arm_synth_locked(self) -> SynthJob | None:
        if self._synth_busy or not self._synth_queue:
            return None
        text = self._synth_queue.popleft()
        generation = self._generation
        dest = self._wav_dir / f"{uuid4().hex}.wav"
        self._synth_busy = True
        self._active_synth_path = dest
        return text, generation, dest

    def _dispatch_synth(self, job: SynthJob | None) -> None:
        if job is None:
            return
        text, generation, dest = job

        async def _run() -> Path:
            try:
                return await self._synthesize(text, dest)
            except BaseException:
                # Cancellation may arrive after a synthesizer has created its
                # destination. Never leave that stale sentence in the cache.
                _unlink(dest)
                raise

        handle = self._submit(
            _run(),
            on_success=lambda path: self._on_wav(generation, path),
            on_error=lambda exc: self._on_synth_failed(generation, dest, exc),
        )
        if handle is None:
            return
        with self._lock:
            stale = generation != self._generation
            if not stale and self._synth_busy and self._active_synth_path == dest:
                self._synth_handle = handle
        if stale:
            # stop() may win the race before submit() returns its handle.
            handle.cancel()
            _unlink(dest)

    def _take_play_locked(self) -> PlayJob | None:
        if self._playing or not self._play_queue:
            return None
        path = self._play_queue.popleft()
        self._playing = True
        self._active_play_path = path
        return path, self._generation

    def _start_play(self, item: PlayJob | None) -> None:
        if item is None:
            return
        path, generation = item
        self._player.play(
            path,
            on_eos=lambda: self._on_eos(generation, path),
            on_error=lambda message: self._on_play_error(generation, path, message),
        )
        if self._on_speaking is not None:
            self._on_speaking()

    def _on_wav(self, generation: int, path: Path) -> None:
        with self._lock:
            if generation != self._generation:
                _unlink(path)
                return
            self._warned_down = False
            source_path = self._active_synth_path
            self._synth_handle = None
            self._active_synth_path = None
            self._synth_busy = False
            self._play_queue.append(path)
            synth = self._arm_synth_locked()
            play = self._take_play_locked()
            idle = self._maybe_idle_locked()
        if source_path is not None and source_path != path:
            _unlink(source_path)
        self._dispatch_synth(synth)
        self._start_play(play)
        if idle:
            self._emit_idle()

    def _on_synth_failed(self, generation: int, path: Path, exc: BaseException) -> None:
        down = isinstance(exc, TtsError) and "nicht erreichbar" in str(exc)
        with self._lock:
            if generation != self._generation:
                stale = True
                handle = None
                stale_paths: list[Path] = []
                synth = None
                play = None
                idle = False
                warn = False
            else:
                stale = False
                self._synth_handle = None
                self._active_synth_path = None
                self._synth_busy = False
                if down:
                    warn = not self._warned_down
                    self._warned_down = True
                    handle, stale_paths = self._reset_locked()
                    synth = None
                    play = None
                    idle = True
                else:
                    handle = None
                    stale_paths = []
                    log.warning("tts sentence skipped: %s", exc)
                    synth = self._arm_synth_locked()
                    play = self._take_play_locked()
                    idle = self._maybe_idle_locked()
                    warn = True
        _unlink(path)
        if stale:
            return
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                log.debug("synth cancellation failed", exc_info=True)
        for stale_path in stale_paths:
            _unlink(stale_path)
        if down:
            try:
                self._player.stop()
            except Exception:
                log.debug("player stop failed", exc_info=True)
        self._dispatch_synth(synth)
        self._start_play(play)
        if warn and self._on_error is not None:
            self._on_error(exc)
        if idle:
            self._emit_idle()

    def _on_eos(self, generation: int, path: Path) -> None:
        _unlink(path)
        with self._lock:
            if generation != self._generation:
                return
            self._playing = False
            self._active_play_path = None
            synth = self._arm_synth_locked()
            play = self._take_play_locked()
            idle = self._maybe_idle_locked()
        self._dispatch_synth(synth)
        self._start_play(play)
        if idle:
            self._emit_idle()

    def _on_play_error(self, generation: int, path: Path, message: str) -> None:
        _unlink(path)
        with self._lock:
            if generation != self._generation:
                return
            self._playing = False
            self._active_play_path = None
            play = self._take_play_locked()
            idle = self._maybe_idle_locked()
        if self._on_error is not None:
            self._on_error(RuntimeError(message))
        self._start_play(play)
        if idle:
            self._emit_idle()

    def _maybe_idle_locked(self) -> bool:
        if self._stream_open or self._synth_busy or self._playing:
            return False
        if self._synth_queue or self._play_queue:
            return False
        if not self._active:
            return False
        self._active = False
        return True

    def _emit_idle(self) -> None:
        if self._on_idle is not None:
            self._on_idle()


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.debug("could not remove %s", path)
