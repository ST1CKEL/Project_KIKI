"""Sentence-chunk TTS: synthesize on the asyncio thread, play on GLib."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from kiki.voice.tts.models import (
    AudioStartedEvent,
    TTSError,
    TTSGenerationResult,
    TTSRequest,
)
from kiki.voice.tts.policy import VoiceResponsePolicy
from kiki.voice.tts_client import TtsError
from kiki.voice.tts_text import flush_buffer, speakable, split_ready

log = logging.getLogger(__name__)


class CancelHandle(Protocol):
    def cancel(self) -> None: ...


class PlayerLike(Protocol):
    """What the director needs from a player. `PipeWirePlayer` satisfies it.

    Typed as a Protocol so an alternative sink can be injected without the
    director knowing which one it got.
    """

    def play(
        self,
        path: Path,
        *,
        on_eos: Callable[[], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None: ...

    def stop(self) -> None: ...


class ControllerLike(Protocol):
    """The slice of `VoicePlaybackController` the director drives."""

    async def speak(self, request: TTSRequest) -> TTSGenerationResult: ...

    async def interrupt(self) -> bool: ...


PolicySource = VoiceResponsePolicy | Callable[[], VoiceResponsePolicy]


def _as_policy_source(
    policy: PolicySource | None,
) -> Callable[[], VoiceResponsePolicy]:
    """Accept a policy or a way to get one; always return a way to get one."""
    if policy is None:
        settled = VoiceResponsePolicy()
        return lambda: settled
    if callable(policy):
        return policy
    return lambda: policy


SubmitFn = Callable[..., CancelHandle | None]
SynthFn = Callable[[str, Path], Awaitable[Path]]
SynthJob = tuple[str, int, Path]
PlayJob = tuple[Path, int]
RouteJob = tuple[str, int]
Jobs = tuple[SynthJob | None, PlayJob | None, RouteJob | None]
NO_JOBS: Jobs = (None, None, None)


def service_is_down(exc: BaseException) -> bool:
    """True when the whole TTS route is gone, not just one sentence.

    A single failed sentence is skipped; an unreachable service resets the
    queue and warns once. The old form of this test was a message comparison
    inlined in `_on_synth_failed`; it stays first so the existing route behaves
    byte-for-byte as before, and the code-based branch covers the adapter, which
    nothing feeds into the director yet.
    """
    if isinstance(exc, TTSError):
        # not_ready belongs here too: a provider that never loaded fails every
        # sentence, and warning once per answer beats warning once per clause.
        return exc.code in {"unreachable", "timeout", "load", "not_ready"}
    return isinstance(exc, TtsError) and "nicht erreichbar" in str(exc)


class SpeechDirector:
    """Queue complete sentences, speak them in order, support barge-in.

    **What the signals mean, per route.** `on_idle` is the same on both: the
    utterance finished, was cancelled, or failed. `on_speaking` is not:

    * file route — the finished WAV is being handed to the player, so audio is
      either running or about to run within milliseconds;
    * controller route — the request was *accepted and handed over*. Synthesis
      has not started. Nothing audible exists yet, and on a cold GPU service the
      first sample can be seconds away.

    `on_audio_started` closes that gap: it fires once per utterance, when the
    first chunk with actual samples is going to the speakers. Both routes emit
    it, so a listener can key the "KIKI is audible" state on this one signal
    without asking which route it is on:

    * file route — at the same instant as `on_speaking`, since the WAV is
      complete by then and playback starts with it;
    * controller route — when the controller hands the first chunk to the sink,
      which on the WAV-based service path is after synthesis finished.
    """

    def __init__(
        self,
        *,
        synthesize: SynthFn,
        player: PlayerLike,
        submit: SubmitFn,
        wav_dir: Path,
        on_speaking: Callable[[], None] | None = None,
        on_audio_started: Callable[[], None] | None = None,
        on_idle: Callable[[], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        service_down: Callable[[BaseException], bool] | None = None,
        controller: ControllerLike | None = None,
        use_controller_route: bool = False,
        policy: PolicySource | None = None,
    ) -> None:
        self._synthesize = synthesize
        # What may be said out loud. The WAV route used to have none, so a
        # secret or a home path in an answer was read aloud.
        #
        # Held as a source rather than a value: a tightened setting must reach
        # an answer that is already being spoken, the same way the panic switch
        # reaches a tool call that is already under way. Loosening it live is
        # the same mechanism; the point is that the switch is never stale.
        self._policy_source = _as_policy_source(policy)
        self._player = player
        self._submit = submit
        self._wav_dir = wav_dir
        self._on_speaking = on_speaking
        self._on_audio_started = on_audio_started
        self._on_idle = on_idle
        self._on_error = on_error
        self._service_down = service_down or service_is_down
        self._controller = controller
        # Both halves must be present. A flag without a controller would turn
        # speech off rather than switching the route.
        self._use_controller = bool(use_controller_route and controller is not None)
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
        # The controller route keeps its own busy flag and handle. It shares the
        # text queue and nothing else — no WAV queue, no _synth_busy.
        self._route_busy = False
        self._route_handle: CancelHandle | None = None
        # Which utterance the controller is on, and whether its audio was
        # already announced. A late event for any other id is dropped.
        self._route_request_id: str | None = None
        self._route_announced = False

    @property
    def _policy(self) -> VoiceResponsePolicy:
        """The rule as it stands right now, not as it stood at startup.

        Cheap on purpose: the source is an attribute read, so resolving it
        inside the queue lock costs nothing and cannot deadlock.
        """
        return self._policy_source()

    def audio_started(self, event: AudioStartedEvent) -> None:
        """The controller reports that the first chunk is going to the speakers.

        Arrives on the asyncio thread — the caller is responsible for marshalling
        onwards to GTK. Kept cheap and lock-only for exactly that reason.

        Events for a superseded or already finished utterance are dropped: after
        `stop()` the id no longer matches, so an answer the user interrupted can
        never announce itself afterwards.
        """
        with self._lock:
            if self._route_request_id != event.request_id or self._route_announced:
                return
            self._route_announced = True
        self._emit_audio_started()

    def disable_controller_route(self) -> None:
        """Fall back to the file route for good.

        Called when the opt-in route proved unusable — a provider that never
        loaded fails every sentence. One-way on purpose: a failure must never
        silently re-enable the flag later.
        """
        with self._lock:
            self._use_controller = False

    @property
    def uses_controller_route(self) -> bool:
        return self._use_controller

    @property
    def active(self) -> bool:
        with self._lock:
            return (
                self._active
                or self._playing
                or self._synth_busy
                or self._route_busy
                or bool(self._play_queue)
            )

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
                self._enqueue_locked(chunk)
            jobs = self._arm_locked()
        self._dispatch(jobs)

    def _enqueue_locked(self, text: str) -> None:
        """The one way into the speech queue.

        Redaction sits here rather than at the four call sites so a new caller
        cannot forget it -- the same reason the audit sanitises in its sink.
        Caller holds `self._lock`.
        """
        spoken = self._policy.redact_chunk(text)
        if spoken:
            self._synth_queue.append(spoken)

    def say(self, text: str) -> None:
        """Speak a complete utterance (prefs test, greet)."""
        spoken = speakable(text)
        if not spoken:
            return
        self.stop()
        with self._lock:
            self._stream_open = False
            self._active = True
            self._enqueue_locked(spoken)
            jobs = self._arm_locked()
        self._dispatch(jobs)

    def flush(self) -> None:
        with self._lock:
            leftover = flush_buffer(self._buffer)
            self._buffer = ""
            self._stream_open = False
            if leftover:
                self._enqueue_locked(leftover)
            jobs = self._arm_locked()
            idle = self._maybe_idle_locked()
        self._dispatch(jobs)
        if idle:
            self._emit_idle()

    def stop(self) -> None:
        """Barge-in. Synchronous for the caller, in two guaranteed halves.

        Synchronously, before this returns: the generation is bumped and both
        queues are cleared, so every callback still in flight is stale and no
        further sentence can start. That is the part the GTK thread depends on.

        Asynchronously, on the bridge: the running provider and sink work is
        cancelled. Actual silence needs the asyncio thread, so it cannot be
        promised here — but nothing new is ever started after this returns.
        """
        with self._lock:
            was_active = (
                self._active
                or self._playing
                or self._synth_busy
                or self._route_busy
                or bool(self._synth_queue)
                or bool(self._play_queue)
            )
            handles, stale_paths = self._reset_locked()
        for handle in handles:
            try:
                handle.cancel()
            except Exception:
                log.debug("synth cancellation failed", exc_info=True)
        if was_active:
            # begin() and say() both call stop() first. Handing the bridge an
            # interrupt for an idle controller would cost a round-trip per
            # answer and buy nothing.
            self._cancel_route()
        try:
            self._player.stop()
        except Exception:
            log.debug("player stop failed", exc_info=True)
        for path in stale_paths:
            _unlink(path)
        if was_active:
            self._emit_idle()

    def _reset_locked(self) -> tuple[list[CancelHandle], list[Path]]:
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
        handles = [
            handle
            for handle in (self._synth_handle, self._route_handle)
            if handle is not None
        ]
        self._synth_handle = None
        self._route_handle = None
        self._route_request_id = None
        self._route_announced = False
        self._active_synth_path = None
        self._active_play_path = None
        self._synth_busy = False
        self._route_busy = False
        self._playing = False
        self._active = False
        return handles, list(dict.fromkeys(stale_paths))

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

    def _arm_locked(self) -> Jobs:
        """Pick the next piece of work for whichever route is active.

        Exactly one of the two routes is ever armed, so the controller path can
        never end up feeding the WAV queue or vice versa.
        """
        if self._use_controller:
            return None, None, self._arm_route_locked()
        return self._arm_synth_locked(), self._take_play_locked(), None

    def _dispatch(self, jobs: Jobs) -> None:
        synth, play, route = jobs
        self._dispatch_synth(synth)
        self._start_play(play)
        self._dispatch_route(route)

    def _arm_route_locked(self) -> RouteJob | None:
        """One sentence at a time.

        `VoicePlaybackController.submit()` supersedes whatever is running, so
        handing it the next sentence early would cut the current one off
        mid-word. The queue that keeps order is the text queue, not a second
        audio queue.
        """
        if self._route_busy or not self._synth_queue:
            return None
        self._route_busy = True
        return self._synth_queue.popleft(), self._generation

    def _dispatch_route(self, job: RouteJob | None) -> None:
        if job is None:
            return
        text, generation = job
        try:
            request = TTSRequest(text=text)
        except ValueError:
            # Nothing speakable survived the chunker. Treat it as a finished
            # utterance so the queue keeps moving.
            self._on_route_done(generation, None)
            return
        with self._lock:
            if generation == self._generation:
                self._route_request_id = request.id
                self._route_announced = False
        # Announced before the hand-over, so the order the UI sees does not
        # depend on whether submit() happens to run the coroutine inline.
        if self._on_speaking is not None:
            self._on_speaking()
        coro = self._controller.speak(request)  # type: ignore[union-attr]
        try:
            handle = self._submit(
                coro,
                on_success=lambda result: self._on_route_done(generation, result),
                on_error=lambda exc: self._on_route_failed(generation, exc),
            )
        except Exception as exc:
            coro.close()
            self._on_route_failed(generation, exc)
            return
        if handle is None:
            return
        with self._lock:
            stale = generation != self._generation
            if not stale and self._route_busy:
                self._route_handle = handle
        if stale:
            # stop() may win the race before submit() returns its handle.
            handle.cancel()

    def _on_route_done(self, generation: int, result: TTSGenerationResult | None) -> None:
        error = getattr(result, "error", "") or ""
        if error:
            self._on_route_failed(generation, TTSError(error, code="route"))
            return
        cancelled = bool(getattr(result, "cancelled", False))
        with self._lock:
            if generation != self._generation:
                return
            self._warned_down = False
            self._route_handle = None
            self._route_busy = False
            self._route_request_id = None
            # A superseded utterance must not pull the next sentence forward:
            # whatever should follow was already cleared by stop().
            jobs = NO_JOBS if cancelled else self._arm_locked()
            idle = self._maybe_idle_locked()
        self._dispatch(jobs)
        if idle:
            self._emit_idle()

    def _on_route_failed(self, generation: int, exc: BaseException) -> None:
        down = self._service_down(exc)
        with self._lock:
            if generation != self._generation:
                return
            self._route_handle = None
            self._route_busy = False
            self._route_request_id = None
            if down:
                warn = not self._warned_down
                self._warned_down = True
                handles, stale_paths = self._reset_locked()
                jobs = NO_JOBS
                idle = True
            else:
                warn = True
                handles, stale_paths = [], []
                log.warning("tts sentence skipped: %s", exc)
                jobs = self._arm_locked()
                idle = self._maybe_idle_locked()
        for handle in handles:
            try:
                handle.cancel()
            except Exception:
                log.debug("route cancellation failed", exc_info=True)
        for path in stale_paths:
            _unlink(path)
        if down:
            self._cancel_route()
        self._dispatch(jobs)
        if warn and self._on_error is not None:
            self._on_error(exc)
        if idle:
            self._emit_idle()

    def _cancel_route(self) -> None:
        """Hand the actual silencing to the bridge and return at once.

        The GTK thread must never wait for the sink, so this only queues the
        interrupt. Nothing new can start regardless: stop() already bumped the
        generation before calling this.
        """
        if not self._use_controller or self._controller is None:
            return
        coro = self._controller.interrupt()
        try:
            self._submit(
                coro,
                on_error=lambda exc: log.debug("voice interrupt failed: %s", exc),
            )
        except Exception:
            coro.close()
            log.debug("could not hand the interrupt to the bridge", exc_info=True)

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
        # Deliberate: the file route emits both signals at once. The WAV is
        # finished by the time it reaches the player, so "accepted" and "audible"
        # really are the same instant here — and a listener can then key the
        # audible state on one signal for both routes.
        self._emit_audio_started()

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
        down = self._service_down(exc)
        with self._lock:
            if generation != self._generation:
                stale = True
                handles: list[CancelHandle] = []
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
                    handles, stale_paths = self._reset_locked()
                    synth = None
                    play = None
                    idle = True
                else:
                    handles = []
                    stale_paths = []
                    log.warning("tts sentence skipped: %s", exc)
                    synth = self._arm_synth_locked()
                    play = self._take_play_locked()
                    idle = self._maybe_idle_locked()
                    warn = True
        _unlink(path)
        if stale:
            return
        for handle in handles:
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
        if self._stream_open or self._synth_busy or self._playing or self._route_busy:
            return False
        if self._synth_queue or self._play_queue:
            return False
        if not self._active:
            return False
        self._active = False
        return True

    def _emit_audio_started(self) -> None:
        if self._on_audio_started is not None:
            self._on_audio_started()

    def _emit_idle(self) -> None:
        if self._on_idle is not None:
            self._on_idle()


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.debug("could not remove %s", path)
