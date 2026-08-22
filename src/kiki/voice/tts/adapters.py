"""Adapters from the existing TTS route onto the neutral contracts.

Two impedance mismatches are bridged here, and nowhere else:

* The service answers one HTTP request with **one complete WAV file**, while
  `TTSProvider` promises a stream of `AudioChunk`. `ServiceTTSProvider` decodes
  the finished file and slices it on frame boundaries, so `capabilities()`
  reports `streaming=False` — the chunks arrive quickly, but none of them exists
  before the whole utterance was synthesised.
* `PipeWirePlayer` plays **files**, while `AudioSink` receives PCM. Every chunk
  therefore becomes a short temporary WAV. That is the honest cost of reusing
  the proven player; a later sink can feed `appsrc` or `pw-cat` directly and
  drop the round-trip through the filesystem.

Deliberately **not** re-exported from `kiki.voice.tts`: that package is imported
by the UI and must stay free of `httpx` and of the player module.

Nothing here changes the production route. `SpeechDirector` keeps using
`synthesize_wav` and `PipeWirePlayer` directly until a later milestone switches
it over.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from uuid import uuid4

from kiki.voice.tts.models import (
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_SAMPLE_RATE,
    AudioChunk,
    TTSError,
    TTSHealth,
    TTSProviderCapabilities,
    TTSProviderStatus,
    TTSRequest,
)
from kiki.voice.tts_client import (
    CUSTOM_VOICE_LANGUAGES,
    CUSTOM_VOICE_SPEAKERS,
    DEFAULT_LANGUAGE,
    DEFAULT_SPEAKER,
    DEFAULT_TTS_URL,
    TtsError,
    TtsHealth,
    synthesize_wav,
    tts_health,
)
from kiki.voice.tts_player import PipeWirePlayer

log = logging.getLogger(__name__)

# Half a second is long enough that the per-chunk WAV round-trip stays
# negligible and short enough that a barge-in is not audibly late.
DEFAULT_CHUNK_SECONDS = 0.5
# Cancellations for ids that never reach synthesize() would otherwise pile up.
MAX_TRACKED_CANCELS = 256

_URLISH = re.compile(r"\b(?:https?|ftp|ws)://\S+", re.IGNORECASE)


def _scrub(text: str) -> str:
    """Drop full URLs from a backend message.

    The client builds its own messages from fixed German strings, but a generic
    `httpx.HTTPError` carries the request URL. The controller sanitises far more
    thoroughly downstream; this only makes the adapter safe on its own.
    """
    return _URLISH.sub("[url]", text).strip()


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.debug("could not remove %s", path)


SynthesizeFn = Callable[..., Awaitable[Path]]
HealthFn = Callable[..., Awaitable[TtsHealth]]


class ServiceTTSProvider:
    """The local Qwen3-TTS HTTP service, seen through `TTSProvider`.

    `synthesize` and `health` are injectable so the contract can be exercised
    without a running service and without stubbing HTTP transports.
    """

    provider_id = "kiki-tts-service"

    def __init__(
        self,
        base_url: str = DEFAULT_TTS_URL,
        *,
        speaker: str = DEFAULT_SPEAKER,
        language: str = DEFAULT_LANGUAGE,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        timeout: float = 180.0,
        wav_dir: Path | None = None,
        synthesize: SynthesizeFn = synthesize_wav,
        health: HealthFn = tts_health,
    ) -> None:
        self._base_url = base_url
        self._speaker = speaker
        self._language = language
        self._chunk_seconds = max(0.05, float(chunk_seconds))
        self._timeout = timeout
        self._wav_dir = wav_dir
        self._owns_wav_dir = wav_dir is None
        self._synthesize = synthesize
        self._health = health
        self._status = TTSProviderStatus.UNLOADED
        self._sample_rate = DEFAULT_SAMPLE_RATE
        self._speakers: tuple[str, ...] = ()
        self._cancelled: dict[str, None] = {}
        self._inflight = 0
        self.loads = 0
        self.unloads = 0

    # --- lifecycle ---------------------------------------------------------

    @property
    def status(self) -> TTSProviderStatus:
        return self._status

    def capabilities(self) -> TTSProviderCapabilities:
        return TTSProviderCapabilities(
            provider_id=self.provider_id,
            speakers=self._speakers or CUSTOM_VOICE_SPEAKERS,
            languages=CUSTOM_VOICE_LANGUAGES,
            # One HTTP request, one finished WAV. Saying otherwise would make a
            # caller expect first audio before synthesis is done.
            streaming=False,
            needs_gpu=True,
            sample_rate=self._sample_rate,
        )

    async def health_check(self) -> TTSHealth:
        try:
            health = await self._health(self._base_url)
        except Exception as exc:  # an injected fake, or an unexpected client bug
            return TTSHealth(
                ok=False,
                status=TTSProviderStatus.ERROR,
                detail=_scrub(str(exc)) or exc.__class__.__name__,
            )
        if health.speakers:
            self._speakers = tuple(health.speakers)
        ok = bool(health.ok and health.ready)
        if ok:
            status = TTSProviderStatus.READY if self._status is TTSProviderStatus.UNLOADED else self._status
        else:
            status = self._status if self._status is TTSProviderStatus.UNLOADED else TTSProviderStatus.ERROR
        return TTSHealth(
            ok=ok,
            status=status,
            detail=_scrub(health.detail),
            capabilities=self.capabilities(),
        )

    async def load(self) -> None:
        """Confirm the service answers. The model itself lives in that process."""
        if self._status in (TTSProviderStatus.READY, TTSProviderStatus.GENERATING):
            return
        self._status = TTSProviderStatus.LOADING
        health = await self.health_check()
        if not health.ok:
            self._status = TTSProviderStatus.ERROR
            raise TTSError(
                health.detail or "TTS-Dienst ist nicht bereit",
                code="load",
                retryable=True,
            )
        self._status = TTSProviderStatus.READY
        self.loads += 1

    async def unload(self) -> None:
        """Forget the connection and drop temporary files.

        This does **not** free the service's VRAM: the model runs in another
        process and the HTTP API offers no unload. Claiming otherwise would let
        the VRAM budget count memory it never got back.
        """
        if self._status is TTSProviderStatus.UNLOADED:
            return
        self._status = TTSProviderStatus.UNLOADED
        self.unloads += 1
        self._discard_wav_dir()

    # --- synthesis ---------------------------------------------------------

    async def synthesize(self, request: TTSRequest) -> AsyncIterator[AudioChunk]:
        if self._status not in (TTSProviderStatus.READY, TTSProviderStatus.GENERATING):
            raise TTSError("TTS-Dienst wurde nicht geladen", code="not_ready")
        if request.speed != 1.0:
            # The HTTP API has no speed knob. Ignoring the field would produce
            # audio that quietly disagrees with what was asked for.
            raise TTSError("Der TTS-Dienst unterstützt kein Tempo", code="unsupported")
        if request.id in self._cancelled:
            self._cancelled.pop(request.id, None)
            return

        dest = self._wav_path()
        self._inflight += 1
        self._status = TTSProviderStatus.GENERATING
        try:
            try:
                path = await self._synthesize(
                    self._base_url,
                    request.text,
                    dest=dest,
                    language=request.language or self._language,
                    speaker=request.speaker or self._speaker,
                    timeout=self._timeout,
                )
            except TtsError as exc:
                raise self._translate(exc) from exc
            if request.id in self._cancelled:
                return
            frames, rate, channels = _read_pcm16(Path(path))
            self._sample_rate = rate
            for chunk in self._slice(request, frames, rate, channels):
                if request.id in self._cancelled:
                    return
                yield chunk
        finally:
            _unlink(dest)
            self._cancelled.pop(request.id, None)
            self._inflight -= 1
            if self._inflight == 0 and self._status is TTSProviderStatus.GENERATING:
                self._status = TTSProviderStatus.READY

    async def cancel(self, request_id: str) -> None:
        """Stop emitting chunks for this id.

        The in-flight HTTP request is not aborted from here — a non-streaming
        backend has nothing to interrupt. The controller cancels the surrounding
        task, which closes the connection; this flag makes sure no chunk that
        already arrived still reaches the speakers.
        """
        self._cancelled[request_id] = None
        while len(self._cancelled) > MAX_TRACKED_CANCELS:
            self._cancelled.pop(next(iter(self._cancelled)))

    # --- internals ---------------------------------------------------------

    def _slice(
        self, request: TTSRequest, frames: bytes, rate: int, channels: int
    ) -> list[AudioChunk]:
        frame_size = 2 * channels
        total = len(frames) // frame_size
        if total <= 0:
            return []
        per_chunk = max(1, int(rate * self._chunk_seconds))
        chunks: list[AudioChunk] = []
        start = 0
        while start < total:
            end = min(total, start + per_chunk)
            chunks.append(
                AudioChunk(
                    request_id=request.id,
                    sequence=len(chunks),
                    pcm=frames[start * frame_size : end * frame_size],
                    sample_rate=rate,
                    channels=channels,
                    final=end >= total,
                )
            )
            start = end
        return chunks

    def _translate(self, exc: TtsError) -> TTSError:
        """Map the client's message onto a code.

        `TtsError` carries no code of its own, so the sniffing lives here rather
        than spreading string comparisons across the callers — which is exactly
        what an adapter is for.
        """
        text = _scrub(str(exc)) or "TTS-Fehler"
        low = text.lower()
        if "nicht erreichbar" in low:
            self._status = TTSProviderStatus.ERROR
            return TTSError(text, code="unreachable", retryable=True)
        if "timeout" in low:
            self._status = TTSProviderStatus.ERROR
            return TTSError(text, code="timeout", retryable=True)
        if "zu groß" in low:
            return TTSError(text, code="too_large")
        if "wav" in low or "riff" in low or "inhalt" in low:
            return TTSError(text, code="format")
        return TTSError(text, code="service")

    def _wav_path(self) -> Path:
        if self._wav_dir is None:
            self._wav_dir = Path(tempfile.mkdtemp(prefix="kiki-tts-"))
            self._owns_wav_dir = True
        self._wav_dir.mkdir(parents=True, exist_ok=True)
        return self._wav_dir / f"{uuid4().hex}.wav"

    def _discard_wav_dir(self) -> None:
        if not self._owns_wav_dir or self._wav_dir is None:
            return
        directory = self._wav_dir
        self._wav_dir = None
        for leftover in directory.glob("*.wav"):
            _unlink(leftover)
        try:
            directory.rmdir()
        except OSError:
            log.debug("could not remove %s", directory)


def _read_pcm16(path: Path) -> tuple[bytes, int, int]:
    """Return raw frames, sample rate and channel count of a 16-bit WAV."""
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (EOFError, OSError, wave.Error) as exc:
        raise TTSError(f"TTS-WAV nicht lesbar: {exc}", code="format") from exc
    if width != 2:
        # AudioChunk declares pcm_s16le and derives its duration from it.
        # Relabelling other widths would misreport every length downstream.
        raise TTSError(
            f"TTS lieferte {width * 8}-Bit-Audio, erwartet werden 16 Bit",
            code="format",
        )
    return frames, rate, channels


class PipeWireAudioSink:
    """`PipeWirePlayer` seen through `AudioSink`.

    Each chunk is written to a temporary WAV and played to completion. The
    player reports through GLib callbacks, so every `play()` awaits a future that
    `stop()` and `close()` can also resolve — without that, a barge-in would drop
    the callbacks and leave the awaiting task hanging forever.
    """

    def __init__(self, player: object | None = None, *, wav_dir: Path | None = None) -> None:
        self._player = player if player is not None else PipeWirePlayer()
        self._wav_dir = wav_dir
        self._owns_wav_dir = wav_dir is None
        self._pending: asyncio.Future[str] | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def play(self, chunk: AudioChunk) -> None:
        if self._closed:
            raise TTSError("Audioausgabe ist geschlossen", code="closed")
        if chunk.audio_format != DEFAULT_AUDIO_FORMAT:
            raise TTSError(
                f"Nicht abspielbares Format: {chunk.audio_format}", code="format"
            )
        frame_size = 2 * max(1, chunk.channels)
        usable = len(chunk.pcm) - (len(chunk.pcm) % frame_size)
        if usable <= 0:
            return  # nothing to make a sound with

        path = self._wav_path()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending = future

        def _resolve(message: str) -> None:
            if not future.done():
                future.set_result(message)

        def _from_glib(message: str) -> None:
            # The player calls back on the GLib thread; without GLib it calls
            # back inline, and call_soon_threadsafe is correct in both cases.
            loop.call_soon_threadsafe(_resolve, message)

        try:
            _write_wav(path, chunk.pcm[:usable], chunk.sample_rate, chunk.channels)
            self._player.play(
                path,
                on_eos=lambda: _from_glib(""),
                on_error=lambda message: _from_glib(message or "Wiedergabe fehlgeschlagen"),
            )
            message = await future
        finally:
            if self._pending is future:
                self._pending = None
            _unlink(path)
        if message:
            raise TTSError(_scrub(message), code="playback")

    async def stop(self) -> None:
        """Barge-in. The pending `play()` returns instead of hanging."""
        pending = self._pending
        self._pending = None
        try:
            self._player.stop()
        finally:
            # Resolved as success, not as an error: an interrupted chunk is
            # what the caller asked for, not a failure it should report.
            if pending is not None and not pending.done():
                pending.set_result("")

    async def close(self) -> None:
        if self._closed:
            return
        await self.stop()
        self._closed = True
        self._discard_wav_dir()

    def _wav_path(self) -> Path:
        if self._wav_dir is None:
            self._wav_dir = Path(tempfile.mkdtemp(prefix="kiki-sink-"))
            self._owns_wav_dir = True
        self._wav_dir.mkdir(parents=True, exist_ok=True)
        return self._wav_dir / f"{uuid4().hex}.wav"

    def _discard_wav_dir(self) -> None:
        if not self._owns_wav_dir or self._wav_dir is None:
            return
        directory = self._wav_dir
        self._wav_dir = None
        for leftover in directory.glob("*.wav"):
            _unlink(leftover)
        try:
            directory.rmdir()
        except OSError:
            log.debug("could not remove %s", directory)


def _write_wav(path: Path, pcm: bytes, sample_rate: int, channels: int) -> None:
    """Create the file privately, or not at all.

    `wav_dir` may be a directory the caller already owns — the director's cache
    sits at 0755 — and a plain open() would leave synthesised speech at 0644 for
    every local user to read. O_EXCL additionally refuses to follow a symlink
    that was planted under the name we chose.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    # wave only closes a file it opened itself, so the inner block flushes and
    # patches the header while the outer one owns the descriptor.
    with open(fd, "wb") as raw, wave.open(raw, "wb") as wav:
        wav.setnchannels(max(1, channels))
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
