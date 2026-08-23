"""Client side of the local PCM streaming route.

Two pieces, both new and both separate from their WAV counterparts:

* `StreamingServiceTTSProvider` reads `POST /v1/synthesize/stream` and turns raw
  bytes into `AudioChunk`s. `ServiceTTSProvider` keeps doing WAV, untouched.
* `PipeWirePcmSink` feeds PCM to one `pw-cat` process per utterance over stdin.
  `PipeWireAudioSink` keeps writing temporary WAVs, untouched.

Neither is wired into the app. Nothing here imports torch, CUDA, GTK, GObject or
a PipeWire library, and importing the module opens no socket and starts no
process.

Two semantics worth stating plainly
-----------------------------------
**`play()` returns on hand-over, not on sound.** `PipeWireAudioSink.play` waits
for the player to finish a chunk; this sink returns once the bytes reached
`pw-cat`'s stdin. It has to: one process spans the whole utterance, so there is
no per-chunk completion to wait for. `on_audio_started` therefore still means
what the controller documents — the first real PCM went to the speakers — and
backpressure comes from the pipe, which blocks the writer once full.

**A new request id supersedes the old process.** Switching ids without a stop
terminates the previous `pw-cat` and starts a fresh one rather than raising:
the controller supersedes utterances on barge-in, and an error there would
break exactly the case that matters.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shutil
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from kiki.voice.tts.models import (
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_CHANNELS,
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
)

log = logging.getLogger(__name__)

STREAM_PATH = "/v1/synthesize/stream"
BYTES_PER_SAMPLE = 2
DEFAULT_CHUNK_MS = 400
MIN_CHUNK_MS = 160
MAX_CHUNK_MS = 1000
# One read from the socket. Small enough to notice a cancel quickly, large
# enough not to wake the loop for every kilobyte.
NETWORK_READ_BYTES = 8192
# An error body that never arrives must not hold the request open.
MAX_ERROR_BYTES = 8192
MAX_TRACKED_CANCELS = 256

PW_CAT = "pw-cat"
# Fixed and validated. Nothing here is ever built from model output, prompt text
# or user input, and no shell is involved.
#   --raw is required: without it pw-cat expects a container and refuses stdin.
#   the trailing "-" is required: without it, "filename or - argument missing".
PW_CAT_ARGS: tuple[str, ...] = (
    "--playback",
    "--raw",
    "--format",
    "s16",
    "--rate",
    str(DEFAULT_SAMPLE_RATE),
    "--channels",
    str(DEFAULT_CHANNELS),
    "-",
)
# A cancelled pw-cat drains its buffer and exits in well under a second.
PROCESS_EXIT_TIMEOUT_S = 2.0

_URLISH = re.compile(r"\b(?:https?|ftp|ws)://\S+", re.IGNORECASE)


def _scrub(text: str) -> str:
    """Drop full URLs from a message before it can reach a caller or a log."""
    return _URLISH.sub("[url]", text).strip()


# --- provider ---------------------------------------------------------------


class StreamingServiceTTSProvider:
    """The local PCM streaming endpoint, seen through `TTSProvider`.

    `client_factory` is injectable so the contract can be tested against a fake
    transport without a running service.
    """

    provider_id = "kiki-tts-stream"

    def __init__(
        self,
        base_url: str = DEFAULT_TTS_URL,
        *,
        speaker: str = DEFAULT_SPEAKER,
        language: str = DEFAULT_LANGUAGE,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        timeout: float = 180.0,
        client_factory: Callable[..., Any] = httpx.AsyncClient,
    ) -> None:
        if not MIN_CHUNK_MS <= int(chunk_ms) <= MAX_CHUNK_MS:
            raise ValueError(
                f"chunk_ms muss zwischen {MIN_CHUNK_MS} und {MAX_CHUNK_MS} liegen"
            )
        self._base_url = base_url.rstrip("/")
        self._speaker = speaker
        self._language = language
        self._chunk_ms = int(chunk_ms)
        self._timeout = timeout
        self._client_factory = client_factory
        self._status = TTSProviderStatus.UNLOADED
        self._speakers: tuple[str, ...] = ()
        self._cancelled: dict[str, None] = {}
        self._inflight = 0
        self.loads = 0
        self.unloads = 0

    @property
    def chunk_bytes(self) -> int:
        """Bytes in one full audio chunk — 19200 at the 400 ms default.

        Kept exact so a later prebuffer of N chunks is a predictable amount of
        audio rather than whatever the network happened to deliver.
        """
        samples = DEFAULT_SAMPLE_RATE * self._chunk_ms // 1000
        return samples * BYTES_PER_SAMPLE * DEFAULT_CHANNELS

    # --- lifecycle ---------------------------------------------------------

    @property
    def status(self) -> TTSProviderStatus:
        return self._status

    def capabilities(self) -> TTSProviderCapabilities:
        return TTSProviderCapabilities(
            provider_id=self.provider_id,
            speakers=self._speakers or CUSTOM_VOICE_SPEAKERS,
            languages=CUSTOM_VOICE_LANGUAGES,
            # Unlike the WAV provider, this one really does stream: the first
            # chunk exists while the model is still generating.
            streaming=True,
            needs_gpu=True,
            sample_rate=DEFAULT_SAMPLE_RATE,
        )

    async def health_check(self) -> TTSHealth:
        """Ask the service whether the PCM route exists at all.

        Deliberately its own request rather than `tts_client.tts_health`: the
        answer needed here is `streaming` and `streaming_reason`, which that
        older contract does not carry, and widening it would touch the WAV path.
        """
        try:
            async with self._client_factory(timeout=5.0) as client:
                response = await client.get(self._base_url + "/health")
                payload = response.json()
        except Exception as exc:
            return TTSHealth(
                ok=False,
                status=TTSProviderStatus.ERROR,
                detail=_scrub(str(exc)) or exc.__class__.__name__,
            )
        if not isinstance(payload, dict):
            payload = {}
        speakers = payload.get("speakers")
        if isinstance(speakers, list) and speakers:
            self._speakers = tuple(str(name) for name in speakers)
        streaming = bool(payload.get("streaming", False))
        reason = str(payload.get("streaming_reason") or "")
        ok = bool(payload.get("ok")) and bool(payload.get("ready", True)) and streaming
        detail = "" if ok else (reason or "streaming nicht verfügbar")
        status = TTSProviderStatus.READY if ok else self._status
        if not ok and self._status is not TTSProviderStatus.UNLOADED:
            status = TTSProviderStatus.ERROR
        return TTSHealth(
            ok=ok, status=status, detail=detail, capabilities=self.capabilities()
        )

    async def load(self) -> None:
        """Confirm the streaming route answers. The service owns the model."""
        if self._status in (TTSProviderStatus.READY, TTSProviderStatus.GENERATING):
            return
        self._status = TTSProviderStatus.LOADING
        health = await self.health_check()
        if not health.ok:
            self._status = TTSProviderStatus.ERROR
            raise TTSError(
                health.detail or "Streaming nicht verfügbar", code="load", retryable=True
            )
        self._status = TTSProviderStatus.READY
        self.loads += 1

    async def unload(self) -> None:
        """Release this client's own state only.

        The model lives in another process; nothing here stops the service or
        touches its VRAM, and claiming otherwise would let a budget count
        memory it never got back.
        """
        if self._status is TTSProviderStatus.UNLOADED:
            return
        self._status = TTSProviderStatus.UNLOADED
        self.unloads += 1
        self._cancelled.clear()

    # --- synthesis ---------------------------------------------------------

    async def synthesize(self, request: TTSRequest) -> AsyncIterator[AudioChunk]:
        if self._status not in (TTSProviderStatus.READY, TTSProviderStatus.GENERATING):
            raise TTSError("Streaming-Dienst wurde nicht geladen", code="not_ready")
        if request.speed != 1.0:
            raise TTSError("Der TTS-Dienst unterstützt kein Tempo", code="unsupported")
        if request.id in self._cancelled:
            self._cancelled.pop(request.id, None)
            return

        self._inflight += 1
        self._status = TTSProviderStatus.GENERATING
        try:
            async for chunk in self._read_stream(request):
                yield chunk
        finally:
            self._cancelled.pop(request.id, None)
            self._inflight -= 1
            if self._inflight == 0 and self._status is TTSProviderStatus.GENERATING:
                self._status = TTSProviderStatus.READY

    async def _read_stream(self, request: TTSRequest) -> AsyncIterator[AudioChunk]:
        body = {
            "text": request.text,
            "language": request.language or self._language,
            "speaker": request.speaker or self._speaker,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "format": DEFAULT_AUDIO_FORMAT,
            "chunk_ms": self._chunk_ms,
        }
        url = self._base_url + STREAM_PATH
        try:
            async with self._client_factory(timeout=self._timeout) as client:
                async with client.stream(
                    "POST", url, json=body, headers={"Accept": "audio/pcm"}
                ) as response:
                    if response.status_code != 200:
                        raise await self._error_from(response)
                    _validate_headers(response.headers)
                    async for chunk in self._emit(request, response):
                        yield chunk
        except TTSError:
            raise
        except httpx.ConnectError as exc:
            raise TTSError("TTS-Dienst nicht erreichbar", code="unreachable", retryable=True) from exc
        except httpx.TimeoutException as exc:
            raise TTSError("TTS-Dienst Timeout", code="timeout", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise TTSError(_scrub(str(exc)) or "HTTP-Fehler", code="service") from exc

    async def _error_from(self, response: Any) -> TTSError:
        """Turn a refusal into a coded error, reading no more than a snippet."""
        status = int(response.status_code)
        with contextlib.suppress(Exception):
            await response.aread()
        if status == 503:
            return TTSError("Streaming nicht verfügbar", code="unavailable", retryable=True)
        if status in (400, 413, 415):
            return TTSError("Anfrage wurde abgelehnt", code="rejected")
        return TTSError(f"TTS-Dienst antwortete {status}", code="service")

    async def _emit(self, request: TTSRequest, response: Any) -> AsyncIterator[AudioChunk]:
        """Regroup transport bytes into audio chunks.

        Transport boundaries are not audio boundaries: a read can end mid-sample
        and often does. The leftover byte waits for its partner rather than
        being passed on as half a sample.

        One chunk is held back so `final` can be truthful — a chunk is only the
        last one once the stream has actually ended.
        """
        buffer = bytearray()
        pending: AudioChunk | None = None
        sequence = 0
        target = self.chunk_bytes
        async for piece in response.aiter_bytes(NETWORK_READ_BYTES):
            if request.id in self._cancelled:
                return
            if not piece:
                continue
            buffer.extend(piece)
            while len(buffer) >= target:
                block = bytes(buffer[:target])
                del buffer[:target]
                if pending is not None:
                    yield pending
                pending = _chunk(request.id, sequence, block)
                sequence += 1
                if request.id in self._cancelled:
                    return
        usable = len(buffer) - (len(buffer) % BYTES_PER_SAMPLE)
        leftover = len(buffer) - usable
        if usable > 0:
            if pending is not None:
                yield pending
            pending = _chunk(request.id, sequence, bytes(buffer[:usable]))
        if pending is not None:
            yield _final(pending)
        if leftover:
            # An odd tail means the server did not send whole samples. Before
            # any audio this is a broken stream; after it, the chunks already
            # handed over stay valid and the category is recorded by the caller.
            raise TTSError(
                "Stream endete mitten in einem Sample", code="protocol"
            )

    async def cancel(self, request_id: str) -> None:
        """Stop one request. Others are untouched, and repeats are harmless.

        The read loop checks this before every transport chunk; leaving the loop
        exits the `async with`, which closes the response and the client.
        """
        self._cancelled[request_id] = None
        while len(self._cancelled) > MAX_TRACKED_CANCELS:
            self._cancelled.pop(next(iter(self._cancelled)))


def _chunk(request_id: str, sequence: int, pcm: bytes) -> AudioChunk:
    return AudioChunk(
        request_id=request_id,
        sequence=sequence,
        pcm=pcm,
        sample_rate=DEFAULT_SAMPLE_RATE,
        channels=DEFAULT_CHANNELS,
        audio_format=DEFAULT_AUDIO_FORMAT,
        final=False,
    )


def _final(chunk: AudioChunk) -> AudioChunk:
    return AudioChunk(
        request_id=chunk.request_id,
        sequence=chunk.sequence,
        pcm=chunk.pcm,
        sample_rate=chunk.sample_rate,
        channels=chunk.channels,
        audio_format=chunk.audio_format,
        final=True,
    )


_REQUIRED_HEADERS: tuple[tuple[str, str], ...] = (
    ("content-type", "audio/pcm"),
    ("x-kiki-audio-format", DEFAULT_AUDIO_FORMAT),
    ("x-kiki-sample-rate", str(DEFAULT_SAMPLE_RATE)),
    ("x-kiki-channels", str(DEFAULT_CHANNELS)),
    ("x-kiki-streaming", "true"),
    ("x-kiki-transfer", "connection-close"),
    ("cache-control", "no-store"),
)


def _validate_headers(headers: Any) -> None:
    """Refuse anything that is not the contract, before a byte is believed.

    Only header names are ever named in the error — a response body could carry
    anything, and this runs before any of it has been read as audio.
    """
    for name, expected in _REQUIRED_HEADERS:
        raw = headers.get(name)
        if raw is None:
            raise TTSError(f"Antwort ohne {name}", code="protocol")
        value = str(raw).split(";")[0].strip().lower()
        if value != expected:
            raise TTSError(f"Unerwarteter Wert in {name}", code="protocol")


# --- sink -------------------------------------------------------------------


class PipeWirePcmSink:
    """PCM straight to `pw-cat` over stdin. One process per utterance.

    `spawn` is injectable so the contract can be tested without PipeWire, a
    sound card or a real process.
    """

    def __init__(
        self,
        *,
        spawn: Callable[..., Any] | None = None,
        binary: str = PW_CAT,
        exit_timeout: float = PROCESS_EXIT_TIMEOUT_S,
    ) -> None:
        self._spawn = spawn or _spawn_pw_cat
        self._binary = binary
        self._exit_timeout = exit_timeout
        self._process: Any = None
        self._request_id: str | None = None
        self._closed = False
        # Serialises stdin: two coroutines writing the same pipe would
        # interleave samples and produce noise.
        self._lock = asyncio.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def active_request_id(self) -> str | None:
        return self._request_id

    async def play(self, chunk: AudioChunk) -> None:
        """Hand one chunk to the player, returning once stdin accepted it."""
        if self._closed:
            raise TTSError("Audioausgabe ist geschlossen", code="closed")
        if chunk.audio_format != DEFAULT_AUDIO_FORMAT:
            raise TTSError(f"Nicht abspielbares Format: {chunk.audio_format}", code="format")
        if chunk.sample_rate != DEFAULT_SAMPLE_RATE or chunk.channels != DEFAULT_CHANNELS:
            # The process was started for one fixed shape; anything else would
            # be played at the wrong speed rather than refused.
            raise TTSError("Nicht unterstützte Abtastrate oder Kanalzahl", code="format")
        frame = BYTES_PER_SAMPLE * max(1, chunk.channels)
        usable = len(chunk.pcm) - (len(chunk.pcm) % frame)
        if usable <= 0 and not chunk.final:
            return

        async with self._lock:
            if self._request_id is not None and self._request_id != chunk.request_id:
                # Barge-in: the controller superseded the utterance. Ending the
                # old process beats raising, which would break exactly that.
                await self._teardown()
            if self._process is None:
                await self._start(chunk.request_id)
            if usable > 0:
                await self._write(chunk.pcm[:usable])
            if chunk.final:
                await self._finish()

    async def stop(self) -> None:
        """Barge-in. Idempotent, and safe when nothing is playing."""
        async with self._lock:
            await self._teardown()

    async def close(self) -> None:
        """Release the device. Idempotent; leaves no process behind."""
        async with self._lock:
            await self._teardown()
            self._closed = True

    # --- internals ---------------------------------------------------------

    async def _start(self, request_id: str) -> None:
        if shutil.which(self._binary) is None and self._spawn is _spawn_pw_cat:
            raise TTSError("pw-cat ist nicht installiert", code="pw_cat_unavailable")
        try:
            self._process = await self._spawn(self._binary, PW_CAT_ARGS)
        except FileNotFoundError as exc:
            raise TTSError("pw-cat ist nicht installiert", code="pw_cat_unavailable") from exc
        except OSError as exc:
            raise TTSError(_scrub(str(exc)) or "pw-cat startete nicht", code="playback") from exc
        self._request_id = request_id

    async def _write(self, data: bytes) -> None:
        process = self._process
        stdin = getattr(process, "stdin", None)
        if stdin is None:
            raise TTSError("pw-cat hat keine Eingabe", code="playback")
        try:
            stdin.write(data)
            await stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            await self._teardown()
            raise TTSError("pw-cat hat die Wiedergabe beendet", code="playback") from exc
        except OSError as exc:
            await self._teardown()
            raise TTSError(_scrub(str(exc)) or "Schreibfehler", code="playback") from exc

    async def _finish(self) -> None:
        """Close stdin and wait for the player to drain what it was given."""
        process, self._process = self._process, None
        self._request_id = None
        if process is None:
            return
        code = await _drain_process(process, self._exit_timeout)
        if code not in (0, None):
            raise TTSError(f"pw-cat endete mit {code}", code="playback")

    async def _teardown(self) -> None:
        process, self._process = self._process, None
        self._request_id = None
        if process is None:
            return
        await _kill_process(process, self._exit_timeout)


async def _spawn_pw_cat(binary: str, args: tuple[str, ...]) -> Any:
    """Start the player. No shell, and every argument is a fixed constant."""
    return await asyncio.create_subprocess_exec(
        binary,
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


def _close_stdin(process: Any) -> None:
    stdin = getattr(process, "stdin", None)
    if stdin is None:
        return
    with contextlib.suppress(Exception):
        stdin.close()


async def _wait_closed(process: Any) -> None:
    stdin = getattr(process, "stdin", None)
    wait_closed = getattr(stdin, "wait_closed", None)
    if callable(wait_closed):
        with contextlib.suppress(Exception):
            await wait_closed()


async def _await_exit(process: Any, timeout: float) -> int | None:
    try:
        return await asyncio.wait_for(process.wait(), timeout=timeout)
    except TimeoutError:
        return None
    except Exception:
        log.debug("waiting for pw-cat failed", exc_info=True)
        return None


async def _drain_process(process: Any, timeout: float) -> int | None:
    """End of utterance: let the player finish what it was already given.

    Closing stdin and waiting is the point here — the audio still sitting in
    pw-cat's buffer is audio the listener is meant to hear.
    """
    _close_stdin(process)
    await _wait_closed(process)
    code = await _await_exit(process, timeout)
    if code is not None:
        return code
    log.debug("pw-cat did not finish in time; terminating")
    return await _kill_process(process, timeout)


async def _kill_process(process: Any, timeout: float) -> int | None:
    """Barge-in: silence now, not once the buffer has played out.

    Measured against the real player: closing stdin and waiting took 1.07 s for
    one second of buffered audio, so KIKI kept talking straight through the
    interruption. Terminating first cuts the sound immediately; stdin is closed
    afterwards so the pipe is released either way.
    """
    with contextlib.suppress(Exception):
        process.terminate()
    code = await _await_exit(process, timeout)
    if code is None:
        with contextlib.suppress(Exception):
            process.kill()
        code = await _await_exit(process, timeout)
    _close_stdin(process)
    await _wait_closed(process)
    return code
