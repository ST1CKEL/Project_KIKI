#!/usr/bin/env python3
"""The PCM streaming contract for the KIKI TTS service.

Deliberately free of torch, CUDA and the model: everything here is validation,
framing and protocol, and every line of it is exercised by a fake generator. The
real engine arrives in a later slice behind `StreamingEngine`.

Transitional transfer contract
------------------------------
The response is a **connection-close** stream, not chunked transfer encoding.
`BaseHTTPRequestHandler` speaks HTTP/1.0 here, and chunked framing is an
HTTP/1.1 feature. Raising the whole handler to HTTP/1.1 was measured first and
rejected: with keep-alive live, any POST that answers before reading the request
body leaves those bytes in the socket, and the *next* request on that connection
is parsed out of the leftovers — a follow-up GET /health came back 400 instead
of 200. The existing WAV route has four such early returns (404, 503, 400, 413),
so HTTP/1.1 would have turned every error answer into a poisoned connection.

Until those paths drain their bodies, the stream announces itself honestly:
`Connection: close`, no Content-Length, no Transfer-Encoding claim, body ends at
EOF. `X-KIKI-Transfer: connection-close` states it in the response so a client
never has to guess.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger("kiki-tts.stream")

# The only shape the runtime produces, and therefore the only one accepted.
STREAM_SAMPLE_RATE = 24_000
STREAM_CHANNELS = 1
STREAM_FORMAT = "pcm_s16le"
BYTES_PER_SAMPLE = 2

MIN_CHUNK_MS = 160
DEFAULT_CHUNK_MS = 400
MAX_CHUNK_MS = 1000

MAX_TEXT_CHARS = 4000
MAX_REQUEST_ID_CHARS = 64
# A correlation id is the one client string that reaches the log, so it is
# reduced to characters that cannot forge a log line.
_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]")

# One generation at a time. A second caller is told so rather than queued: the
# model is single, and a queue would only hide the wait.
LOCK_TIMEOUT_S = 0.25


class StreamValidationError(Exception):
    """Rejected before anything was generated. Carries no request content."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class EngineUnavailable(Exception):
    """No usable streaming engine. The WAV route is unaffected."""


@dataclass(frozen=True)
class StreamSpec:
    """One validated streaming request."""

    text: str
    language: str
    speaker: str
    chunk_ms: int = DEFAULT_CHUNK_MS
    sample_rate: int = STREAM_SAMPLE_RATE
    audio_format: str = STREAM_FORMAT
    channels: int = STREAM_CHANNELS
    request_id: str = ""

    @property
    def chunk_bytes(self) -> int:
        """How many bytes one chunk of this length holds."""
        samples = self.sample_rate * self.chunk_ms // 1000
        return samples * BYTES_PER_SAMPLE * self.channels


class CancelToken:
    """Request-bound cooperative cancellation. One token, one generation.

    A flag rather than an exception: the engine checks it at its own step
    boundary and returns normally, so a cancel never travels as control flow.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)


@runtime_checkable
class StreamingEngine(Protocol):
    """Produces PCM16LE while it generates. Injected, never imported."""

    @property
    def available(self) -> bool:
        """False when the runtime cannot stream. The endpoint then answers 503."""
        ...

    def stream(self, spec: StreamSpec, token: CancelToken) -> Iterator[bytes]:
        """Yield raw PCM16LE for one request.

        Must stop promptly once the token is cancelled, and must not yield
        anything after that. Odd-length pieces are allowed — the writer
        reassembles them — but the total must be whole samples.
        """
        ...


def _safe_id(value: Any) -> str:
    text = str(value or "").strip()[:MAX_REQUEST_ID_CHARS]
    return _ID_SAFE.sub("", text)


def _known(value: str, allowed: list[str], field: str) -> str:
    """Case-insensitive membership, returning the caller's own spelling.

    The model lists its voices lower-case while KIKI's config capitalises them;
    a plain `in` test once silently switched the voice's gender.
    """
    by_lower = {str(name).lower(): str(name) for name in allowed}
    match = by_lower.get(value.lower())
    if match is None:
        # Names the caller sent are never echoed: the list of valid ones is.
        raise StreamValidationError(f"{field} ist unbekannt")
    return match


def validate_stream_request(
    payload: Any,
    *,
    speakers: list[str],
    languages: list[str],
    default_language: str,
    default_speaker: str,
    max_text_chars: int = MAX_TEXT_CHARS,
) -> StreamSpec:
    """Turn a request body into a StreamSpec, or refuse it.

    Every rejection message is a fixed string. Request text never reaches the
    client's error, the log, or anything else.
    """
    if not isinstance(payload, dict):
        raise StreamValidationError("JSON-Objekt erwartet")

    text = str(payload.get("text") or "").strip()
    if not text:
        raise StreamValidationError("text fehlt oder ist leer")
    if len(text) > max_text_chars:
        raise StreamValidationError(f"text länger als {max_text_chars} Zeichen", status=413)

    language = _known(
        str(payload.get("language") or default_language).strip() or default_language,
        languages,
        "language",
    )
    speaker = _known(
        str(payload.get("speaker") or default_speaker).strip() or default_speaker,
        speakers,
        "speaker",
    )

    rate = payload.get("sample_rate", STREAM_SAMPLE_RATE)
    if not isinstance(rate, int) or isinstance(rate, bool) or rate != STREAM_SAMPLE_RATE:
        raise StreamValidationError(f"sample_rate muss {STREAM_SAMPLE_RATE} sein")

    audio_format = str(payload.get("format") or STREAM_FORMAT).strip().lower()
    if audio_format != STREAM_FORMAT:
        raise StreamValidationError(f"format muss {STREAM_FORMAT} sein")

    chunk_ms = payload.get("chunk_ms", DEFAULT_CHUNK_MS)
    if not isinstance(chunk_ms, int) or isinstance(chunk_ms, bool):
        raise StreamValidationError("chunk_ms muss eine ganze Zahl sein")
    if not MIN_CHUNK_MS <= chunk_ms <= MAX_CHUNK_MS:
        raise StreamValidationError(
            f"chunk_ms muss zwischen {MIN_CHUNK_MS} und {MAX_CHUNK_MS} liegen"
        )

    speed = payload.get("speed", 1.0)
    if not isinstance(speed, (int, float)) or isinstance(speed, bool) or float(speed) != 1.0:
        # The runtime has no speed knob. Accepting the field and ignoring it
        # would return audio that quietly disagrees with the request.
        raise StreamValidationError("speed wird nicht unterstützt")

    return StreamSpec(
        text=text,
        language=language,
        speaker=speaker,
        chunk_ms=chunk_ms,
        request_id=_safe_id(payload.get("request_id")),
    )


@dataclass
class StreamOutcome:
    """What one streamed response cost. Never carries request content."""

    chunks: int = 0
    bytes_sent: int = 0
    started: bool = False
    cancelled: bool = False
    disconnected: bool = False
    # A fixed category plus an exception class name — never a message.
    error: str = ""

    @property
    def audio_seconds(self) -> float:
        return self.bytes_sent / BYTES_PER_SAMPLE / STREAM_SAMPLE_RATE


def pump_pcm(
    source: Iterator[bytes],
    *,
    on_first: Callable[[], None],
    write: Callable[[bytes], None],
    token: CancelToken,
    peer_gone: Callable[[], bool] | None = None,
) -> StreamOutcome:
    """Move PCM from the engine to the socket, keeping samples whole.

    `on_first` runs exactly once, immediately before the first byte leaves — that
    is the line between "error becomes an HTTP status" and "error becomes a
    closed connection". An engine failure before it is re-raised for the caller
    to turn into JSON; after it, nothing but PCM may ever reach this socket.

    `peer_gone` is asked before every chunk. A failing `write` is the second
    line of defence, not the first: the kernel accepts writes into its send
    buffer long after the listener left, so waiting for EPIPE would keep the GPU
    busy for megabytes of audio nobody can hear.
    """
    outcome = StreamOutcome()
    carry = b""
    try:
        for piece in source:
            if token.cancelled:
                outcome.cancelled = True
                break
            if peer_gone is not None and peer_gone():
                token.cancel()
                outcome.disconnected = True
                outcome.cancelled = True
                break
            if not piece:
                continue
            data = carry + piece
            # A chunk must never end mid-sample: the odd byte waits for its
            # partner instead of being sent as half a sample.
            usable = len(data) - (len(data) % (BYTES_PER_SAMPLE * STREAM_CHANNELS))
            carry = data[usable:]
            if usable <= 0:
                continue
            block = data[:usable]
            if not outcome.started:
                on_first()
                outcome.started = True
            try:
                write(block)
            except (BrokenPipeError, ConnectionResetError, OSError):
                # The listener is gone. Stop the GPU rather than finish an
                # answer nobody will hear.
                token.cancel()
                outcome.disconnected = True
                outcome.cancelled = True
                break
            outcome.chunks += 1
            outcome.bytes_sent += len(block)
    except BaseException as exc:
        token.cancel()
        if not outcome.started:
            raise
        # Past the first byte the protocol is pure audio. The category is
        # logged; the connection simply ends.
        outcome.error = f"engine:{type(exc).__name__}"
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            close()
    if carry:
        # Only reachable if an engine ends on an odd byte, which is a bug in it.
        outcome.error = outcome.error or "engine:odd-tail"
    return outcome


class StreamGate:
    """At most one streaming generation. A second caller gets 503, not a queue."""

    def __init__(self, lock: threading.Lock, timeout: float = LOCK_TIMEOUT_S) -> None:
        self._lock = lock
        self._timeout = timeout

    def acquire(self) -> bool:
        return self._lock.acquire(timeout=self._timeout)

    def release(self) -> None:
        try:
            self._lock.release()
        except RuntimeError:
            log.debug("stream gate was not held")


class FakeStreamingEngine:
    """A generator with no model behind it, for the contract tests.

    Every failure mode the endpoint must handle is reachable from here: nothing
    at all, a failure before the first byte, a failure part way through, an odd
    trailing byte, and a run long enough to cancel.
    """

    def __init__(
        self,
        *,
        chunks: int = 4,
        available: bool = True,
        fail_before_first: bool = False,
        fail_after_chunk: int | None = None,
        odd_tail: bool = False,
        delay_s: float = 0.0,
        pattern: bytes = b"\x01\x02",
    ) -> None:
        self._chunks = chunks
        self._available = available
        self._fail_before_first = fail_before_first
        self._fail_after = fail_after_chunk
        self._odd_tail = odd_tail
        self._delay = delay_s
        self._pattern = pattern
        self.calls: list[StreamSpec] = []
        self.produced = 0
        self.closed = False
        self.saw_cancel = False

    @property
    def available(self) -> bool:
        return self._available

    def stream(self, spec: StreamSpec, token: CancelToken) -> Iterator[bytes]:
        self.calls.append(spec)
        try:
            if self._fail_before_first:
                raise RuntimeError("Generator kaputt vor dem ersten Byte")
            for index in range(self._chunks):
                if token.cancelled:
                    self.saw_cancel = True
                    return
                if self._delay:
                    token.wait(self._delay)
                if self._fail_after is not None and index == self._fail_after:
                    raise RuntimeError("Generator kaputt mitten im Strom")
                block = self._pattern * (spec.chunk_bytes // len(self._pattern))
                self.produced += 1
                yield block
            if self._odd_tail:
                yield b"\x07"
        finally:
            self.closed = True
            if token.cancelled:
                self.saw_cancel = True
