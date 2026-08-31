#!/usr/bin/env python3
"""The PCM streaming contract for the KIKI Kokoro TTS service.

Deliberately free of torch, CUDA and the model: everything here is validation,
framing and protocol.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger("kiki-kokoro-tts.stream")

STREAM_SAMPLE_RATE = 24_000
STREAM_CHANNELS = 1
STREAM_FORMAT = "pcm_s16le"
BYTES_PER_SAMPLE = 2

MIN_CHUNK_MS = 160
DEFAULT_CHUNK_MS = 400
MAX_CHUNK_MS = 1000

MAX_TEXT_CHARS = 4000
MAX_REQUEST_ID_CHARS = 64
_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]")

LOCK_TIMEOUT_S = 0.25


class StreamValidationError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class EngineUnavailable(Exception):
    """No usable streaming engine. The WAV route is unaffected."""


@dataclass(frozen=True)
class StreamSpec:
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
        samples = self.sample_rate * self.chunk_ms // 1000
        return samples * BYTES_PER_SAMPLE * self.channels


class CancelToken:
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
    @property
    def available(self) -> bool:
        ...

    def stream(self, spec: StreamSpec, token: CancelToken) -> Iterator[bytes]:
        ...


def _safe_id(value: Any) -> str:
    text = str(value or "").strip()[:MAX_REQUEST_ID_CHARS]
    return _ID_SAFE.sub("", text)


def _known(value: str, allowed: list[str], field: str) -> str:
    by_lower = {str(name).lower(): str(name) for name in allowed}
    match = by_lower.get(value.lower())
    if match is None:
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
    chunks: int = 0
    bytes_sent: int = 0
    started: bool = False
    cancelled: bool = False
    disconnected: bool = False
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
        outcome.error = f"engine:{type(exc).__name__}"
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            close()
    if carry:
        outcome.error = outcome.error or "engine:odd-tail"
    return outcome


class StreamGate:
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
