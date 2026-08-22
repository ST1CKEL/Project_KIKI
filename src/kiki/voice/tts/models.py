"""Data types shared by every TTS provider.

Deliberately free of torch, GTK and audio libraries: the policy, the chunker and
the tests import these, and none of them may pull CUDA into the process.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum

# Streaming format. 24 kHz mono PCM16 is what Qwen3-TTS emits and what PipeWire
# takes without resampling, so it is the one shape every provider must offer.
DEFAULT_SAMPLE_RATE = 24_000
DEFAULT_CHANNELS = 1
DEFAULT_AUDIO_FORMAT = "pcm_s16le"


class TTSError(Exception):
    """Synthesis failed. `code` lets the caller decide without parsing text."""

    def __init__(self, message: str, *, code: str = "tts", retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class TTSProviderStatus(StrEnum):
    """Lifecycle of a provider. Every state may fail into ERROR."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    GENERATING = "generating"
    UNLOADING = "unloading"
    ERROR = "error"


@dataclass(frozen=True)
class TTSProviderCapabilities:
    """What a provider can do, so the core need not special-case names."""

    provider_id: str
    speakers: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    # True when audio arrives in pieces while generation is still running.
    streaming: bool = False
    needs_gpu: bool = False
    sample_rate: int = DEFAULT_SAMPLE_RATE


@dataclass(frozen=True)
class TTSHealth:
    ok: bool
    status: TTSProviderStatus
    detail: str = ""
    capabilities: TTSProviderCapabilities | None = None


@dataclass(frozen=True)
class TTSRequest:
    """One utterance. `id` is what cancellation addresses."""

    text: str
    speaker: str = ""
    language: str = ""
    speed: float = 1.0
    sample_rate: int = DEFAULT_SAMPLE_RATE
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("TTSRequest braucht sprechbaren Text")
        if self.speed <= 0:
            raise ValueError("speed muss positiv sein")


@dataclass(frozen=True)
class AudioChunk:
    """One piece of audio, tied to the request that produced it.

    `sequence` is checked by the playback controller: a chunk from a cancelled
    or superseded request must never reach the speakers.
    """

    request_id: str
    sequence: int
    pcm: bytes
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    audio_format: str = DEFAULT_AUDIO_FORMAT
    final: bool = False

    @property
    def duration_s(self) -> float:
        frame = 2 * max(1, self.channels)  # PCM16
        return len(self.pcm) / frame / max(1, self.sample_rate)


@dataclass
class TTSGenerationResult:
    """What one completed (or aborted) generation cost."""

    request_id: str
    chunks: int = 0
    audio_seconds: float = 0.0
    synthesis_seconds: float = 0.0
    time_to_first_audio: float | None = None
    cancelled: bool = False
    error: str = ""

    @property
    def realtime_factor(self) -> float | None:
        """Synthesis time over audio length. Below 1.0 can stream indefinitely."""
        if self.audio_seconds <= 0:
            return None
        return self.synthesis_seconds / self.audio_seconds
