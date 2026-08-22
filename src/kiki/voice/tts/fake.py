"""Providers that need neither a model nor a sound card.

`FakeTTSProvider` produces silent PCM of a length derived from the text, so
timing, queueing and cancellation can be exercised deterministically. Benchmarks
use it to prove the harness itself is not the bottleneck.

`NullTTSProvider` accepts everything and emits nothing — the honest stand-in for
"speech is switched off".
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from kiki.voice.tts.models import (
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    AudioChunk,
    TTSError,
    TTSHealth,
    TTSProviderCapabilities,
    TTSProviderStatus,
    TTSRequest,
)

# Roughly German speaking pace, so fake audio lengths resemble real ones.
CHARS_PER_SECOND = 15.0


def _silence(seconds: float, sample_rate: int, channels: int) -> bytes:
    frames = max(1, int(seconds * sample_rate))
    return b"\x00\x00" * frames * max(1, channels)


class FakeTTSProvider:
    """Deterministic provider for tests and benchmarks."""

    provider_id = "fake"

    def __init__(
        self,
        *,
        chunk_seconds: float = 0.5,
        latency_s: float = 0.0,
        fail_on_load: bool = False,
        fail_on_synthesize: bool = False,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self._status = TTSProviderStatus.UNLOADED
        self._chunk_seconds = max(0.05, chunk_seconds)
        self._latency = max(0.0, latency_s)
        self._fail_on_load = fail_on_load
        self._fail_on_synthesize = fail_on_synthesize
        self._sample_rate = sample_rate
        self._cancelled: set[str] = set()
        self.loads = 0
        self.unloads = 0
        self.requests: list[TTSRequest] = []

    @property
    def status(self) -> TTSProviderStatus:
        return self._status

    def capabilities(self) -> TTSProviderCapabilities:
        return TTSProviderCapabilities(
            provider_id=self.provider_id,
            speakers=("serena", "vivian"),
            languages=("german", "english"),
            streaming=True,
            needs_gpu=False,
            sample_rate=self._sample_rate,
        )

    async def health_check(self) -> TTSHealth:
        return TTSHealth(
            ok=self._status is not TTSProviderStatus.ERROR,
            status=self._status,
            detail=f"fake provider ({self._status.value})",
            capabilities=self.capabilities(),
        )

    async def load(self) -> None:
        if self._status is TTSProviderStatus.READY:
            return
        self._status = TTSProviderStatus.LOADING
        if self._fail_on_load:
            self._status = TTSProviderStatus.ERROR
            raise TTSError("Fake-Provider konnte nicht laden", code="load")
        self.loads += 1
        self._status = TTSProviderStatus.READY

    async def unload(self) -> None:
        if self._status is TTSProviderStatus.UNLOADED:
            return
        self._status = TTSProviderStatus.UNLOADING
        self.unloads += 1
        self._status = TTSProviderStatus.UNLOADED

    async def cancel(self, request_id: str) -> None:
        self._cancelled.add(request_id)

    async def synthesize(self, request: TTSRequest) -> AsyncIterator[AudioChunk]:
        if self._status is not TTSProviderStatus.READY:
            raise TTSError("Fake-Provider ist nicht geladen", code="not_ready")
        if self._fail_on_synthesize:
            self._status = TTSProviderStatus.ERROR
            raise TTSError("Fake-Synthese fehlgeschlagen", code="synthesize")

        self.requests.append(request)
        self._status = TTSProviderStatus.GENERATING
        try:
            total = max(self._chunk_seconds, len(request.text) / CHARS_PER_SECOND)
            produced = 0.0
            sequence = 0
            while produced < total:
                if request.id in self._cancelled:
                    return
                if self._latency:
                    await asyncio.sleep(self._latency)
                span = min(self._chunk_seconds, total - produced)
                produced += span
                yield AudioChunk(
                    request_id=request.id,
                    sequence=sequence,
                    pcm=_silence(span, self._sample_rate, DEFAULT_CHANNELS),
                    sample_rate=self._sample_rate,
                    final=produced >= total,
                )
                sequence += 1
        finally:
            self._cancelled.discard(request.id)
            if self._status is TTSProviderStatus.GENERATING:
                self._status = TTSProviderStatus.READY


class NullTTSProvider:
    """Speech disabled. Accepts requests, produces no audio, never fails."""

    provider_id = "null"

    @property
    def status(self) -> TTSProviderStatus:
        return TTSProviderStatus.READY

    def capabilities(self) -> TTSProviderCapabilities:
        return TTSProviderCapabilities(provider_id=self.provider_id, streaming=False)

    async def health_check(self) -> TTSHealth:
        return TTSHealth(
            ok=True,
            status=TTSProviderStatus.READY,
            detail="Sprachausgabe ist deaktiviert",
            capabilities=self.capabilities(),
        )

    async def load(self) -> None:
        return None

    async def unload(self) -> None:
        return None

    async def cancel(self, request_id: str) -> None:
        return None

    async def synthesize(self, request: TTSRequest) -> AsyncIterator[AudioChunk]:
        del request
        return
        yield  # pragma: no cover - marks this as an async generator
