"""One transcript source for spoken audio: whisper service first, Vosk second.

Vosk hears and segments inside the app — the wake word and the end of an
utterance can only come from a streamer. Its transcript, however, mangles
proper nouns, so the captured audio passage is handed to the local
faster-whisper service whenever that service answers. Everything here fails
soft: a down service costs a 30-second window of Vosk quality, never a voice
turn.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path

from kiki.config.settings import Settings
from kiki.voice.stt import SpeechError, transcribe_wav
from kiki.voice.stt_client import (
    SttServiceError,
    pcm16_to_wav,
    transcribe_wav_remote,
)

log = logging.getLogger(__name__)

# Mirrors the TTS fallback: one failure mutes the service for a short window
# instead of every utterance paying the connect timeout.
RETRY_COOLDOWN_S = 30.0


class SpokenTranscriber:
    def __init__(
        self,
        settings: Settings,
        *,
        clock: Callable[[], float] = time.monotonic,
        remote: Callable[..., object] = transcribe_wav_remote,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._remote = remote
        self._retry_after = 0.0
        self._last_error = ""

    @property
    def last_error(self) -> str:
        return self._last_error

    async def from_wav(self, path: Path) -> str:
        """Transcript for a push-to-talk recording (a WAV file on disk)."""
        if self._clock() >= self._retry_after:
            try:
                wav_bytes = path.read_bytes()
            except OSError as exc:
                raise SpeechError(f"Aufnahme kann nicht gelesen werden: {exc}") from exc
            text = await self._transcribe_remote(wav_bytes)
            if text is not None:
                return text
        if self._settings.voice.stt_fallback_vosk:
            return await asyncio.to_thread(
                transcribe_wav, path, model_id=self._settings.voice.stt_model
            )
        raise SpeechError(
            self._last_error or "Spracherkennungsdienst vorübergehend nicht erreichbar"
        )

    async def from_pcm(self, vosk_text: str, pcm: bytes) -> str:
        """Transcript for a captured wake command; never blocks on a down service.

        The Vosk transcript is already in hand, so when the service is in its
        cooldown window (or no audio was captured) it is simply used as-is.
        """
        if pcm and self._clock() >= self._retry_after:
            text = await self._transcribe_remote(pcm16_to_wav(pcm))
            if text is not None:
                return text or vosk_text
        return vosk_text

    async def _transcribe_remote(self, wav_bytes: bytes) -> str | None:
        """One service round trip; None means "unavailable, use the fallback"."""
        now = self._clock()
        if now < self._retry_after:
            return None
        try:
            text = await self._remote(self._settings.voice.stt_service, wav_bytes)
        except SttServiceError as exc:
            self._retry_after = self._clock() + RETRY_COOLDOWN_S
            self._last_error = str(exc)
            log.info("whisper STT unavailable, using the Vosk transcript: %s", exc)
            return None
        self._retry_after = 0.0
        self._last_error = ""
        return text
