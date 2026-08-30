"""SpokenTranscriber: whisper first, Vosk second, and never a blocked turn."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kiki.config.settings import Settings
from kiki.voice.stt import SpeechError
from kiki.voice.stt_client import SttServiceError
from kiki.voice.transcription import SpokenTranscriber


class FakeRemote:
    """Stands in for the HTTP call; records what was asked."""

    def __init__(self, *, answer: str = "öffne thunderbird", fail: bool = False) -> None:
        self.answer = answer
        self.fail = fail
        self.calls: list[tuple[str, bytes]] = []

    async def __call__(self, base_url: str, wav_bytes: bytes) -> str:
        self.calls.append((base_url, wav_bytes))
        if self.fail:
            raise SttServiceError("STT-Dienst nicht erreichbar")
        return self.answer


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _transcriber(monkeypatch, remote: FakeRemote, *, fallback: bool = True, clock=None):
    settings = Settings()
    transcriber = SpokenTranscriber(settings, clock=clock or FakeClock(), remote=remote)
    if not fallback:
        settings.voice.stt_fallback_vosk = False
    monkeypatch.setattr(
        "kiki.voice.transcription.transcribe_wav",
        lambda _path, *, model_id="": f"vosk:{model_id}",
    )
    return transcriber


def _wav(tmp_path: Path) -> Path:
    path = tmp_path / "take.wav"
    path.write_bytes(b"RIFF....WAVEjunk")
    return path


def test_whisper_answer_wins_over_vosk(tmp_path, monkeypatch) -> None:
    remote = FakeRemote(answer="öffne Thunderbird")
    transcriber = _transcriber(monkeypatch, remote)
    text = asyncio.run(transcriber.from_wav(_wav(tmp_path)))
    assert text == "öffne Thunderbird"
    assert remote.calls[0][0] == Settings().voice.stt_service
    assert remote.calls[0][1] == _wav(tmp_path).read_bytes()


def test_down_service_falls_back_to_vosk_and_cooldowns(tmp_path, monkeypatch) -> None:
    remote = FakeRemote(fail=True)
    clock = FakeClock()
    transcriber = _transcriber(monkeypatch, remote, clock=clock)
    # The failure falls back to the local transcript ...
    assert asyncio.run(transcriber.from_wav(_wav(tmp_path))) == "vosk:vosk-model-small-de-0.15"
    # ... and mutes the service for the cooldown window ...
    remote.fail = False
    assert asyncio.run(transcriber.from_wav(_wav(tmp_path))) == "vosk:vosk-model-small-de-0.15"
    assert len(remote.calls) == 1
    # ... until the window has passed.
    clock.advance(31.0)
    assert asyncio.run(transcriber.from_wav(_wav(tmp_path))) == "öffne thunderbird"
    assert len(remote.calls) == 2


def test_fallback_disabled_propagates_the_service_error(tmp_path, monkeypatch) -> None:
    transcriber = _transcriber(monkeypatch, FakeRemote(fail=True), fallback=False)
    with pytest.raises(SpeechError, match="nicht erreichbar"):
        asyncio.run(transcriber.from_wav(_wav(tmp_path)))


def test_pcm_commands_prefer_whisper_and_keep_vosk_when_silent(monkeypatch) -> None:
    remote = FakeRemote(answer="")
    transcriber = _transcriber(monkeypatch, remote)
    assert asyncio.run(transcriber.from_pcm("öffne sander bord", b"\x00\x00" * 16)) == "öffne sander bord"
    remote.answer = "öffne Thunderbird"
    assert asyncio.run(transcriber.from_pcm("öffne sander bord", b"\x00\x00" * 16)) == "öffne Thunderbird"


def test_empty_pcm_never_touches_the_service(monkeypatch) -> None:
    remote = FakeRemote()
    transcriber = _transcriber(monkeypatch, remote)
    assert asyncio.run(transcriber.from_pcm("hallo", b"")) == "hallo"
    assert remote.calls == []
