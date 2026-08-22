"""The two adapters that put the existing TTS route behind the new contracts.

Nothing here talks to the service, to PipeWire or to a sound card: the HTTP call
and the player are injected, so what is under test is the adapter itself — the
WAV/PCM conversion, the request identity, the error translation and the
lifecycle guarantees the protocols promise.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import stat
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import pytest

from kiki.voice.tts import (
    DEFAULT_SAMPLE_RATE,
    AudioChunk,
    AudioSink,
    TTSError,
    TTSProvider,
    TTSProviderStatus,
    TTSRequest,
    VoicePlaybackController,
)
from kiki.voice.tts.adapters import (
    MAX_TRACKED_CANCELS,
    PipeWireAudioSink,
    ServiceTTSProvider,
)
from kiki.voice.tts_client import TtsError, TtsHealth

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def temp_root(tmp_path, monkeypatch):
    """Redirect every implicit temp path into the test's own directory.

    The adapters fall back to a temp directory when no wav_dir is given, and
    thirty tests take that path. Pointed at the real /tmp they left a directory
    behind per test — which is how the leak under investigation grew as fast as
    it did. Yielded so a test can assert the root came out empty.
    """
    root = tmp_path / "tmproot"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


# --- helpers ---------------------------------------------------------------


def _pcm(frames: int, channels: int = 1) -> bytes:
    """Distinguishable PCM16, so a mis-sliced chunk shows up as wrong bytes."""
    return b"".join(
        (i % 30_000).to_bytes(2, "little", signed=False) * channels for i in range(frames)
    )


def _write_wav(path: Path, pcm: bytes, *, rate: int = DEFAULT_SAMPLE_RATE, channels: int = 1,
               width: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return path


def _healthy(**overrides) -> TtsHealth:
    values = dict(ok=True, ready=True, detail="TTS erreichbar", device="cuda", model="qwen3-tts")
    values.update(overrides)
    return TtsHealth(**values)


def _provider(*, pcm: bytes | None = None, rate: int = DEFAULT_SAMPLE_RATE, channels: int = 1,
              width: int = 2, raises: BaseException | None = None,
              health: TtsHealth | None = None, chunk_seconds: float = 0.5,
              corrupt: bool = False, **kwargs) -> tuple[ServiceTTSProvider, list[dict]]:
    """A provider whose HTTP call is a local WAV writer. Records what it was asked."""
    seen: list[dict] = []
    payload = _pcm(1000) if pcm is None else pcm

    async def _synth(base_url, text, *, dest, language, speaker, timeout):
        seen.append(
            {"base_url": base_url, "text": text, "language": language,
             "speaker": speaker, "timeout": timeout, "dest": Path(dest)}
        )
        if raises is not None:
            raise raises
        if corrupt:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"RIFF....WAVEnope")
            return Path(dest)
        return _write_wav(Path(dest), payload, rate=rate, channels=channels, width=width)

    async def _health(base_url, **_kwargs):
        return health if health is not None else _healthy()

    return (
        ServiceTTSProvider(
            synthesize=_synth, health=_health, chunk_seconds=chunk_seconds, **kwargs
        ),
        seen,
    )


def _collect(provider: ServiceTTSProvider, request: TTSRequest) -> list[AudioChunk]:
    async def go():
        await provider.load()
        return [chunk async for chunk in provider.synthesize(request)]

    return asyncio.run(go())


class _FakePlayer:
    """Stands in for PipeWirePlayer, reading back whatever WAV it was handed."""

    def __init__(self, *, auto_eos: bool = True, fail_with: str | None = None) -> None:
        self.paths: list[Path] = []
        self.frames: list[tuple[bytes, int, int]] = []
        self.stopped = 0
        self._auto_eos = auto_eos
        self._fail_with = fail_with
        self._eos = None
        self._err = None

    def play(self, path: Path, *, on_eos=None, on_error=None) -> None:
        self.paths.append(Path(path))
        with wave.open(str(path), "rb") as wav:
            self.frames.append(
                (wav.readframes(wav.getnframes()), wav.getframerate(), wav.getnchannels())
            )
        self._eos, self._err = on_eos, on_error
        if self._fail_with is not None:
            self.fail(self._fail_with)
        elif self._auto_eos:
            self.finish()

    def stop(self) -> None:
        self.stopped += 1
        self._eos = self._err = None

    def finish(self) -> None:
        callback, self._eos, self._err = self._eos, None, None
        if callback is not None:
            callback()

    def fail(self, message: str) -> None:
        callback, self._eos, self._err = self._err, None, None
        if callback is not None:
            callback(message)


async def _until_playing(player: _FakePlayer) -> None:
    for _ in range(20):
        if player.paths:
            return
        await asyncio.sleep(0)
    raise AssertionError("Der Sink hat den Player nie erreicht")


# --- provider: contract and lifecycle --------------------------------------


def test_the_client_adapter_satisfies_the_provider_protocol() -> None:
    provider, _ = _provider()
    assert isinstance(provider, TTSProvider)
    assert provider.capabilities().provider_id == provider.provider_id


def test_capabilities_do_not_claim_streaming() -> None:
    """One HTTP request returns one finished WAV. Saying `streaming=True` would
    make a caller expect first audio before synthesis is done."""
    provider, _ = _provider()
    assert provider.capabilities().streaming is False


def test_lifecycle_moves_through_the_documented_states() -> None:
    provider, _ = _provider()

    async def go():
        assert provider.status is TTSProviderStatus.UNLOADED
        await provider.load()
        assert provider.status is TTSProviderStatus.READY
        async for _chunk in provider.synthesize(TTSRequest(text="Hallo Martin.")):
            pass
        assert provider.status is TTSProviderStatus.READY
        await provider.unload()
        assert provider.status is TTSProviderStatus.UNLOADED

    asyncio.run(go())


def test_loading_twice_only_probes_once() -> None:
    provider, _ = _provider()

    async def go():
        await provider.load()
        await provider.load()

    asyncio.run(go())
    assert provider.loads == 1


def test_unloading_when_never_loaded_is_safe() -> None:
    provider, _ = _provider()
    asyncio.run(provider.unload())
    assert provider.status is TTSProviderStatus.UNLOADED
    assert provider.unloads == 0


def test_an_unready_service_fails_the_load_and_leaves_it_in_error() -> None:
    provider, _ = _provider(health=TtsHealth(ok=True, ready=False, detail="Modell lädt noch"))
    with pytest.raises(TTSError) as excinfo:
        asyncio.run(provider.load())
    assert excinfo.value.code == "load"
    assert excinfo.value.retryable is True
    assert provider.status is TTSProviderStatus.ERROR


def test_synthesising_before_loading_fails_clearly() -> None:
    provider, seen = _provider()

    async def go():
        async for _chunk in provider.synthesize(TTSRequest(text="Zu früh.")):
            pass

    with pytest.raises(TTSError) as excinfo:
        asyncio.run(go())
    assert excinfo.value.code == "not_ready"
    assert seen == []  # nothing was sent


def test_health_check_survives_a_broken_client() -> None:
    async def _boom(_base_url, **_kwargs):
        raise RuntimeError("kaputt")

    provider = ServiceTTSProvider(health=_boom, synthesize=None)  # type: ignore[arg-type]
    health = asyncio.run(provider.health_check())
    assert health.ok is False
    assert health.status is TTSProviderStatus.ERROR
    # Reporting an error is not the same as entering one: a probe must not
    # change the lifecycle behind the caller's back.
    assert provider.status is TTSProviderStatus.UNLOADED


def test_health_check_adopts_the_speakers_the_service_reports() -> None:
    provider, _ = _provider(health=_healthy(speakers=("serena", "aiden")))
    asyncio.run(provider.health_check())
    assert provider.capabilities().speakers == ("serena", "aiden")


# --- provider: WAV to chunks -----------------------------------------------


def test_the_wav_becomes_chunks_that_reassemble_to_the_original_audio() -> None:
    payload = _pcm(24_000)  # exactly one second at 24 kHz
    provider, _ = _provider(pcm=payload, chunk_seconds=0.5)
    chunks = _collect(provider, TTSRequest(text="Eine Sekunde."))

    assert len(chunks) == 2
    assert b"".join(c.pcm for c in chunks) == payload
    assert pytest.approx(sum(c.duration_s for c in chunks)) == 1.0


def test_every_chunk_carries_the_requesting_id_and_a_dense_sequence() -> None:
    """The controller drops chunks whose id it does not expect, so an adapter
    that invents its own id would silently produce total silence."""
    provider, _ = _provider(pcm=_pcm(24_000), chunk_seconds=0.25)
    request = TTSRequest(text="Wer hat das bestellt?", id="req-4711")
    chunks = _collect(provider, request)

    assert {c.request_id for c in chunks} == {"req-4711"}
    assert [c.sequence for c in chunks] == list(range(len(chunks)))


def test_exactly_the_last_chunk_is_final() -> None:
    provider, _ = _provider(pcm=_pcm(24_000), chunk_seconds=0.25)
    chunks = _collect(provider, TTSRequest(text="Vier Stücke."))

    assert [c.final for c in chunks] == [False] * (len(chunks) - 1) + [True]


def test_a_short_utterance_is_one_final_chunk() -> None:
    provider, _ = _provider(pcm=_pcm(600), chunk_seconds=0.5)
    chunks = _collect(provider, TTSRequest(text="Ja."))

    assert len(chunks) == 1
    assert chunks[0].final is True


def test_chunks_never_split_a_frame() -> None:
    """A stereo chunk cut mid-frame swaps the channels for the rest of the
    utterance and sounds like noise."""
    provider, _ = _provider(pcm=_pcm(24_000, channels=2), channels=2, chunk_seconds=0.1)
    chunks = _collect(provider, TTSRequest(text="Stereo."))

    assert all(len(c.pcm) % 4 == 0 for c in chunks)
    assert {c.channels for c in chunks} == {2}


def test_the_rate_of_the_wav_wins_over_the_requested_one() -> None:
    """The service returns what it returns. The chunk reports the truth rather
    than the wish, so downstream duration arithmetic stays correct."""
    provider, _ = _provider(pcm=_pcm(16_000), rate=16_000)
    chunks = _collect(provider, TTSRequest(text="Andere Rate.", sample_rate=48_000))

    assert {c.sample_rate for c in chunks} == {16_000}
    assert provider.capabilities().sample_rate == 16_000


def test_speaker_and_language_of_the_request_reach_the_service() -> None:
    provider, seen = _provider(speaker="Serena", language="German")
    _collect(provider, TTSRequest(text="Hallo.", speaker="Vivian", language="English"))
    assert seen[-1]["speaker"] == "Vivian"
    assert seen[-1]["language"] == "English"


def test_an_empty_speaker_falls_back_to_the_configured_default() -> None:
    provider, seen = _provider(speaker="Serena", language="German")
    _collect(provider, TTSRequest(text="Hallo."))
    assert seen[-1]["speaker"] == "Serena"
    assert seen[-1]["language"] == "German"


def test_the_temporary_wav_is_removed_after_synthesis(tmp_path: Path) -> None:
    provider, _ = _provider(wav_dir=tmp_path)
    _collect(provider, TTSRequest(text="Aufräumen."))
    assert list(tmp_path.glob("*.wav")) == []


def test_unload_removes_the_temporary_directory_it_created() -> None:
    provider, _ = _provider()

    async def go():
        await provider.load()
        async for _chunk in provider.synthesize(TTSRequest(text="Kurz.")):
            pass
        directory = provider._wav_dir
        await provider.unload()
        return directory

    directory = asyncio.run(go())
    assert directory is not None and not directory.exists()


# --- provider: error translation -------------------------------------------


@pytest.mark.parametrize(
    ("message", "code", "retryable"),
    [
        ("TTS-Dienst nicht erreichbar", "unreachable", True),
        ("TTS-Dienst Timeout", "timeout", True),
        ("TTS-WAV ist zu groß (maximal 64 MiB).", "too_large", False),
        ("TTS lieferte keine gültigen RIFF/WAVE-Daten.", "format", False),
        ("Unerwarteter TTS-Inhalt: text/html", "format", False),
        ("TTS-Fehler: 500 Internal Server Error", "service", False),
    ],
)
def test_client_errors_become_coded_tts_errors(message, code, retryable) -> None:
    provider, _ = _provider(raises=TtsError(message))

    with pytest.raises(TTSError) as excinfo:
        _collect(provider, TTSRequest(text="Egal."))
    assert excinfo.value.code == code
    assert excinfo.value.retryable is retryable


def test_an_unreachable_service_puts_the_provider_into_error() -> None:
    """A single bad sentence is not the same as a dead service, and the status
    is what a caller uses to decide whether to reload."""
    provider, _ = _provider(raises=TtsError("TTS-Dienst nicht erreichbar"))
    with pytest.raises(TTSError):
        _collect(provider, TTSRequest(text="Egal."))
    assert provider.status is TTSProviderStatus.ERROR


def test_a_single_failed_sentence_keeps_the_provider_usable() -> None:
    provider, _ = _provider(raises=TtsError("TTS-Fehler: 500 Internal Server Error"))
    with pytest.raises(TTSError):
        _collect(provider, TTSRequest(text="Egal."))
    assert provider.status is TTSProviderStatus.READY


def test_a_full_url_never_survives_the_translation() -> None:
    provider, _ = _provider(
        raises=TtsError("Verbindung zu https://tts.internal:18765/v1/synthesize?key=abc verloren")
    )
    with pytest.raises(TTSError) as excinfo:
        _collect(provider, TTSRequest(text="Egal."))
    assert "tts.internal" not in str(excinfo.value)
    assert "[url]" in str(excinfo.value)


def test_non_16_bit_audio_is_refused_instead_of_relabelled() -> None:
    """AudioChunk declares pcm_s16le and derives duration from it; passing 8-bit
    audio through would misreport every length downstream."""
    provider, _ = _provider(pcm=b"\x80" * 4000, width=1)
    with pytest.raises(TTSError) as excinfo:
        _collect(provider, TTSRequest(text="Acht Bit."))
    assert excinfo.value.code == "format"
    assert "16 Bit" in str(excinfo.value)


def test_an_unreadable_wav_is_a_format_error(tmp_path: Path) -> None:
    provider, _ = _provider(corrupt=True, wav_dir=tmp_path)
    with pytest.raises(TTSError) as excinfo:
        _collect(provider, TTSRequest(text="Kaputt."))
    assert excinfo.value.code == "format"
    assert list(tmp_path.glob("*.wav")) == []  # still cleaned up


def test_a_speed_the_service_cannot_do_is_refused() -> None:
    """The HTTP API has no speed knob. Ignoring the field would return audio
    that quietly disagrees with what was asked for."""
    provider, seen = _provider()
    with pytest.raises(TTSError) as excinfo:
        _collect(provider, TTSRequest(text="Schneller.", speed=1.5))
    assert excinfo.value.code == "unsupported"
    assert seen == []


# --- provider: cancellation ------------------------------------------------


def test_cancelling_before_synthesis_produces_no_audio() -> None:
    provider, seen = _provider()
    request = TTSRequest(text="Doch nicht.", id="req-early")

    async def go():
        await provider.load()
        await provider.cancel("req-early")
        return [c async for c in provider.synthesize(request)]

    assert asyncio.run(go()) == []
    assert seen == []  # not even sent


def test_cancelling_mid_stream_stops_the_chunks() -> None:
    provider, _ = _provider(pcm=_pcm(24_000), chunk_seconds=0.1)
    request = TTSRequest(text="Zehn Stücke.", id="req-mid")

    async def go():
        await provider.load()
        out = []
        async for chunk in provider.synthesize(request):
            out.append(chunk)
            if len(out) == 3:
                await provider.cancel("req-mid")
        return out

    chunks = asyncio.run(go())
    assert len(chunks) == 3
    assert chunks[-1].final is False  # a cancelled stream has no final chunk


def test_cancelling_an_unknown_id_is_not_an_error() -> None:
    provider, _ = _provider()
    asyncio.run(provider.cancel("nie-gesehen"))


def test_forgotten_cancellations_stay_bounded() -> None:
    """Ids that never reach synthesize() would otherwise accumulate for the
    lifetime of the process."""
    provider, _ = _provider()

    async def go():
        for index in range(MAX_TRACKED_CANCELS + 50):
            await provider.cancel(f"req-{index}")

    asyncio.run(go())
    assert len(provider._cancelled) == MAX_TRACKED_CANCELS


def test_a_completed_request_stops_being_tracked() -> None:
    provider, _ = _provider(pcm=_pcm(2000), chunk_seconds=0.05)
    request = TTSRequest(text="Fertig.", id="req-done")

    async def go():
        await provider.load()
        async for chunk in provider.synthesize(request):
            if chunk.sequence == 0:
                await provider.cancel("req-done")

    asyncio.run(go())
    assert "req-done" not in provider._cancelled


# --- sink -------------------------------------------------------------------


def test_the_player_adapter_satisfies_the_sink_protocol() -> None:
    assert isinstance(PipeWireAudioSink(_FakePlayer()), AudioSink)


def test_a_chunk_reaches_the_player_as_a_playable_wav(tmp_path: Path) -> None:
    player = _FakePlayer()
    sink = PipeWireAudioSink(player, wav_dir=tmp_path)
    payload = _pcm(1200)

    asyncio.run(
        sink.play(AudioChunk(request_id="r", sequence=0, pcm=payload, sample_rate=16_000))
    )

    assert player.frames == [(payload, 16_000, 1)]


def test_stereo_survives_the_round_trip(tmp_path: Path) -> None:
    player = _FakePlayer()
    sink = PipeWireAudioSink(player, wav_dir=tmp_path)
    payload = _pcm(600, channels=2)

    asyncio.run(
        sink.play(
            AudioChunk(request_id="r", sequence=0, pcm=payload, channels=2, sample_rate=24_000)
        )
    )

    assert player.frames == [(payload, 24_000, 2)]


def test_the_temporary_wav_does_not_outlive_the_chunk(tmp_path: Path) -> None:
    player = _FakePlayer()
    sink = PipeWireAudioSink(player, wav_dir=tmp_path)
    asyncio.run(sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(400))))

    assert list(tmp_path.glob("*.wav")) == []


def test_a_partial_frame_is_trimmed_rather_than_written_as_noise(tmp_path: Path) -> None:
    player = _FakePlayer()
    sink = PipeWireAudioSink(player, wav_dir=tmp_path)
    asyncio.run(
        sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(100) + b"\x01"))
    )

    assert player.frames[0][0] == _pcm(100)


def test_an_empty_chunk_never_reaches_the_player(tmp_path: Path) -> None:
    player = _FakePlayer()
    sink = PipeWireAudioSink(player, wav_dir=tmp_path)
    asyncio.run(sink.play(AudioChunk(request_id="r", sequence=0, pcm=b"", final=True)))

    assert player.paths == []


def test_the_temporary_wav_is_unreadable_for_other_users(tmp_path: Path) -> None:
    """`wav_dir` may be a directory the caller owns — the director's cache is
    0755 — and a plain open() left the audio at 0644 for every local account."""
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o777)
    os.chmod(shared, 0o777)
    modes: list[int] = []

    class _Peek(_FakePlayer):
        def play(self, path, *, on_eos=None, on_error=None):
            modes.append(stat.S_IMODE(Path(path).stat().st_mode))
            super().play(path, on_eos=on_eos, on_error=on_error)

    sink = PipeWireAudioSink(_Peek(), wav_dir=shared)
    asyncio.run(sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(400))))

    assert modes == [0o600]


def test_an_existing_name_is_refused_rather_than_overwritten(tmp_path: Path) -> None:
    """O_EXCL: a symlink planted under the chosen name must not be followed."""
    from kiki.voice.tts.adapters import _write_wav

    victim = tmp_path / "victim"
    victim.write_bytes(b"wichtig")
    with pytest.raises(FileExistsError):
        _write_wav(victim, _pcm(10), DEFAULT_SAMPLE_RATE, 1)
    assert victim.read_bytes() == b"wichtig"


def test_no_filename_anywhere_carries_the_text_or_the_request_id(tmp_path: Path) -> None:
    """Names land in ls, in logs and in crash dumps. The only thing they may
    contain is a random hex id."""
    secret = "sk-live-4711 https://intern.example.com/v1 Passwort hunter2"
    provider, seen = _provider(pcm=_pcm(1200), wav_dir=tmp_path / "tts")
    player = _FakePlayer()
    sink = PipeWireAudioSink(player, wav_dir=tmp_path / "sink")

    async def go():
        await provider.load()
        controller = VoicePlaybackController(provider, sink)
        await controller.speak(TTSRequest(text=secret, id="tok-sk-live-4711"))
        await controller.shutdown()

    asyncio.run(go())

    names = [call["dest"].name for call in seen] + [path.name for path in player.paths]
    assert names, "es wurde gar keine Datei erzeugt"
    for name in names:
        assert re.fullmatch(r"[0-9a-f]{32}\.wav", name), name


def test_cancelling_a_pending_play_removes_the_temporary_wav(tmp_path: Path) -> None:
    """The controller cancels the task rather than calling stop() on every path,
    so the cleanup may not depend on stop() running."""
    player = _FakePlayer(auto_eos=False)

    async def go():
        sink = PipeWireAudioSink(player, wav_dir=tmp_path)
        task = asyncio.create_task(
            sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(48_000)))
        )
        await _until_playing(player)
        assert list(tmp_path.glob("*.wav")), "die Datei existierte nie"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert list(tmp_path.glob("*.wav")) == []


def test_shutdown_while_a_chunk_plays_leaves_no_temporary_wav(tmp_path: Path) -> None:
    provider, _ = _provider(pcm=_pcm(24_000), chunk_seconds=0.25, wav_dir=tmp_path / "tts")
    player = _FakePlayer(auto_eos=False)
    sink = PipeWireAudioSink(player, wav_dir=tmp_path / "sink")

    async def go():
        await provider.load()
        controller = VoicePlaybackController(provider, sink)
        task = await controller.submit(TTSRequest(text="Wird abgebrochen."))
        await _until_playing(player)
        await controller.shutdown()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert list((tmp_path / "sink").glob("*.wav")) == []
    assert list((tmp_path / "tts").glob("*.wav")) == []


def test_an_unplayable_format_is_refused(tmp_path: Path) -> None:
    player = _FakePlayer()
    sink = PipeWireAudioSink(player, wav_dir=tmp_path)
    chunk = AudioChunk(request_id="r", sequence=0, pcm=b"\x00" * 100, audio_format="opus")

    with pytest.raises(TTSError) as excinfo:
        asyncio.run(sink.play(chunk))
    assert excinfo.value.code == "format"
    assert player.paths == []


def test_a_playback_failure_becomes_a_tts_error(tmp_path: Path) -> None:
    player = _FakePlayer(fail_with="pulsesink: kein Gerät")
    sink = PipeWireAudioSink(player, wav_dir=tmp_path)

    with pytest.raises(TTSError) as excinfo:
        asyncio.run(sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(400))))
    assert excinfo.value.code == "playback"
    assert "kein Gerät" in str(excinfo.value)
    assert list(tmp_path.glob("*.wav")) == []


def test_a_playback_error_leaves_the_sink_usable(tmp_path: Path) -> None:
    """The protocol allows play() to raise, but forbids it to break the sink."""
    player = _FakePlayer(fail_with="einmalig")

    async def go():
        sink = PipeWireAudioSink(player, wav_dir=tmp_path)
        with pytest.raises(TTSError):
            await sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(400)))
        player._fail_with = None
        await sink.play(AudioChunk(request_id="r", sequence=1, pcm=_pcm(400)))

    asyncio.run(go())
    assert len(player.paths) == 2


def test_stop_releases_a_pending_play_instead_of_hanging(tmp_path: Path) -> None:
    """PipeWirePlayer.stop() drops its callbacks. Without the sink resolving the
    future itself, every barge-in would strand the awaiting task forever."""
    player = _FakePlayer(auto_eos=False)

    async def go():
        sink = PipeWireAudioSink(player, wav_dir=tmp_path)
        task = asyncio.create_task(
            sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(48_000)))
        )
        await _until_playing(player)
        await sink.stop()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(go())
    assert player.stopped == 1
    assert list(tmp_path.glob("*.wav")) == []


def test_an_interrupted_chunk_is_not_reported_as_a_failure(tmp_path: Path) -> None:
    """A barge-in is what the caller asked for, not an error it should surface."""
    player = _FakePlayer(auto_eos=False)

    async def go():
        sink = PipeWireAudioSink(player, wav_dir=tmp_path)
        task = asyncio.create_task(
            sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(48_000)))
        )
        await _until_playing(player)
        await sink.stop()
        return await asyncio.wait_for(task, timeout=1)

    assert asyncio.run(go()) is None


def test_stop_is_idempotent_and_safe_when_nothing_plays(tmp_path: Path) -> None:
    player = _FakePlayer()

    async def go():
        sink = PipeWireAudioSink(player, wav_dir=tmp_path)
        await sink.stop()
        await sink.stop()
        await sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(400)))
        await sink.stop()

    asyncio.run(go())
    assert player.stopped == 3
    assert len(player.paths) == 1


def test_close_is_idempotent(tmp_path: Path) -> None:
    player = _FakePlayer()

    async def go():
        sink = PipeWireAudioSink(player, wav_dir=tmp_path)
        await sink.close()
        await sink.close()
        return sink

    sink = asyncio.run(go())
    assert sink.closed is True
    assert player.stopped == 1  # the second close() does nothing at all


def test_playing_after_close_is_refused(tmp_path: Path) -> None:
    player = _FakePlayer()

    async def go():
        sink = PipeWireAudioSink(player, wav_dir=tmp_path)
        await sink.close()
        with pytest.raises(TTSError) as excinfo:
            await sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(400)))
        return excinfo.value

    assert asyncio.run(go()).code == "closed"
    assert player.paths == []


def test_close_releases_a_pending_play(tmp_path: Path) -> None:
    player = _FakePlayer(auto_eos=False)

    async def go():
        sink = PipeWireAudioSink(player, wav_dir=tmp_path)
        task = asyncio.create_task(
            sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(48_000)))
        )
        await _until_playing(player)
        await sink.close()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(go())


def test_close_removes_the_directory_it_created_itself() -> None:
    player = _FakePlayer()

    async def go():
        sink = PipeWireAudioSink(player)
        await sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(400)))
        directory = sink._wav_dir
        await sink.close()
        return directory

    directory = asyncio.run(go())
    assert directory is not None and not directory.exists()


def test_a_borrowed_directory_is_left_alone(tmp_path: Path) -> None:
    """Deleting a directory the caller owns would take the director's cache with
    it the day both share one."""
    player = _FakePlayer()

    async def go():
        sink = PipeWireAudioSink(player, wav_dir=tmp_path)
        await sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(400)))
        await sink.close()

    asyncio.run(go())
    assert tmp_path.is_dir()


# --- both adapters through the controller ----------------------------------


def test_the_two_adapters_fit_together_under_the_controller(tmp_path: Path) -> None:
    """The point of the milestone: the existing route, driven by the new queue,
    with no piece of it knowing about the other."""
    payload = _pcm(24_000)
    provider, _ = _provider(pcm=payload, chunk_seconds=0.25, wav_dir=tmp_path / "tts")
    player = _FakePlayer()
    sink = PipeWireAudioSink(player, wav_dir=tmp_path / "sink")

    async def go():
        await provider.load()
        controller = VoicePlaybackController(provider, sink)
        result = await controller.speak(TTSRequest(text="Alles zusammen."))
        await controller.shutdown()
        return result

    result = asyncio.run(go())

    assert result.error == ""
    assert result.cancelled is False
    assert result.chunks == 4
    assert b"".join(frames for frames, _rate, _channels in player.frames) == payload
    assert list((tmp_path / "sink").glob("*.wav")) == []


def test_a_dead_service_surfaces_as_a_result_error_not_an_exception(tmp_path: Path) -> None:
    provider, _ = _provider(raises=TtsError("TTS-Dienst nicht erreichbar"))
    player = _FakePlayer()
    sink = PipeWireAudioSink(player, wav_dir=tmp_path)

    async def go():
        await provider.load()
        controller = VoicePlaybackController(provider, sink)
        result = await controller.speak(TTSRequest(text="Niemand da."))
        await controller.shutdown()
        return result

    result = asyncio.run(go())
    assert "nicht erreichbar" in result.error
    assert player.paths == []


# --- import hygiene ---------------------------------------------------------


def test_importing_the_adapters_pulls_in_no_gpu_no_gtk_and_starts_no_process() -> None:
    """An adapter that boots CUDA or spawns pw-play at import time would make
    every `import kiki` in a test suite cost a device."""
    probe = """
import sys
started = []
sys.addaudithook(
    lambda event, args: started.append(event)
    if event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"}
    else None
)
import kiki.voice.tts.adapters  # noqa: F401
forbidden = [name for name in ("torch", "gi", "gi.repository", "numpy") if name in sys.modules]
assert not forbidden, forbidden
assert not started, started
print("sauber")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "sauber" in result.stdout


def test_the_tts_package_itself_stays_free_of_http() -> None:
    """`kiki.voice.tts` is imported by the UI, so the adapters are deliberately
    not re-exported from it."""
    probe = """
import sys
import kiki.voice.tts  # noqa: F401
assert "httpx" not in sys.modules, "httpx"
assert "kiki.voice.tts.adapters" not in sys.modules, "adapters"
print("sauber")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


# --- the director seam ------------------------------------------------------


def test_the_director_default_still_recognises_only_the_old_error() -> None:
    from kiki.voice.director import service_is_down

    assert service_is_down(TtsError("TTS-Dienst nicht erreichbar")) is True
    assert service_is_down(TtsError("TTS-Fehler: 500")) is False
    assert service_is_down(RuntimeError("irgendwas")) is False


def test_the_director_default_also_understands_the_adapter_error() -> None:
    from kiki.voice.director import service_is_down

    assert service_is_down(TTSError("weg", code="unreachable")) is True
    assert service_is_down(TTSError("langsam", code="timeout")) is True
    assert service_is_down(TTSError("ein Satz kaputt", code="service")) is False


def test_the_director_accepts_an_injected_predicate(tmp_path: Path) -> None:
    """The seam exists so a later milestone can swap the route without editing
    the director; today it must change nothing."""
    from kiki.voice.director import SpeechDirector

    events: list[str] = []

    async def _fail(_text: str, _dest: Path) -> Path:
        raise RuntimeError("nach Hausdefinition tot")

    director = SpeechDirector(
        synthesize=_fail,
        player=_FakePlayer(),
        submit=_sync_submit,
        wav_dir=tmp_path,
        on_idle=lambda: events.append("idle"),
        on_error=lambda _exc: events.append("error"),
        service_down=lambda exc: isinstance(exc, RuntimeError),
    )
    director.say("Hallo")

    assert events == ["error", "idle"]
    assert director.active is False


class _FakeHandle:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def _sync_submit(coro, *, on_success=None, on_error=None, on_complete=None) -> _FakeHandle:
    handle = _FakeHandle()
    try:
        result = asyncio.run(coro)
    except Exception as exc:
        if on_error is not None:
            on_error(exc)
        if on_complete is not None:
            on_complete()
        return handle
    if on_success is not None:
        on_success(result)
    if on_complete is not None:
        on_complete()
    return handle


# --- the lifetime of the working directory itself ---------------------------
#
# Owner:     whichever adapter created it. A wav_dir handed in by the caller is
#            never owned and never removed.
# Lifetime:  first synthesise/play until unload()/close(), the adapter being
#            garbage-collected, or interpreter exit — whichever comes first.
# The tests below walk every one of those exits.


def _dirs(root: Path, prefix: str) -> list[Path]:
    return sorted(root.glob(f"{prefix}*"))


def test_the_provider_creates_its_directory_only_when_it_needs_one(temp_root) -> None:
    provider, _ = _provider()
    asyncio.run(provider.load())

    assert _dirs(temp_root, "kiki-tts-") == []  # a health probe writes nothing


def test_the_working_directory_is_private(temp_root) -> None:
    """0700 on the directory, 0600 on the files inside it."""
    import stat

    provider, _ = _provider()
    _collect(provider, TTSRequest(text="Kurz."))
    created = _dirs(temp_root, "kiki-tts-")

    assert len(created) == 1
    assert stat.S_IMODE(created[0].stat().st_mode) == 0o700
    assert created[0].name.startswith("kiki-tts-")
    asyncio.run(provider.unload())


def test_success_leaves_no_directory_behind(temp_root) -> None:
    provider, _ = _provider()
    _collect(provider, TTSRequest(text="Alles gut."))
    asyncio.run(provider.unload())

    assert _dirs(temp_root, "kiki-tts-") == []


def test_a_provider_error_leaves_no_directory_behind(temp_root) -> None:
    provider, _ = _provider(raises=TtsError("TTS-Dienst nicht erreichbar"))
    with pytest.raises(TTSError):
        _collect(provider, TTSRequest(text="Weg."))
    asyncio.run(provider.unload())

    assert _dirs(temp_root, "kiki-tts-") == []


def test_a_playback_error_leaves_no_sink_directory_behind(temp_root) -> None:
    player = _FakePlayer(fail_with="kein Gerät")

    async def go():
        sink = PipeWireAudioSink(player)
        with pytest.raises(TTSError):
            await sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(400)))
        await sink.close()

    asyncio.run(go())
    assert _dirs(temp_root, "kiki-sink-") == []


def test_a_cancelled_task_leaves_no_sink_directory_behind(temp_root) -> None:
    player = _FakePlayer(auto_eos=False)

    async def go():
        sink = PipeWireAudioSink(player)
        task = asyncio.create_task(
            sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(48_000)))
        )
        await _until_playing(player)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await sink.close()

    asyncio.run(go())
    assert _dirs(temp_root, "kiki-sink-") == []


def test_stop_then_close_leaves_no_sink_directory_behind(temp_root) -> None:
    player = _FakePlayer(auto_eos=False)

    async def go():
        sink = PipeWireAudioSink(player)
        task = asyncio.create_task(
            sink.play(AudioChunk(request_id="r", sequence=0, pcm=_pcm(48_000)))
        )
        await _until_playing(player)
        await sink.stop()
        await task
        await sink.close()

    asyncio.run(go())
    assert _dirs(temp_root, "kiki-sink-") == []


def test_a_controller_shutdown_clears_the_sink_directory(temp_root) -> None:
    provider, _ = _provider(pcm=_pcm(24_000), chunk_seconds=0.25)
    player = _FakePlayer()
    sink = PipeWireAudioSink(player)

    async def go():
        await provider.load()
        controller = VoicePlaybackController(provider, sink)
        await controller.speak(TTSRequest(text="Und tschüss."))
        await controller.shutdown()
        await provider.unload()

    asyncio.run(go())
    assert _dirs(temp_root, "kiki-sink-") == []
    assert _dirs(temp_root, "kiki-tts-") == []


def test_a_forgotten_unload_is_cleaned_up_when_the_adapter_is_collected(temp_root) -> None:
    """The leak itself: mkdtemp had no owner, so a provider that was simply
    dropped left its directory in /tmp for good."""
    import gc

    def _use() -> None:
        provider, _ = _provider()
        _collect(provider, TTSRequest(text="Und weg."))
        assert len(_dirs(temp_root, "kiki-tts-")) == 1
        # no unload(), no close() — exactly what production did

    _use()
    gc.collect()

    assert _dirs(temp_root, "kiki-tts-") == []


def test_repeated_use_does_not_grow_the_temp_root(temp_root) -> None:
    """The shape of the reported bug: hundreds of empty kiki-tts-* directories.
    Twenty cycles must leave the root exactly as they found it."""
    import gc

    for _ in range(20):
        provider, _ = _provider()
        _collect(provider, TTSRequest(text="Immer wieder."))
        asyncio.run(provider.unload())
    gc.collect()

    assert list(temp_root.iterdir()) == []


def test_an_unloaded_provider_takes_a_fresh_directory_next_time(temp_root) -> None:
    provider, _ = _provider()
    _collect(provider, TTSRequest(text="Erste Runde."))
    first = _dirs(temp_root, "kiki-tts-")[0]
    asyncio.run(provider.unload())

    asyncio.run(provider.load())
    _collect(provider, TTSRequest(text="Zweite Runde."))
    second = _dirs(temp_root, "kiki-tts-")[0]

    assert second != first
    asyncio.run(provider.unload())
    assert _dirs(temp_root, "kiki-tts-") == []


def test_a_caller_supplied_provider_directory_is_never_removed(tmp_path: Path) -> None:
    """Deleting a directory the caller owns would take the director's cache with
    it the day both share one."""
    borrowed = tmp_path / "borrowed"
    borrowed.mkdir()
    provider, _ = _provider(wav_dir=borrowed)
    _collect(provider, TTSRequest(text="Geliehen."))
    asyncio.run(provider.unload())

    assert borrowed.is_dir()
    assert list(borrowed.iterdir()) == []


def test_the_directory_is_removed_when_the_process_ends(tmp_path: Path, temp_root) -> None:
    """The ordered-shutdown case: a service that exits without calling unload()
    must still leave nothing behind.

    TMPDIR points the child at the same injected root, so nothing touches the
    real /tmp here either.
    """
    root = temp_root
    probe = """
import asyncio, sys, wave
from pathlib import Path
from kiki.voice.tts.adapters import ServiceTTSProvider
from kiki.voice.tts.models import TTSRequest
from kiki.voice.tts_client import TtsHealth

async def _synth(base_url, text, *, dest, language, speaker, timeout):
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(b"\\x01\\x02" * 2400)
    return Path(dest)

async def _health(base_url, **k):
    return TtsHealth(ok=True, ready=True, detail="")

async def main():
    p = ServiceTTSProvider(synthesize=_synth, health=_health)
    await p.load()
    async for _c in p.synthesize(TTSRequest(text="Tschüss.")):
        pass
    print(p._wav_dir)          # deliberately no unload()

asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
             "TMPDIR": str(root)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    created = Path(result.stdout.strip())

    assert created.name.startswith("kiki-tts-")
    assert created.parent == root          # it really went to the injected root
    assert not created.exists()            # and the process took it with it
    assert list(root.iterdir()) == []
