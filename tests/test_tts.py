from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path

import httpx
import pytest

from kiki.config.settings import default_mapping, load_settings, settings_from_mapping
from kiki.runtime.async_bridge import AsyncBridge
from kiki.voice import tts_client
from kiki.voice.director import SpeechDirector
from kiki.voice.system_tts import synthesize_system_wav, system_tts_available
from kiki.voice.tts_client import TtsError, synthesize_wav
from kiki.voice.tts_text import flush_buffer, speakable, split_ready


def test_speakable_strips_markdown() -> None:
    text = speakable("**Hallo** `code` und [Link](https://example)\n```\nsecret\n```\nWelt.")
    assert "secret" not in text
    assert "code" not in text
    assert "Hallo" in text
    assert "Welt" in text
    assert "Link" in text


def test_split_ready_keeps_open_fence() -> None:
    ready, rest = split_ready("Vorher. ```python\nprint(1)")
    assert ready == []
    assert "```" in rest


def test_split_ready_sentences() -> None:
    ready, rest = split_ready("Hallo Welt. Wie geht es dir? Fast")
    assert ready == ["Hallo Welt.", "Wie geht es dir?"]
    assert rest.strip() == "Fast"
    assert flush_buffer(rest) == "Fast"


def test_tts_defaults_and_panic(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "missing.toml")
    assert settings.tts.enabled is True
    assert settings.tts.speaker == "Serena"
    assert settings.tts.language == "German"
    assert settings.tts.base_url.startswith("http://127.0.0.1")
    assert settings.tts.fallback_to_system is True
    assert settings.tts_allowed() is True
    settings.app.privacy_panic = True
    assert settings.tts_allowed() is False


def test_tts_unknown_speaker_falls_back() -> None:
    data = default_mapping()
    data["tts"]["speaker"] = "NotAVoice"
    data["tts"]["language"] = "Klingon"
    settings = settings_from_mapping(data)
    assert settings.tts.speaker == "Serena"
    assert settings.tts.language == "German"


def test_synthesize_connect_error() -> None:
    async def _run() -> None:
        with pytest.raises(TtsError, match="nicht erreichbar"):
            await synthesize_wav(
                "http://127.0.0.1:1",
                "Hallo",
                dest=Path("/tmp/kiki-tts-nope.wav"),
                timeout=0.2,
            )

    asyncio.run(_run())


def _wav_bytes(
    *,
    channels: int = 1,
    sample_width: int = 2,
    frame_rate: int = 24_000,
    frame_count: int = 240,
) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(frame_rate)
        wav.writeframes(b"\0" * channels * sample_width * frame_count)
    return output.getvalue()


def _mock_tts_client(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, headers={"Content-Type": "audio/wav"}, content=payload)
    )
    client = httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: client)


def test_synthesize_accepts_valid_pcm_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _wav_bytes()
    _mock_tts_client(monkeypatch, payload)
    dest = tmp_path / "voice.wav"

    result = asyncio.run(synthesize_wav("http://127.0.0.1:18765", "Hallo", dest=dest))

    assert result == dest
    assert dest.read_bytes() == payload


def test_synthesize_rejects_oversized_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _wav_bytes()
    _mock_tts_client(monkeypatch, payload)
    monkeypatch.setattr(tts_client, "MAX_WAV_BYTES", len(payload) - 1)
    dest = tmp_path / "voice.wav"

    with pytest.raises(TtsError, match="zu groß"):
        asyncio.run(synthesize_wav("http://127.0.0.1:18765", "Hallo", dest=dest))

    assert not dest.exists()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not a wave file" * 8, "RIFF/WAVE"),
        (_wav_bytes(frame_rate=4_000), "Abtastrate"),
        (_wav_bytes(channels=3), "Kanalzahl"),
        (_wav_bytes()[:-20], "unvollständige"),
    ],
)
def test_synthesize_rejects_invalid_wav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    message: str,
) -> None:
    _mock_tts_client(monkeypatch, payload)
    dest = tmp_path / "voice.wav"

    with pytest.raises(TtsError, match=message):
        asyncio.run(synthesize_wav("http://127.0.0.1:18765", "Hallo", dest=dest))

    assert not dest.exists()


class _FakePlayer:
    def __init__(self) -> None:
        self.played: list[Path] = []
        self._eos = None
        self.stopped = 0

    def play(self, path: Path, *, on_eos=None, on_error=None) -> None:
        self.played.append(path)
        self._eos = on_eos
        self._err = on_error

    def stop(self) -> None:
        self.stopped += 1
        self._eos = None

    def finish(self) -> None:
        callback = self._eos
        self._eos = None
        if callback is not None:
            callback()


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


def test_director_speaks_sentences_in_order(tmp_path: Path) -> None:
    player = _FakePlayer()
    events: list[str] = []

    async def _synth(text: str, dest: Path) -> Path:
        dest.write_text(text, encoding="utf-8")
        return dest

    director = SpeechDirector(
        synthesize=_synth,
        player=player,
        submit=_sync_submit,
        wav_dir=tmp_path,
        on_speaking=lambda: events.append("speaking"),
        on_idle=lambda: events.append("idle"),
    )
    director.begin()
    director.feed("Hallo Welt. Zweiter Satz!")
    director.flush()
    assert len(player.played) == 1
    assert player.played[0].read_text(encoding="utf-8") == "Hallo Welt."
    assert events == ["speaking"]
    player.finish()
    assert len(player.played) == 2
    assert player.played[1].read_text(encoding="utf-8") == "Zweiter Satz!"
    assert events[-1] != "idle"
    player.finish()
    assert events[-1] == "idle"
    assert director.active is False


def test_director_stop_returns_to_idle(tmp_path: Path) -> None:
    player = _FakePlayer()
    events: list[str] = []

    async def _synth(text: str, dest: Path) -> Path:
        dest.write_text(text, encoding="utf-8")
        return dest

    director = SpeechDirector(
        synthesize=_synth,
        player=player,
        submit=_sync_submit,
        wav_dir=tmp_path,
        on_idle=lambda: events.append("idle"),
    )
    director.say("Hallo")
    active_wav = player.played[0]
    assert active_wav.is_file()
    director.stop()

    assert events == ["idle"]
    assert director.active is False
    assert not active_wav.exists()


def test_director_stop_cancels_synthesis_and_removes_late_wav(tmp_path: Path) -> None:
    async def _run() -> None:
        bridge = AsyncBridge()
        bridge._loop = asyncio.get_running_loop()
        player = _FakePlayer()
        started = asyncio.Event()
        cancellation_seen = asyncio.Event()

        async def _synth(_text: str, dest: Path) -> Path:
            dest.write_bytes(b"in progress")
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                # Simulate a backend that writes once more while unwinding. The
                # director wrapper must still remove this late artifact.
                dest.write_bytes(b"stale after cancellation")
                cancellation_seen.set()
                raise

        director = SpeechDirector(
            synthesize=_synth,
            player=player,
            submit=bridge.submit,
            wav_dir=tmp_path,
        )
        director.say("Dieser Satz läuft noch.")
        await asyncio.wait_for(started.wait(), timeout=1)
        assert list(tmp_path.glob("*.wav"))

        director.stop()

        await asyncio.wait_for(cancellation_seen.wait(), timeout=1)
        await asyncio.sleep(0)
        assert list(tmp_path.glob("*.wav")) == []
        assert director.active is False
        assert player.played == []

    asyncio.run(_run())


def test_director_unreachable_service_returns_to_idle(tmp_path: Path) -> None:
    events: list[str] = []

    async def _fail(_text: str, _dest: Path) -> Path:
        raise TtsError("TTS-Dienst nicht erreichbar")

    director = SpeechDirector(
        synthesize=_fail,
        player=_FakePlayer(),
        submit=_sync_submit,
        wav_dir=tmp_path,
        on_idle=lambda: events.append("idle"),
        on_error=lambda _exc: events.append("error"),
    )
    director.say("Hallo")

    assert events == ["error", "idle"]
    assert director.active is False


@pytest.mark.skipif(not system_tts_available(), reason="espeak-ng is not installed")
def test_system_tts_creates_valid_wav(tmp_path: Path) -> None:
    dest = asyncio.run(synthesize_system_wav("Hallo, ich bin KIKI.", dest=tmp_path / "sample.wav"))

    with wave.open(str(dest), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 1000


def test_first_chunk_may_end_at_a_clause() -> None:
    """Synthesis runs slower than realtime, so the wait before KIKI starts
    talking is roughly the first chunk's length. Cutting it at a comma halves
    that silence without changing total throughput."""
    from kiki.voice.tts_text import split_ready

    text = "Der Speicherplatz ist ausreichend, es sind noch fünfhundert Gigabyte frei."
    first, rest = split_ready(text, first=True)
    assert first == ["Der Speicherplatz ist ausreichend,"]
    assert "fünfhundert" in rest

    # Later chunks keep whole sentences so the prosody stays natural.
    later, _ = split_ready(text, first=False)
    assert later == [text]


def test_a_clause_too_short_to_be_worth_it_is_not_split() -> None:
    from kiki.voice.tts_text import split_ready

    assert split_ready("Kurz: ja.", first=True)[0] == ["Kurz: ja."]
    assert split_ready("Ja, gern.", first=True)[0] == ["Ja, gern."]


def test_a_sentence_boundary_still_wins_when_it_comes_first() -> None:
    """No clause break before the first full stop, so nothing is cut early and
    the ordinary sentence path runs — which yields every finished sentence."""
    from kiki.voice.tts_text import split_ready

    text = "Ein Reverse Proxy nimmt Anfragen entgegen. Er kann auch TLS beenden, sagt man."
    chunks, _ = split_ready(text, first=True)
    assert chunks[0] == "Ein Reverse Proxy nimmt Anfragen entgegen."
    # The comma in the second sentence must not have split it.
    assert chunks[1] == "Er kann auch TLS beenden, sagt man."


def test_code_fences_still_hold_back_everything() -> None:
    from kiki.voice.tts_text import split_ready

    assert split_ready("Nutze ```bash\ndf -h", first=True) == ([], "Nutze ```bash\ndf -h")
