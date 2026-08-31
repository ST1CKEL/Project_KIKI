"""Closed-WAV playback: sample rate, queue, paplay, start/finish callbacks."""

from __future__ import annotations

import time
import wave
from pathlib import Path

from kiki.tts.playback import MAX_QUEUE_SECONDS, PipeWirePcmPlayback, _pcm_to_wav


class _FakeProc:
    def __init__(self, args, **kwargs) -> None:
        self.args = list(args)
        self.returncode = 0
        self._wav = Path(str(args[-1]))
        self.rate = 0
        if self._wav.is_file():
            with wave.open(str(self._wav), "rb") as wav:
                self.rate = wav.getframerate()
                self.nframes = wav.getnframes()
        else:
            self.nframes = 0

    def communicate(self, timeout=None):
        return b"", b""

    def wait(self, timeout=None):
        return 0

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def test_pcm_wav_uses_engine_rate(tmp_path: Path) -> None:
    dest = tmp_path / "t.wav"
    pcm = b"\x00\x00" * 160
    _pcm_to_wav(pcm, 16000, dest)
    with wave.open(str(dest), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 160


def test_sole_chunk_is_never_dropped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("kiki.tts.playback.runtime_dir", lambda: tmp_path)
    monkeypatch.setattr("kiki.tts.playback.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("kiki.tts.playback._default_sink", lambda: "mock-sink")
    monkeypatch.setattr("kiki.tts.playback.threading.Thread.start", lambda self: None)
    pb = PipeWirePcmPlayback(sample_rate=16000)
    try:
        huge = b"\x00\x00" * int((MAX_QUEUE_SECONDS + 1.0) * 16000)
        pb.enqueue_pcm("only", huge, sample_rate=16000)
        extra = b"\x00\x00" * 8000
        pb.enqueue_pcm("only", extra, sample_rate=16000)
        queued = list(pb._queue)
        assert len(queued) == 1
        assert queued[0][0] == "only"
        assert len(queued[0][1]) == len(huge)
        assert pb.underrun_count == 1
    finally:
        pb.close()


def test_playback_invokes_paplay_and_callbacks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("kiki.tts.playback.runtime_dir", lambda: tmp_path)
    monkeypatch.setattr("kiki.tts.playback.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("kiki.tts.playback._default_sink", lambda: "mock-sink")
    calls: list[_FakeProc] = []

    def _spawn(args, **kwargs):
        proc = _FakeProc(args, **kwargs)
        calls.append(proc)
        return proc

    monkeypatch.setattr("kiki.tts.playback.subprocess.Popen", _spawn)
    started: list[str] = []
    finished: list[str] = []
    pb = PipeWirePcmPlayback(
        sample_rate=24000,
        on_audio_started=started.append,
        on_audio_finished=finished.append,
    )
    try:
        pcm = b"\x00\x01" * 320  # 16000 Hz, 20 ms if we pass 16000
        pb.enqueue_pcm("turn-a", pcm, sample_rate=16000)
        deadline = time.time() + 2.0
        while time.time() < deadline and not finished:
            time.sleep(0.02)
        assert calls, "player was never spawned"
        assert calls[0].args[0].endswith("paplay")
        assert any(a.startswith("--device=") for a in calls[0].args)
        assert calls[0].rate == 16000
        assert started == ["turn-a"]
        assert finished == ["turn-a"]
    finally:
        pb.close()
