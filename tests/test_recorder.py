from __future__ import annotations

import types
from pathlib import Path

import gi.repository

from kiki.voice.recorder import AudioRecorder


def test_stop_waits_for_eos_before_forcing_pipeline_null(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []

    class FakeBus:
        def timed_pop_filtered(self, timeout: int, kinds: int):
            assert timeout == 2
            assert kinds == 3
            events.append("wait-eos")
            return types.SimpleNamespace(type=1)

    class FakePipeline:
        def send_event(self, _event: object) -> None:
            events.append("send-eos")

        def get_bus(self) -> FakeBus:
            return FakeBus()

        def set_state(self, state: str) -> None:
            assert state == "null"
            events.append("set-null")

    fake_gst = types.SimpleNamespace(
        SECOND=1,
        Event=types.SimpleNamespace(new_eos=lambda: object()),
        MessageType=types.SimpleNamespace(EOS=1, ERROR=2),
        State=types.SimpleNamespace(NULL="null"),
    )
    monkeypatch.setattr(gi.repository, "Gst", fake_gst, raising=False)
    recorder = AudioRecorder()
    recorder._pipeline = FakePipeline()
    recorder._path = tmp_path / "take.wav"

    assert recorder.stop() == tmp_path / "take.wav"
    assert events == ["send-eos", "wait-eos", "set-null"]
