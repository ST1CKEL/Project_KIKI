"""Voice-first stack: honesty, telemetry, IPC, policy mapping, no silent fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from kiki.audio.vad import EnergySpeechGate, build_speech_gate
from kiki.audio.wake import MissingWakeSpotter, build_wake_spotter, default_model_path
from kiki.config.runtime import load_runtime
from kiki.ipc.paths import runtime_dir, socket_path
from kiki.ipc.protocol import ProtocolError, dumps, loads
from kiki.orchestrator.confirm import confirmation_prompt, parse_spoken_verdict
from kiki.orchestrator.gpu import GpuResourceManager
from kiki.orchestrator.health import HealthState, SubsystemHealth
from kiki.orchestrator.telemetry import TurnTelemetry
from kiki.orchestrator.tools import SecurityClass, security_class
from kiki.orchestrator.vision import VisionError, apply_action
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec
from kiki.tts.chunker import StreamingTtsChunker
from kiki.tts.engine import STANDARD_PHRASES
from kiki.tts.normalizer import GermanTextNormalizer


def test_runtime_dir_uses_xdg_and_never_hardcodes_uid_1000(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.delenv("KIKI_RUNTIME_DIR", raising=False)
    path = runtime_dir()
    assert path == tmp_path / "run" / "kiki"
    assert "1000" not in str(path)
    assert socket_path("audio").name == "audio.sock"


def test_json_protocol_roundtrip() -> None:
    line = dumps({"event": "wake_detected", "turn_id": "t1"})
    assert line.endswith(b"\n")
    msg = loads(line)
    assert msg["event"] == "wake_detected"
    with pytest.raises(ProtocolError):
        loads(b"not-json")


def test_load_runtime_packaged_defaults() -> None:
    cfg = load_runtime(Path("does-not-exist-runtime.toml"))
    assert "large-v3-turbo" in cfg.stt.model_name
    assert cfg.stt.allow_cpu is False
    assert cfg.tts.allow_espeak is False
    assert cfg.vad.model == "silero_vad"
    assert cfg.wake.engine == "openwakeword"
    assert cfg.vad.min_silence_ms <= 300


def test_silero_without_onnx_is_not_ready(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    from kiki.audio.vad import SileroSpeechGate

    gate = SileroSpeechGate(model_path=tmp_path / "missing.onnx")
    assert gate.ready is False
    assert "Silero" in gate.error
    assert gate.is_speech(b"\x00\x00" * 512) is False


def test_energy_vad_is_explicit_and_labelled_degraded() -> None:
    gate = build_speech_gate("energy")
    assert isinstance(gate, EnergySpeechGate)
    assert "Testmodus" in gate.error or "energy" in gate.backend
    silence = b"\x00\x00" * 512
    assert gate.is_speech(silence) is False


def test_missing_wake_model_is_not_ready(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    spotter = build_wake_spotter(model_name="kiki")
    assert spotter.ready is False
    assert "Vosk" not in (spotter.error or "") or "nicht" in spotter.error.lower()
    assert "kiki.onnx" in default_model_path().name or default_model_path().suffix in {".onnx", ".tflite"}
    assert isinstance(spotter, MissingWakeSpotter) or spotter.ready is False
    assert spotter.feed(b"\x00\x00" * 1280) is False


def test_stt_service_source_does_not_import_vosk() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "kiki" / "stt" / "service.py"
    text = src.read_text(encoding="utf-8")
    assert "import vosk" not in text.lower()
    assert "from vosk" not in text.lower()


def test_tts_service_source_rejects_espeak() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "kiki" / "tts" / "service.py"
    text = src.read_text(encoding="utf-8")
    assert "allow_espeak is set — ignored" in text or "espeak-ng is never" in text
    engine = Path(__file__).resolve().parents[1] / "src" / "kiki" / "tts" / "engine.py"
    etext = engine.read_text(encoding="utf-8")
    assert "espeak-ng ist kein gültiger Wert" in etext


def test_chunker_sentence_and_reset() -> None:
    chunker = StreamingTtsChunker(max_chunk_chars=80, semantic_min_words=4, stream_timeout_ms=50)
    got = chunker.push_token("Hallo Welt. ")
    assert any("Hallo" in c for c in got)
    leftover = chunker.flush()
    chunker.reset()
    assert chunker.flush() == []
    assert leftover == [] or isinstance(leftover, list)


def test_normalizer_speaks_numbers_and_strips_code() -> None:
    n = GermanTextNormalizer()
    out = n.normalize("SSH auf Port 22, siehe `rm -rf /`.")
    assert "S-S-H" in out
    assert "zweiundzwanzig" in out
    assert "rm -rf" not in out


def test_telemetry_ack_and_ttfa() -> None:
    t = TurnTelemetry(turn_id="t")
    t.t_wake = 1.0
    t.t_ack = 1.12
    t.t_eos = 2.0
    t.t_stt_final = 2.08
    t.t_llm_first_token = 2.20
    t.t_playback_start = 2.35
    t.calculate()
    assert t.ack_ms == pytest.approx(120.0)
    assert t.stt_latency_ms == pytest.approx(80.0)
    assert t.ttfa_ms == pytest.approx(350.0)


def test_gpu_unknown_refuses_allocation() -> None:
    mgr = GpuResourceManager(safety_margin_mb=2048)
    status = mgr.get_memory_status()
    if not status.available:
        ok, reason = mgr.can_allocate_vram(100)
        assert ok is False
        assert reason


def test_health_spoken_fault_is_first_person() -> None:
    h = SubsystemHealth()
    assert h.overall() is HealthState.HEALTHY
    h.audio = {"ready": False, "status": "failed"}
    assert "Mikrofon" in h.spoken_fault()
    assert h.overall() is HealthState.DEGRADED_AUDIO
    h.audio = {"ready": True, "status": "healthy", "wake_ready": True}
    h.stt = {"ready": False, "status": "failed"}
    assert "Spracherkennung" in h.spoken_fault()


def test_spoken_verdict_does_not_guess_yes() -> None:
    assert parse_spoken_verdict("ja") is True
    assert parse_spoken_verdict("nein") is False
    assert parse_spoken_verdict("vielleicht später") is None
    prompt = confirmation_prompt("Datei löschen", "Löscht notes.txt", "write")
    assert "Ja oder nein" in prompt
    assert "folgenschwer" in prompt


def test_security_class_maps_risk_without_new_executor() -> None:
    def handler(_p):
        return {}

    read = ToolSpec(
        name="x",
        title="x",
        description="x",
        risk=RiskLevel.READ,
        parameters={},
        handler=handler,
        effect="e",
        auto_allow=True,
    )
    write = ToolSpec(
        name="y",
        title="y",
        description="y",
        risk=RiskLevel.WRITE,
        parameters={},
        handler=handler,
        effect="e",
        auto_allow=False,
    )
    assert security_class(read) is SecurityClass.READ_ONLY
    assert security_class(write) is SecurityClass.DESTRUCTIVE


def test_orchestrator_tools_do_not_unlink_files() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "kiki" / "orchestrator" / "tools.py"
    text = src.read_text(encoding="utf-8")
    assert "unlink" not in text
    assert "remove_file" not in text
    assert "ToolGateway" in text


def test_vision_rejects_secret_typing_without_ydotool() -> None:
    with pytest.raises(VisionError):
        apply_action({"action": "type", "text": "export API_KEY=secret"})
    with pytest.raises(VisionError):
        apply_action({"action": "key", "key": "ctrl+alt+f3"})


def test_instant_phrases_exist_for_ack() -> None:
    assert "ack" in STANDARD_PHRASES
    assert STANDARD_PHRASES["ack"].endswith("?")


def test_setup_wakeword_skips_tflite_runtime() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "setup-wakeword.sh"
    text = script.read_text(encoding="utf-8")
    assert "--no-deps" in text
    assert "openwakeword==0.6.0" in text
    assert "tflite-runtime" in text  # explained as something we refuse to install
    assert 'pip install -U "openwakeword' not in text


def test_systemd_units_are_portable() -> None:
    root = Path(__file__).resolve().parents[1] / "systemd" / "user"
    for name in ("kiki-audio", "kiki-stt", "kiki-tts", "kiki-orchestrator", "kiki-pet"):
        text = (root / f"{name}.service").read_text(encoding="utf-8")
        assert "/home/martin" not in text
        assert "Restart=" in text
        assert "voice.env" in text or "PYTHONUNBUFFERED" in text


def test_audio_daemon_turn_with_fakes() -> None:
    from dataclasses import replace

    from kiki.audio.daemon import AudioDaemon

    class FakeCapture:
        ready = True
        error = ""

        def read(self, timeout_ms: int = 40) -> bytes:  # noqa: ARG002
            return b""

        def close(self) -> None:
            return None

    class FakeVad:
        ready = True
        error = ""
        backend = "silero_vad"

        def __init__(self) -> None:
            self.n = 0

        def is_speech(self, pcm: bytes) -> bool:  # noqa: ARG002
            self.n += 1
            return self.n < 8

    class FakeWake:
        ready = True
        error = ""
        backend = "openwakeword"

        def feed(self, pcm: bytes) -> bool:  # noqa: ARG002
            return False

    events: list[dict] = []
    cfg = replace(load_runtime(), socket_dir=runtime_dir())
    daemon = AudioDaemon(
        cfg,
        capture=FakeCapture(),  # type: ignore[arg-type]
        vad=FakeVad(),  # type: ignore[arg-type]
        wake=FakeWake(),  # type: ignore[arg-type]
        on_event=events.append,
    )
    daemon.listen_cue = False
    turn_id = daemon.trigger_turn(source="test")
    frame = b"\x00\x00" * (cfg.audio.sample_rate * cfg.audio.frame_ms // 1000)
    for _ in range(20):
        daemon.process_frame(frame)
    kinds = [e.get("event") for e in events]
    assert "wake_detected" in kinds
    assert "speech_ended" in kinds
    ended = next(e for e in events if e.get("event") == "speech_ended")
    assert ended["turn_id"] == turn_id
    path = Path(str(ended["audio_path"]))
    assert path.is_file()
    path.unlink(missing_ok=True)
