"""The local STT service answers health and transcribe, and stays bounded."""

from __future__ import annotations

import importlib.util
import io
import json
import wave
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "services" / "kiki-stt" / "kiki_stt_server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("kiki_stt_server", SERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_wav(seconds: float = 0.1, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


@pytest.fixture
def server():
    mod = _load_server()
    handler = mod.SttHandler
    handler.transcriber = mod.DummyTranscriber()
    try:
        httpd = mod.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    except PermissionError:
        pytest.skip("Ausführungs-Sandbox verbietet selbst lokale Loopback-Sockets")
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield mod, HTTPConnection(host, port, timeout=5)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_health_and_transcribe_round_trip(server) -> None:
    _mod, conn = server
    conn.request("GET", "/health")
    response = conn.getresponse()
    payload = json.loads(response.read())
    assert response.status == 200
    assert payload["ok"] is True
    assert payload["dummy"] is True
    assert payload["ready"] is True

    conn.request(
        "POST",
        "/v1/transcribe",
        body=_tiny_wav(),
        headers={"Content-Type": "audio/wav"},
    )
    response = conn.getresponse()
    payload = json.loads(response.read())
    assert response.status == 200
    assert payload["ok"] is True
    assert payload["text"] == "dies ist ein test des spracherkennungsdienstes"


def test_transcribe_refuses_oversized_audio(server) -> None:
    mod, conn = server
    conn.request(
        "POST",
        "/v1/transcribe",
        body=_tiny_wav(seconds=mod.MAX_AUDIO_SECONDS + 5),
        headers={"Content-Type": "audio/wav"},
    )
    response = conn.getresponse()
    payload = json.loads(response.read())
    assert response.status == 413
    assert payload["ok"] is False


def test_unknown_path_and_garbage_body(server) -> None:
    _mod, conn = server
    conn.request("GET", "/nope")
    assert conn.getresponse().status == 404
    conn.request("POST", "/v1/transcribe", body=b"not a wav")
    response = conn.getresponse()
    assert response.status == 400
    payload = json.loads(response.read())
    assert payload["ok"] is False


def test_server_refuses_non_loopback() -> None:
    mod = _load_server()
    with pytest.raises(SystemExit):
        mod._bind_host("0.0.0.0")


# --- engine selection ----------------------------------------------------------


def test_qwen_language_mapping() -> None:
    mod = _load_server()
    assert mod._qwen_language("de") == "German"
    assert mod._qwen_language("DE") == "German"
    assert mod._qwen_language("en") == "English"
    assert mod._qwen_language("auto") is None
    assert mod._qwen_language("") is None
    # Unbekannte Codes werden als Name durchgereicht (Title-Case).
    assert mod._qwen_language("Klingon") == "Klingon"


def test_engine_defaults_and_choice() -> None:
    mod = _load_server()
    args = mod.parse_args([])
    assert args.engine == "auto"
    args = mod.parse_args(["--engine", "qwen"])
    assert args.engine == "qwen"
