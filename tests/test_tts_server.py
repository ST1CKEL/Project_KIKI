from __future__ import annotations

import importlib.util
import json
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "services" / "qwen3-tts" / "kiki_tts_server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("kiki_tts_server", SERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dummy_server_health_and_wav() -> None:
    mod = _load_server()
    handler = mod.TtsHandler
    handler.synthesizer = mod.DummySynthesizer()
    try:
        httpd = mod.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    except PermissionError:
        pytest.skip("Ausführungs-Sandbox verbietet selbst lokale Loopback-Sockets")
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/health")
        response = conn.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["ok"] is True
        assert payload["dummy"] is True
        assert "Serena" in payload["speakers"]

        body = json.dumps({"text": "Hallo KIKI", "language": "German", "speaker": "Serena"})
        conn.request("POST", "/v1/synthesize", body=body, headers={"Content-Type": "application/json"})
        audio = conn.getresponse()
        wav = audio.read()
        assert audio.status == 200
        assert wav[:4] == b"RIFF"
        assert len(wav) > 1000

        conn.request("POST", "/v1/synthesize", body=json.dumps({"text": ""}), headers={"Content-Type": "application/json"})
        empty = conn.getresponse()
        assert empty.status == 400
        empty.read()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_server_refuses_non_loopback() -> None:
    mod = _load_server()
    with pytest.raises(SystemExit):
        mod._bind_host("0.0.0.0")


def test_speaker_and_language_match_case_insensitively() -> None:
    """KIKI's config says "Serena"/"German"; the model reports lower-case.

    A plain membership test missed every time and fell through to speakers[0] —
    "aiden", a male voice — while the language stayed at an invalid value. KIKI
    therefore spoke with the wrong voice on a correctly configured system.
    """
    resolve = _load_server().QwenSynthesizer._resolve
    speakers = ["aiden", "dylan", "serena", "vivian"]
    languages = ["auto", "english", "german"]

    assert resolve("Serena", speakers, "Serena") == "serena"
    assert resolve("serena", speakers, "Serena") == "serena"
    assert resolve("VIVIAN", speakers, "Serena") == "vivian"
    assert resolve("German", languages, "German") == "german"

    # An unknown name falls back to the configured default, not to the first
    # entry, so a typo cannot silently switch the voice's gender.
    assert resolve("Gibtsnicht", speakers, "Serena") == "serena"
    assert resolve("", speakers, "Serena") == "serena"
    # Only when the fallback is unusable does the first entry win.
    assert resolve("x", speakers, "auch-nicht-da") == "aiden"
    assert resolve("x", [], "y") == "x"
