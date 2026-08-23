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


# --- the WAV route, pinned before the streaming route was added -------------
#
# The HTTP protocol gate: these had to be green before any global HTTP change
# could be considered. They stay as the guard that the WAV route is exactly
# what it was.


def _dummy_server():
    mod = _load_server()
    mod.TtsHandler.synthesizer = mod.DummySynthesizer()
    mod.TtsHandler.stream_engine = None
    try:
        httpd = mod.ThreadingHTTPServer(("127.0.0.1", 0), mod.TtsHandler)
    except PermissionError:
        pytest.skip("Ausführungs-Sandbox verbietet selbst lokale Loopback-Sockets")
    Thread(target=httpd.serve_forever, daemon=True).start()
    return mod, httpd


def test_the_wav_response_is_fully_framed() -> None:
    """Status, type, length and a body that matches the announced length."""
    _mod, httpd = _dummy_server()
    try:
        conn = HTTPConnection(*httpd.server_address[:2], timeout=5)
        conn.request(
            "POST",
            "/v1/synthesize",
            body=json.dumps({"text": "Hallo KIKI", "language": "German", "speaker": "Serena"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = response.read()
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert response.status == 200
    assert response.getheader("Content-Type") == "audio/wav"
    assert response.getheader("Cache-Control") == "no-store"
    assert int(response.getheader("Content-Length")) == len(body)
    assert body[:4] == b"RIFF"
    assert body[8:12] == b"WAVE"
    # No streaming header may appear on the file route.
    assert response.getheader("X-KIKI-Streaming") is None


def test_repeated_wav_requests_over_one_client_connection() -> None:
    """Whatever the server offers for connection reuse, three requests in a row
    on one client object must all come back whole."""
    _mod, httpd = _dummy_server()
    try:
        conn = HTTPConnection(*httpd.server_address[:2], timeout=5)
        results = []
        for _ in range(3):
            conn.request(
                "POST",
                "/v1/synthesize",
                body=json.dumps({"text": "Hallo KIKI"}),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            body = response.read()
            results.append((response.status, int(response.getheader("Content-Length")), len(body)))
        conn.request("GET", "/health")
        health = conn.getresponse()
        payload = json.loads(health.read())
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert [status for status, _a, _b in results] == [200, 200, 200]
    assert all(announced == actual for _s, announced, actual in results)
    assert health.status == 200
    assert payload["ok"] is True


def test_the_wav_route_is_unchanged_without_a_streaming_engine() -> None:
    """The default deployment has no engine; nothing about it may differ."""
    _mod, httpd = _dummy_server()
    try:
        conn = HTTPConnection(*httpd.server_address[:2], timeout=5)
        conn.request("GET", "/health")
        payload = json.loads(conn.getresponse().read())
        conn.request(
            "POST", "/v1/synthesize", body=json.dumps({"text": "Hallo"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = response.read()
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert payload["streaming"] is False
    assert response.status == 200
    assert body[:4] == b"RIFF"


def test_the_handler_still_speaks_http_1_0() -> None:
    """Raising the handler to HTTP/1.1 was measured and rejected: with
    keep-alive live, the four POST paths that answer before reading the request
    body leave those bytes in the socket, and the next request is parsed out of
    the leftovers — a follow-up GET /health came back 400 instead of 200."""
    mod = _load_server()
    assert mod.TtsHandler.protocol_version == "HTTP/1.0"


def test_health_reports_why_streaming_is_off() -> None:
    """A client needs to know *before* an answer whether the PCM route exists,
    and the reason must be a category — never a path or an internal message."""
    mod, httpd = _dummy_server()

    class _Refusing:
        available = False
        reason = "runtime_incompatible"

    mod.TtsHandler.stream_engine = _Refusing()
    try:
        conn = HTTPConnection(*httpd.server_address[:2], timeout=5)
        conn.request("GET", "/health")
        payload = json.loads(conn.getresponse().read())
        conn.close()
    finally:
        mod.TtsHandler.stream_engine = None
        httpd.shutdown()
        httpd.server_close()

    assert payload["streaming"] is False
    assert payload["streaming_reason"] == "runtime_incompatible"
    assert "/" not in payload["streaming_reason"]


def test_health_reports_no_reason_when_streaming_works() -> None:
    mod, httpd = _dummy_server()

    class _Ready:
        available = True
        reason = None

    mod.TtsHandler.stream_engine = _Ready()
    try:
        conn = HTTPConnection(*httpd.server_address[:2], timeout=5)
        conn.request("GET", "/health")
        payload = json.loads(conn.getresponse().read())
        conn.close()
    finally:
        mod.TtsHandler.stream_engine = None
        httpd.shutdown()
        httpd.server_close()

    assert payload["streaming"] is True
    assert payload["streaming_reason"] is None


def test_health_names_a_missing_engine() -> None:
    _mod, httpd = _dummy_server()
    try:
        conn = HTTPConnection(*httpd.server_address[:2], timeout=5)
        conn.request("GET", "/health")
        payload = json.loads(conn.getresponse().read())
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert payload["streaming"] is False
    assert payload["streaming_reason"] == "no_engine"
