"""The PCM streaming contract. No GPU, no model, no audio hardware.

Everything below runs against `FakeStreamingEngine`, so what is under test is
the endpoint itself: validation, framing, the line between "error becomes an
HTTP status" and "error closes the connection", disconnect handling, and the
promise that only one generation runs at a time.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import time
from contextlib import contextmanager
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "services" / "qwen3-tts" / "kiki_tts_server.py"
STREAM = ROOT / "services" / "qwen3-tts" / "streaming_http.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sh = _load(STREAM, "streaming_http")


def _server_module():
    return _load(SERVER, "kiki_tts_server")


@contextmanager
def running(engine=None, *, synthesizer=None):
    mod = _server_module()
    mod.TtsHandler.synthesizer = synthesizer if synthesizer is not None else mod.DummySynthesizer()
    mod.TtsHandler.stream_engine = engine
    mod.TtsHandler.stream_lock = threading.Lock()
    try:
        httpd = mod.ThreadingHTTPServer(("127.0.0.1", 0), mod.TtsHandler)
    except PermissionError:
        pytest.skip("Ausführungs-Sandbox verbietet selbst lokale Loopback-Sockets")
    Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[:2]
    finally:
        httpd.shutdown()
        httpd.server_close()


def _post(address, path, payload, *, timeout=10):
    conn = HTTPConnection(*address, timeout=timeout)
    conn.request(
        "POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    return conn, conn.getresponse()


def _stream(address, payload=None, **kwargs):
    body = {"text": "Hallo Martin.", "language": "German", "speaker": "Serena"}
    body.update(payload or {})
    return _post(address, "/v1/synthesize/stream", body, **kwargs)


def _raw_stream(address, payload=None):
    """Open the stream over a bare socket.

    `HTTPConnection.close()` is not a disconnect: the response holds a
    `makefile()` reference to the same descriptor, so the socket stays open and
    the server keeps happily writing. Only a real close tests a real departure.
    """
    body = {"text": "Hallo Martin.", "language": "German", "speaker": "Serena"}
    body.update(payload or {})
    encoded = json.dumps(body).encode()
    raw = socket.create_connection(address, timeout=10)
    raw.sendall(
        b"POST /v1/synthesize/stream HTTP/1.0\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(encoded)).encode() + b"\r\n\r\n" + encoded
    )
    return raw


def _read_at_least(raw, count):
    got = b""
    while len(got) < count:
        piece = raw.recv(65536)
        if not piece:
            break
        got += piece
    return got


def _await(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# --- availability -----------------------------------------------------------


def test_without_an_engine_the_route_answers_503() -> None:
    with running(engine=None) as address:
        conn, response = _stream(address)
        payload = json.loads(response.read())
        conn.close()
    assert response.status == 503
    assert payload["streaming"] is False


def test_an_unavailable_engine_answers_503() -> None:
    with running(sh.FakeStreamingEngine(available=False)) as address:
        conn, response = _stream(address)
        response.read()
        conn.close()
    assert response.status == 503


def test_health_reports_whether_streaming_exists() -> None:
    for engine, expected in ((None, False), (sh.FakeStreamingEngine(), True)):
        with running(engine) as address:
            conn = HTTPConnection(*address, timeout=5)
            conn.request("GET", "/health")
            payload = json.loads(conn.getresponse().read())
            conn.close()
        assert payload["streaming"] is expected
        assert payload["stream_format"] == "pcm_s16le"
        assert payload["stream_sample_rate"] == 24_000


# --- the response contract --------------------------------------------------


def test_a_successful_stream_carries_the_documented_headers() -> None:
    with running(sh.FakeStreamingEngine(chunks=3)) as address:
        conn, response = _stream(address)
        body = response.read()
        conn.close()

    assert response.status == 200
    assert response.getheader("Content-Type") == "audio/pcm"
    assert response.getheader("X-KIKI-Audio-Format") == "pcm_s16le"
    assert response.getheader("X-KIKI-Sample-Rate") == "24000"
    assert response.getheader("X-KIKI-Channels") == "1"
    assert response.getheader("X-KIKI-Streaming") == "true"
    assert response.getheader("Cache-Control") == "no-store"
    # The transitional contract is stated, not left to be inferred.
    assert response.getheader("X-KIKI-Transfer") == "connection-close"
    assert response.getheader("Content-Length") is None
    assert response.getheader("Transfer-Encoding") is None
    assert len(body) > 0


def test_the_body_is_whole_pcm16_samples() -> None:
    with running(sh.FakeStreamingEngine(chunks=3)) as address:
        conn, response = _stream(address, {"chunk_ms": 400})
        body = response.read()
        conn.close()

    assert len(body) % 2 == 0
    # 400 ms mono PCM16 at 24 kHz is 19200 bytes.
    assert len(body) == 3 * 19_200


@pytest.mark.parametrize(("chunk_ms", "expected"), [(160, 7_680), (400, 19_200), (1000, 48_000)])
def test_the_chunk_length_decides_the_byte_count(chunk_ms, expected) -> None:
    with running(sh.FakeStreamingEngine(chunks=1)) as address:
        conn, response = _stream(address, {"chunk_ms": chunk_ms})
        body = response.read()
        conn.close()
    assert len(body) == expected


def test_an_odd_trailing_byte_never_reaches_the_client() -> None:
    """Half a sample is a click. The writer holds it back instead."""
    with running(sh.FakeStreamingEngine(chunks=2, odd_tail=True)) as address:
        conn, response = _stream(address)
        body = response.read()
        conn.close()
    assert len(body) % 2 == 0
    assert len(body) == 2 * 19_200


def test_audio_arrives_before_generation_has_finished() -> None:
    """The whole point: the first chunk must be readable while the engine is
    still working, not after it is done."""
    engine = sh.FakeStreamingEngine(chunks=6, delay_s=0.15)
    with running(engine) as address:
        conn, response = _stream(address)
        began = time.monotonic()
        first = response.read(19_200)
        after_first = time.monotonic() - began
        rest = response.read()
        total = time.monotonic() - began
        conn.close()

    assert len(first) == 19_200
    assert len(rest) > 0
    # Six chunks at 150 ms cannot all have been produced by then.
    assert after_first < total / 2


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"text": ""}, 400),
        ({"text": "   "}, 400),
        ({"text": "Hallo", "speaker": "Gibtsnicht"}, 400),
        ({"text": "Hallo", "language": "Klingonisch"}, 400),
        ({"text": "Hallo", "sample_rate": 48_000}, 400),
        ({"text": "Hallo", "sample_rate": "24000"}, 400),
        ({"text": "Hallo", "format": "wav"}, 400),
        ({"text": "Hallo", "format": "pcm_f32le"}, 400),
        ({"text": "Hallo", "chunk_ms": 159}, 400),
        ({"text": "Hallo", "chunk_ms": 1001}, 400),
        ({"text": "Hallo", "chunk_ms": "400"}, 400),
        ({"text": "Hallo", "chunk_ms": True}, 400),
        ({"text": "Hallo", "speed": 1.5}, 400),
        ({"text": "x" * 5000}, 413),
    ],
)
def test_a_request_outside_the_contract_is_refused(payload, status) -> None:
    engine = sh.FakeStreamingEngine()
    with running(engine) as address:
        conn, response = _stream(address, payload)
        body = response.read()
        conn.close()

    assert response.status == status
    assert json.loads(body)["ok"] is False
    # Refused before anything was generated.
    assert engine.calls == []


def test_a_known_speaker_in_the_wrong_case_is_accepted() -> None:
    """KIKI's config capitalises, the model reports lower-case."""
    engine = sh.FakeStreamingEngine(chunks=1)
    with running(engine) as address:
        conn, response = _stream(address, {"speaker": "SERENA", "language": "german"})
        response.read()
        conn.close()
    assert response.status == 200
    assert engine.calls[0].speaker == "Serena"


def test_broken_json_is_refused_before_generation() -> None:
    engine = sh.FakeStreamingEngine()
    with running(engine) as address:
        conn = HTTPConnection(*address, timeout=5)
        conn.request("POST", "/v1/synthesize/stream", body=b"{nicht json",
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
    assert response.status == 400
    assert engine.calls == []


def test_the_defaults_match_the_documented_contract() -> None:
    engine = sh.FakeStreamingEngine(chunks=1)
    with running(engine) as address:
        conn, response = _stream(address, {})
        response.read()
        conn.close()
    spec = engine.calls[0]
    assert spec.chunk_ms == 400
    assert spec.sample_rate == 24_000
    assert spec.audio_format == "pcm_s16le"
    assert spec.channels == 1


# --- failure on either side of the first byte -------------------------------


def test_a_failure_before_the_first_byte_is_an_http_error() -> None:
    with running(sh.FakeStreamingEngine(fail_before_first=True)) as address:
        conn, response = _stream(address)
        body = response.read()
        conn.close()

    assert response.status == 500
    assert json.loads(body)["ok"] is False


def test_a_failure_after_the_first_byte_never_mixes_protocols() -> None:
    """Once PCM is flowing, an error may only close the connection. A JSON tail
    would be decoded as audio and heard as noise."""
    with running(sh.FakeStreamingEngine(chunks=5, fail_after_chunk=2)) as address:
        conn, response = _stream(address)
        body = response.read()
        conn.close()

    assert response.status == 200
    assert len(body) == 2 * 19_200
    assert len(body) % 2 == 0
    assert b"{" not in body
    assert b"error" not in body


def test_the_service_still_works_after_a_mid_stream_failure() -> None:
    with running(sh.FakeStreamingEngine(chunks=5, fail_after_chunk=1)) as address:
        conn, response = _stream(address)
        response.read()
        conn.close()

        conn2 = HTTPConnection(*address, timeout=5)
        conn2.request("GET", "/health")
        health = conn2.getresponse()
        payload = json.loads(health.read())
        conn2.close()

    assert health.status == 200
    assert payload["ok"] is True


# --- disconnect and cancellation --------------------------------------------


def test_a_client_that_leaves_stops_the_generator() -> None:
    """The GPU must not finish an answer nobody is listening to."""
    engine = sh.FakeStreamingEngine(chunks=200, delay_s=0.02)
    with running(engine) as address:
        raw = _raw_stream(address)
        assert len(_read_at_least(raw, 19_200)) >= 19_200
        raw.close()

        assert _await(lambda: engine.closed), "Generator wurde nie geschlossen"
        assert _await(lambda: engine.saw_cancel), "Generator sah den Abbruch nicht"
        produced = engine.produced
        time.sleep(0.3)
        assert engine.produced == produced, "Generator lief nach dem Abbruch weiter"
        # 200 chunks would be the whole answer; a prompt stop is far short of it.
        assert produced < 40


def test_a_disconnect_before_any_audio_leaves_nothing_running() -> None:
    engine = sh.FakeStreamingEngine(chunks=50, delay_s=0.05)
    with running(engine) as address:
        raw = socket.create_connection(address, timeout=5)
        body = json.dumps({"text": "Hallo Martin."}).encode()
        raw.sendall(
            b"POST /v1/synthesize/stream HTTP/1.0\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        )
        time.sleep(0.05)
        raw.close()
        assert _await(lambda: engine.closed, timeout=8)


def test_the_service_is_usable_again_after_a_disconnect() -> None:
    engine = sh.FakeStreamingEngine(chunks=200, delay_s=0.02)
    with running(engine) as address:
        raw = _raw_stream(address)
        _read_at_least(raw, 19_200)
        raw.close()
        assert _await(lambda: engine.closed)

        conn2, second = _stream(address)
        body = second.read()
        conn2.close()

    assert second.status == 200
    assert len(body) % 2 == 0


# --- one generation at a time -----------------------------------------------


def test_a_second_stream_is_refused_while_one_runs() -> None:
    engine = sh.FakeStreamingEngine(chunks=200, delay_s=0.02)
    with running(engine) as address:
        first_conn, first = _stream(address)
        assert len(first.read(19_200)) == 19_200      # the first one holds the gate

        second_conn, second = _stream(address)
        payload = json.loads(second.read())
        second_conn.close()
        first_conn.close()

    assert second.status == 503
    assert payload["ok"] is False


def test_the_gate_is_released_for_the_next_caller() -> None:
    with running(sh.FakeStreamingEngine(chunks=1)) as address:
        for _ in range(3):
            conn, response = _stream(address)
            body = response.read()
            conn.close()
            assert response.status == 200
            assert len(body) == 19_200


def test_the_gate_is_released_after_a_failure_too() -> None:
    with running(sh.FakeStreamingEngine(fail_before_first=True)) as address:
        conn, response = _stream(address)
        response.read()
        conn.close()
        assert response.status == 500

    with running(sh.FakeStreamingEngine(chunks=1)) as address:
        conn, response = _stream(address)
        response.read()
        conn.close()
        assert response.status == 200


# --- log hygiene ------------------------------------------------------------


def test_no_request_text_reaches_the_log(caplog) -> None:
    secret = "Streng geheim sk-live-4711 https://intern.example.com"
    with caplog.at_level("DEBUG"), running(sh.FakeStreamingEngine(chunks=2)) as address:
        conn, response = _stream(address, {"text": secret, "request_id": "abc-123"})
        response.read()
        conn.close()

    assert "Streng geheim" not in caplog.text
    assert "sk-live-4711" not in caplog.text
    assert "intern.example.com" not in caplog.text
    assert "abc-123" in caplog.text          # the correlation id is the point


def test_a_forged_correlation_id_cannot_shape_a_log_line(caplog) -> None:
    with caplog.at_level("INFO"), running(sh.FakeStreamingEngine(chunks=1)) as address:
        conn, response = _stream(address, {"request_id": "a b\nERROR fake line"})
        response.read()
        conn.close()

    assert "ERROR fake line" not in caplog.text
    assert "\nERROR" not in caplog.text


# --- the WAV route is untouched ---------------------------------------------


def test_the_wav_route_still_works_alongside_a_stream() -> None:
    with running(sh.FakeStreamingEngine(chunks=1)) as address:
        conn, response = _stream(address)
        response.read()
        conn.close()

        conn2, wav = _post(address, "/v1/synthesize", {"text": "Hallo KIKI"})
        body = wav.read()
        conn2.close()

    assert wav.status == 200
    assert body[:4] == b"RIFF"
    assert int(wav.getheader("Content-Length")) == len(body)


def test_the_stream_route_does_not_answer_the_wav_path() -> None:
    with running(sh.FakeStreamingEngine(chunks=1)) as address:
        conn, wav = _post(address, "/v1/synthesize", {"text": "Hallo KIKI"})
        body = wav.read()
        conn.close()
    assert wav.getheader("Content-Type") == "audio/wav"
    assert wav.getheader("X-KIKI-Streaming") is None
    assert body[:4] == b"RIFF"


# --- the writer, without a socket -------------------------------------------


def _pump(pieces, **kwargs):
    written: list[bytes] = []
    firsts: list[int] = []
    token = kwargs.pop("token", None) or sh.CancelToken()
    outcome = sh.pump_pcm(
        iter(pieces),
        on_first=lambda: firsts.append(1),
        write=kwargs.pop("write", written.append),
        token=token,
        **kwargs,
    )
    return outcome, written, firsts


def test_the_writer_announces_the_first_byte_exactly_once() -> None:
    outcome, written, firsts = _pump([b"\x01\x02", b"\x03\x04", b"\x05\x06"])
    assert firsts == [1]
    assert outcome.chunks == 3
    assert b"".join(written) == b"\x01\x02\x03\x04\x05\x06"


def test_an_odd_byte_waits_for_its_partner() -> None:
    """Splitting a sample across two writes would be heard as a click."""
    outcome, written, _ = _pump([b"\x01\x02\x03", b"\x04\x05\x06"])
    assert all(len(block) % 2 == 0 for block in written)
    assert b"".join(written) == b"\x01\x02\x03\x04\x05\x06"
    assert outcome.error == ""


def test_a_lone_odd_byte_never_becomes_a_write() -> None:
    outcome, written, firsts = _pump([b"\x01"])
    assert written == []
    assert firsts == []
    assert outcome.started is False
    assert outcome.error == "engine:odd-tail"


def test_empty_pieces_are_skipped() -> None:
    outcome, written, _ = _pump([b"", b"\x01\x02", b""])
    assert written == [b"\x01\x02"]
    assert outcome.chunks == 1


def test_a_failure_before_the_first_byte_is_raised_for_the_caller() -> None:
    def _source():
        raise RuntimeError("kaputt")
        yield b""  # pragma: no cover - generator marker

    with pytest.raises(RuntimeError):
        sh.pump_pcm(
            _source(), on_first=lambda: None, write=lambda _b: None, token=sh.CancelToken()
        )


def test_a_failure_after_the_first_byte_is_recorded_not_raised() -> None:
    def _source():
        yield b"\x01\x02"
        raise RuntimeError("kaputt")

    written: list[bytes] = []
    outcome = sh.pump_pcm(
        _source(), on_first=lambda: None, write=written.append, token=sh.CancelToken()
    )
    assert written == [b"\x01\x02"]
    assert outcome.started is True
    assert outcome.error == "engine:RuntimeError"
    # The category only — an exception message could carry a path or a prompt.
    assert "kaputt" not in outcome.error


def test_a_write_failure_cancels_the_token() -> None:
    token = sh.CancelToken()

    def _boom(_block):
        raise BrokenPipeError

    outcome, _written, _ = _pump([b"\x01\x02", b"\x03\x04"], write=_boom, token=token)
    assert outcome.disconnected is True
    assert outcome.cancelled is True
    assert token.cancelled is True


def test_a_departed_peer_stops_the_pump_before_the_next_chunk() -> None:
    token = sh.CancelToken()
    sent: list[bytes] = []
    outcome = sh.pump_pcm(
        iter([b"\x01\x02"] * 5),
        on_first=lambda: None,
        write=sent.append,
        token=token,
        peer_gone=lambda: len(sent) >= 2,
    )
    assert outcome.disconnected is True
    assert token.cancelled is True
    assert outcome.chunks == 2


def test_the_source_is_always_closed() -> None:
    closed = []

    def _source():
        try:
            yield b"\x01\x02"
        finally:
            closed.append(True)

    sh.pump_pcm(
        _source(), on_first=lambda: None, write=lambda _b: None, token=sh.CancelToken()
    )
    assert closed == [True]


def test_a_cancelled_token_stops_the_pump() -> None:
    token = sh.CancelToken()
    token.cancel()
    outcome, written, firsts = _pump([b"\x01\x02"] * 3, token=token)
    assert written == []
    assert firsts == []
    assert outcome.cancelled is True


# --- the language both routes hand to Qwen ----------------------------------


def _resolvers():
    mod = _server_module()
    languages = ["auto", "chinese", "english", "french", "german", "italian"]
    speakers = ["aiden", "serena", "vivian"]

    def wav(wanted: str) -> str:
        return mod.QwenSynthesizer._resolve(wanted, languages, mod.DEFAULT_LANGUAGE)

    def pcm(wanted: str) -> str:
        spec = sh.validate_stream_request(
            {"text": "x", "language": wanted, "speaker": "Serena"},
            speakers=speakers,
            languages=languages,
            default_language=mod.DEFAULT_LANGUAGE,
            default_speaker=mod.DEFAULT_SPEAKER,
        )
        return spec.language

    return wav, pcm


@pytest.mark.parametrize("wanted", ["German", "german", "GERMAN", "  German  "])
def test_both_routes_resolve_the_language_to_the_same_value(wanted) -> None:
    """The WAV path goes through QwenSynthesizer._resolve, the PCM path through
    validate_stream_request. Whatever KIKI is configured with, Qwen must be
    called with the identical string on both."""
    wav, pcm = _resolvers()
    assert wav(wanted) == pcm(wanted) == "german"


def test_both_routes_resolve_the_speaker_to_the_same_value() -> None:
    mod = _server_module()
    speakers = ["aiden", "serena", "vivian"]
    assert mod.QwenSynthesizer._resolve("Serena", speakers, mod.DEFAULT_SPEAKER) == "serena"

    spec = sh.validate_stream_request(
        {"text": "x", "speaker": "Serena", "language": "German"},
        speakers=speakers,
        languages=["german"],
        default_language=mod.DEFAULT_LANGUAGE,
        default_speaker=mod.DEFAULT_SPEAKER,
    )
    assert spec.speaker == "serena"


def test_an_unknown_language_is_where_the_two_routes_differ() -> None:
    """Documented on purpose: the WAV path falls back to the configured default
    — the rule that stopped a typo from silently switching the voice's gender —
    while the streaming route refuses the request outright.

    Only reachable from a hand-edited config; for every value KIKI itself uses,
    the test above shows the two agree.
    """
    wav, pcm = _resolvers()
    assert wav("Klingonisch") == "german"
    with pytest.raises(sh.StreamValidationError):
        pcm("Klingonisch")
