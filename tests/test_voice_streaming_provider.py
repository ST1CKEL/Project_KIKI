"""The PCM streaming client: provider and sink. No service, no PipeWire, no GPU.

The HTTP side runs against a fake transport that can be told to split bytes at
awkward boundaries; the sink runs against a fake process that records what was
written and how it was ended. What is under test is the part that only shows up
under stress: reassembly across transport boundaries, truthful `final` flags,
and a process that never survives its request.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import tempfile
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
from kiki.voice.tts.streaming_adapters import (
    PW_CAT_ARGS,
    PipeWirePcmSink,
    StreamingServiceTTSProvider,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOOD_HEADERS = {
    "content-type": "audio/pcm",
    "x-kiki-audio-format": "pcm_s16le",
    "x-kiki-sample-rate": "24000",
    "x-kiki-channels": "1",
    "x-kiki-streaming": "true",
    "x-kiki-transfer": "connection-close",
    "cache-control": "no-store",
}


@pytest.fixture(autouse=True)
def temp_root(tmp_path, monkeypatch):
    """No test here may leave anything in the real /tmp."""
    root = tmp_path / "tmproot"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


# --- fake HTTP --------------------------------------------------------------


class FakeResponse:
    def __init__(self, *, status=200, headers=None, pieces=(), json_body=None, raise_on=None):
        self.status_code = status
        self.headers = dict(GOOD_HEADERS if headers is None else headers)
        self._pieces = list(pieces)
        self._json = json_body
        self._raise_on = raise_on
        self.closed = False
        self.read_calls = 0
        self.delivered = 0

    def json(self):
        return self._json

    async def aread(self):
        return b""

    async def aiter_bytes(self, size=None):  # noqa: ARG002 — the fake decides
        for index, piece in enumerate(self._pieces):
            if self._raise_on is not None and index == self._raise_on:
                raise RuntimeError("Transport kaputt")
            self.read_calls += 1
            self.delivered += len(piece)
            yield piece
            await asyncio.sleep(0)

    async def aclose(self):
        self.closed = True


class _StreamContext:
    def __init__(self, response, client):
        self._response = response
        self._client = client

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *_exc):
        await self._response.aclose()
        return False


class FakeClient:
    """Records requests and hands back a scripted response."""

    def __init__(self, responses, *, log=None, get_payload=None):
        self._responses = list(responses)
        self._log = log if log is not None else []
        self._get_payload = get_payload
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        self.closed = True
        return False

    def stream(self, method, url, *, json=None, headers=None):
        self._log.append({"method": method, "url": url, "json": json, "headers": headers})
        response = self._responses.pop(0) if self._responses else FakeResponse(pieces=[])
        return _StreamContext(response, self)

    async def get(self, url):
        self._log.append({"method": "GET", "url": url})
        payload = self._get_payload if self._get_payload is not None else {
            "ok": True, "ready": True, "streaming": True, "streaming_reason": None,
            "speakers": ["serena", "aiden"],
        }
        return FakeResponse(json_body=payload)


def _factory(responses, *, log=None, get_payload=None):
    def _make(**_kwargs):
        return FakeClient(responses, log=log, get_payload=get_payload)

    return _make


def _pcm(frames: int) -> bytes:
    return b"".join((index % 30_000).to_bytes(2, "little") for index in range(frames))


async def _ready(provider):
    await provider.load()
    return provider


def _collect(provider, request=None, *, cancel_after=None):
    request = request or TTSRequest(text="Hallo Martin.")

    async def go():
        await provider.load()
        out = []
        async for chunk in provider.synthesize(request):
            out.append(chunk)
            if cancel_after is not None and len(out) >= cancel_after:
                await provider.cancel(request.id)
        return out

    return asyncio.run(go())


# --- provider: the request --------------------------------------------------


def test_the_provider_satisfies_the_protocol() -> None:
    provider = StreamingServiceTTSProvider(client_factory=_factory([]))
    assert isinstance(provider, TTSProvider)
    assert provider.capabilities().streaming is True


def test_the_request_goes_to_the_streaming_path() -> None:
    log: list[dict] = []
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(pieces=[_pcm(9600)])], log=log)
    )
    _collect(provider, TTSRequest(text="Guten Abend.", speaker="Vivian", language="English"))

    post = [entry for entry in log if entry["method"] == "POST"][0]
    assert post["url"].endswith("/v1/synthesize/stream")
    assert post["headers"]["Accept"] == "audio/pcm"
    assert post["json"] == {
        "text": "Guten Abend.",
        "language": "English",
        "speaker": "Vivian",
        "sample_rate": 24_000,
        "format": "pcm_s16le",
        "chunk_ms": 400,
    }


def test_an_empty_speaker_falls_back_to_the_configured_default() -> None:
    log: list[dict] = []
    provider = StreamingServiceTTSProvider(
        speaker="Serena", language="German",
        client_factory=_factory([FakeResponse(pieces=[_pcm(9600)])], log=log),
    )
    _collect(provider)
    post = [entry for entry in log if entry["method"] == "POST"][0]
    assert post["json"]["speaker"] == "Serena"
    assert post["json"]["language"] == "German"


def test_a_chunk_length_outside_the_contract_is_refused() -> None:
    for bad in (0, 159, 1001):
        with pytest.raises(ValueError):
            StreamingServiceTTSProvider(chunk_ms=bad)


def test_the_chunk_size_matches_the_documented_contract() -> None:
    assert StreamingServiceTTSProvider(chunk_ms=400).chunk_bytes == 19_200
    assert StreamingServiceTTSProvider(chunk_ms=160).chunk_bytes == 7_680


# --- provider: lifecycle ----------------------------------------------------


def test_load_requires_the_service_to_offer_streaming() -> None:
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([], get_payload={
            "ok": True, "ready": True, "streaming": False,
            "streaming_reason": "runtime_incompatible",
        })
    )
    with pytest.raises(TTSError) as excinfo:
        asyncio.run(provider.load())
    assert excinfo.value.code == "load"
    assert "runtime_incompatible" in str(excinfo.value)
    assert provider.status is TTSProviderStatus.ERROR


def test_synthesising_before_loading_fails_clearly() -> None:
    provider = StreamingServiceTTSProvider(client_factory=_factory([]))

    async def go():
        async for _chunk in provider.synthesize(TTSRequest(text="Zu früh.")):
            pass

    with pytest.raises(TTSError) as excinfo:
        asyncio.run(go())
    assert excinfo.value.code == "not_ready"


def test_health_adopts_the_speakers_the_service_reports() -> None:
    provider = StreamingServiceTTSProvider(client_factory=_factory([]))
    asyncio.run(provider.health_check())
    assert provider.capabilities().speakers == ("serena", "aiden")


def test_unload_touches_only_this_client() -> None:
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(pieces=[_pcm(9600)])])
    )
    _collect(provider)
    asyncio.run(provider.unload())
    assert provider.status is TTSProviderStatus.UNLOADED
    assert provider.unloads == 1


def test_a_speed_the_service_cannot_do_is_refused() -> None:
    provider = StreamingServiceTTSProvider(client_factory=_factory([]))

    async def go():
        await provider.load()
        async for _chunk in provider.synthesize(TTSRequest(text="Schneller.", speed=1.5)):
            pass

    with pytest.raises(TTSError) as excinfo:
        asyncio.run(go())
    assert excinfo.value.code == "unsupported"


# --- provider: header validation --------------------------------------------


@pytest.mark.parametrize("name", sorted(GOOD_HEADERS))
def test_a_missing_header_is_a_protocol_error(name) -> None:
    headers = {key: value for key, value in GOOD_HEADERS.items() if key != name}
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(headers=headers, pieces=[_pcm(9600)])])
    )
    with pytest.raises(TTSError) as excinfo:
        _collect(provider)
    assert excinfo.value.code == "protocol"
    assert name in str(excinfo.value)


@pytest.mark.parametrize(
    ("name", "wrong"),
    [
        ("content-type", "application/json"),
        ("x-kiki-audio-format", "pcm_f32le"),
        ("x-kiki-sample-rate", "48000"),
        ("x-kiki-channels", "2"),
        ("x-kiki-streaming", "false"),
        ("x-kiki-transfer", "chunked"),
        ("cache-control", "max-age=60"),
    ],
)
def test_a_wrong_header_value_is_a_protocol_error(name, wrong) -> None:
    headers = dict(GOOD_HEADERS)
    headers[name] = wrong
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(headers=headers, pieces=[_pcm(9600)])])
    )
    with pytest.raises(TTSError) as excinfo:
        _collect(provider)
    assert excinfo.value.code == "protocol"


def test_a_content_type_with_parameters_is_accepted() -> None:
    headers = dict(GOOD_HEADERS)
    headers["content-type"] = "audio/pcm; rate=24000"
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(headers=headers, pieces=[_pcm(9600)])])
    )
    assert _collect(provider)


def test_headers_are_checked_before_any_byte_is_believed() -> None:
    response = FakeResponse(headers={}, pieces=[_pcm(9600)])
    provider = StreamingServiceTTSProvider(client_factory=_factory([response]))
    with pytest.raises(TTSError):
        _collect(provider)
    assert response.read_calls == 0


# --- provider: reassembly ---------------------------------------------------


def _split(data: bytes, size: int) -> list[bytes]:
    return [data[index : index + size] for index in range(0, len(data), size)]


@pytest.mark.parametrize("transport", [1, 3, 4097, 8192, 19_200, 100_000])
def test_awkward_transport_boundaries_reassemble_into_whole_samples(transport) -> None:
    payload = _pcm(24_000)          # exactly one second
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(pieces=_split(payload, transport))])
    )
    chunks = _collect(provider)

    assert b"".join(chunk.pcm for chunk in chunks) == payload
    assert all(len(chunk.pcm) % 2 == 0 for chunk in chunks)
    assert all(chunk.pcm for chunk in chunks)


def test_the_chunks_carry_the_documented_metadata() -> None:
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(pieces=_split(_pcm(24_000), 1000))])
    )
    request = TTSRequest(text="Hallo.", id="req-4711")
    chunks = _collect(provider, request)

    assert {chunk.request_id for chunk in chunks} == {"req-4711"}
    assert {chunk.sample_rate for chunk in chunks} == {DEFAULT_SAMPLE_RATE}
    assert {chunk.channels for chunk in chunks} == {1}
    assert {chunk.audio_format for chunk in chunks} == {"pcm_s16le"}


def test_the_sequence_numbers_are_dense() -> None:
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(pieces=_split(_pcm(24_000), 777))])
    )
    chunks = _collect(provider)
    assert [chunk.sequence for chunk in chunks] == list(range(len(chunks)))


def test_only_the_last_chunk_is_final() -> None:
    """`final` has to wait for the stream to actually end, which is why one
    chunk is always held back."""
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(pieces=_split(_pcm(24_000), 5000))])
    )
    chunks = _collect(provider)

    assert len(chunks) > 1
    assert [chunk.final for chunk in chunks] == [False] * (len(chunks) - 1) + [True]


def test_full_chunks_are_the_promised_length() -> None:
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(pieces=[_pcm(24_000)])])
    )
    chunks = _collect(provider)
    assert [len(chunk.pcm) for chunk in chunks[:-1]] == [19_200] * (len(chunks) - 1)


def test_a_single_short_stream_is_one_final_chunk() -> None:
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(pieces=[_pcm(400)])])
    )
    chunks = _collect(provider)
    assert len(chunks) == 1
    assert chunks[0].final is True
    assert len(chunks[0].pcm) == 800


def test_empty_transport_chunks_are_ignored() -> None:
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(pieces=[b"", _pcm(400), b"", b""])])
    )
    chunks = _collect(provider)
    assert len(chunks) == 1
    assert chunks[0].pcm == _pcm(400)


def test_a_stream_with_no_audio_yields_nothing() -> None:
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(pieces=[])])
    )
    assert _collect(provider) == []


# --- provider: protocol errors ----------------------------------------------


def test_an_odd_tail_before_any_audio_is_a_protocol_error() -> None:
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(pieces=[b"\x01"])])
    )
    with pytest.raises(TTSError) as excinfo:
        _collect(provider)
    assert excinfo.value.code == "protocol"


def test_an_odd_tail_after_audio_keeps_the_audio_and_still_reports() -> None:
    """The chunks already handed over stay valid; the caller records the
    category rather than the audio being thrown away."""
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(pieces=[_pcm(24_000) + b"\x07"])])
    )
    seen: list[AudioChunk] = []

    async def go():
        await provider.load()
        async for chunk in provider.synthesize(TTSRequest(text="Hallo.")):
            seen.append(chunk)

    with pytest.raises(TTSError) as excinfo:
        asyncio.run(go())
    assert excinfo.value.code == "protocol"
    assert seen
    assert seen[-1].final is True
    assert b"".join(chunk.pcm for chunk in seen) == _pcm(24_000)


def test_a_refusal_before_audio_is_translated() -> None:
    for status, code in ((503, "unavailable"), (400, "rejected"), (500, "service")):
        provider = StreamingServiceTTSProvider(
            client_factory=_factory([FakeResponse(status=status, pieces=[])])
        )
        with pytest.raises(TTSError) as excinfo:
            _collect(provider)
        assert excinfo.value.code == code, status


def test_a_refusal_body_is_never_read_as_audio() -> None:
    body = b'{"ok": false, "error": "kaputt"}'
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(status=503, pieces=[body])])
    )
    with pytest.raises(TTSError):
        _collect(provider)


def test_a_transport_failure_after_audio_delivers_only_what_arrived() -> None:
    """Once PCM started, only the bytes that really arrived may become chunks.

    Checking for JSON *inside* the PCM would be meaningless — 0x7b is a valid
    sample byte. The property that matters is that nothing is invented and
    nothing from the failed read is handed on.
    """
    good = _pcm(19_200)
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([
            FakeResponse(pieces=[good, b'{"error": "kaputt"}'], raise_on=1)
        ])
    )
    seen: list[AudioChunk] = []

    async def go():
        await provider.load()
        async for chunk in provider.synthesize(TTSRequest(text="Hallo.")):
            seen.append(chunk)

    with pytest.raises(Exception):  # noqa: B017 — the transport failed, somehow
        asyncio.run(go())
    delivered = b"".join(chunk.pcm for chunk in seen)
    assert good.startswith(delivered)
    assert b"kaputt" not in delivered


def test_no_url_survives_an_http_error() -> None:
    import httpx

    class _Boom(FakeClient):
        def stream(self, *_args, **_kwargs):
            raise httpx.HTTPError("failed for https://secret.internal:18765/v1?token=sk-1")

    provider = StreamingServiceTTSProvider(client_factory=lambda **_k: _Boom([]))
    with pytest.raises(TTSError) as excinfo:
        _collect(provider)
    assert "secret.internal" not in str(excinfo.value)
    assert "[url]" in str(excinfo.value)


# --- provider: cancellation -------------------------------------------------


def test_a_cancel_closes_the_response_stream() -> None:
    response = FakeResponse(pieces=_split(_pcm(240_000), 4096))
    provider = StreamingServiceTTSProvider(client_factory=_factory([response]))
    chunks = _collect(provider, cancel_after=1)

    assert len(chunks) == 1
    assert response.closed is True
    assert response.delivered < len(_pcm(240_000))


def test_a_cancel_before_any_audio_yields_nothing() -> None:
    response = FakeResponse(pieces=_split(_pcm(24_000), 4096))
    provider = StreamingServiceTTSProvider(client_factory=_factory([response]))
    request = TTSRequest(text="Doch nicht.", id="req-early")

    async def go():
        await provider.load()
        await provider.cancel("req-early")
        return [chunk async for chunk in provider.synthesize(request)]

    assert asyncio.run(go()) == []
    assert response.read_calls == 0


def test_a_cancel_names_exactly_one_request() -> None:
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(pieces=_split(_pcm(24_000), 4096))])
    )

    async def go():
        await provider.load()
        await provider.cancel("jemand-anderes")
        return [chunk async for chunk in provider.synthesize(TTSRequest(text="Ich nicht."))]

    assert asyncio.run(go())


def test_cancelling_twice_is_harmless() -> None:
    provider = StreamingServiceTTSProvider(client_factory=_factory([]))

    async def go():
        await provider.cancel("a")
        await provider.cancel("a")

    asyncio.run(go())


def test_a_new_request_works_after_a_cancel() -> None:
    first = FakeResponse(pieces=_split(_pcm(240_000), 4096))
    second = FakeResponse(pieces=[_pcm(9600)])
    provider = StreamingServiceTTSProvider(client_factory=_factory([first, second]))
    _collect(provider, TTSRequest(text="Erster.", id="a"), cancel_after=1)
    chunks = _collect(provider, TTSRequest(text="Zweiter.", id="b"))

    assert chunks
    assert chunks[-1].final is True
    assert provider.status is TTSProviderStatus.READY


def test_forgotten_cancellations_stay_bounded() -> None:
    from kiki.voice.tts.streaming_adapters import MAX_TRACKED_CANCELS

    provider = StreamingServiceTTSProvider(client_factory=_factory([]))

    async def go():
        for index in range(MAX_TRACKED_CANCELS + 50):
            await provider.cancel(f"req-{index}")

    asyncio.run(go())
    assert len(provider._cancelled) == MAX_TRACKED_CANCELS


# --- the sink ---------------------------------------------------------------


class FakeStdin:
    def __init__(self, *, fail_on_write: Exception | None = None):
        self.written = bytearray()
        self.closed = False
        self.concurrent = 0
        self.max_concurrent = 0
        self._fail = fail_on_write

    def write(self, data: bytes) -> None:
        if self._fail is not None:
            raise self._fail
        self.written.extend(data)

    async def drain(self) -> None:
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            await asyncio.sleep(0)
        finally:
            self.concurrent -= 1

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class FakeProcess:
    def __init__(self, *, returncode=0, hang=False, fail_on_write=None):
        self.stdin = FakeStdin(fail_on_write=fail_on_write)
        self._returncode = returncode
        self._hang = hang
        self.terminated = 0
        self.killed = 0
        self.waits = 0

    async def wait(self) -> int:
        self.waits += 1
        if self._hang and self.terminated == 0 and self.killed == 0:
            await asyncio.Event().wait()
        return self._returncode

    def terminate(self) -> None:
        self.terminated += 1

    def kill(self) -> None:
        self.killed += 1


def _sink(processes=None, **kwargs):
    made: list[FakeProcess] = []
    queue = list(processes or [])
    calls: list[tuple] = []

    async def _spawn(binary, args):
        calls.append((binary, args))
        process = queue.pop(0) if queue else FakeProcess()
        made.append(process)
        return process

    return PipeWirePcmSink(spawn=_spawn, exit_timeout=0.2, **kwargs), made, calls


def _chunk(sequence=0, *, pcm=None, final=False, request_id="r"):
    return AudioChunk(
        request_id=request_id,
        sequence=sequence,
        pcm=pcm if pcm is not None else _pcm(1200),
        sample_rate=DEFAULT_SAMPLE_RATE,
        channels=1,
        final=final,
    )


def test_the_sink_satisfies_the_protocol() -> None:
    assert isinstance(PipeWirePcmSink(), AudioSink)


def test_the_player_is_started_with_fixed_safe_arguments() -> None:
    sink, _made, calls = _sink()
    asyncio.run(sink.play(_chunk(final=True)))

    assert len(calls) == 1
    binary, args = calls[0]
    assert binary == "pw-cat"
    assert args == PW_CAT_ARGS
    # RAW mode and the stdin marker are what make headerless PCM work at all.
    assert "--raw" in args
    assert args[-1] == "-"
    assert all(isinstance(item, str) for item in args)


def test_the_arguments_hold_no_text_from_anywhere() -> None:
    """Nothing in the command line may ever come from a prompt or an answer."""
    joined = " ".join(PW_CAT_ARGS)
    assert re.fullmatch(r"[-a-z0-9 ]+", joined), joined


def test_one_process_serves_every_chunk_of_a_request() -> None:
    sink, made, calls = _sink()

    async def go():
        for index in range(4):
            await sink.play(_chunk(index, final=index == 3))

    asyncio.run(go())
    assert len(calls) == 1
    assert made[0].stdin.closed is True


def test_the_pcm_arrives_in_order() -> None:
    sink, made, _calls = _sink()
    blocks = [_pcm(600), _pcm(700), _pcm(800)]

    async def go():
        for index, block in enumerate(blocks):
            await sink.play(_chunk(index, pcm=block, final=index == 2))

    asyncio.run(go())
    assert bytes(made[0].stdin.written) == b"".join(blocks)


def test_stdin_is_never_written_concurrently() -> None:
    sink, made, _calls = _sink()

    async def go():
        await asyncio.gather(*(sink.play(_chunk(index)) for index in range(5)))
        await sink.stop()

    asyncio.run(go())
    assert made[0].stdin.max_concurrent == 1


def test_the_final_chunk_closes_stdin_and_waits() -> None:
    sink, made, _calls = _sink()
    asyncio.run(sink.play(_chunk(final=True)))

    assert made[0].stdin.closed is True
    assert made[0].waits >= 1
    assert sink.active_request_id is None


def test_a_nonzero_exit_becomes_a_playback_error() -> None:
    sink, _made, _calls = _sink([FakeProcess(returncode=3)])

    with pytest.raises(TTSError) as excinfo:
        asyncio.run(sink.play(_chunk(final=True)))
    assert excinfo.value.code == "playback"


def test_a_partial_frame_is_trimmed() -> None:
    sink, made, _calls = _sink()
    asyncio.run(sink.play(_chunk(pcm=_pcm(100) + b"\x01", final=True)))
    assert bytes(made[0].stdin.written) == _pcm(100)


def test_an_empty_chunk_starts_no_process() -> None:
    sink, _made, calls = _sink()
    asyncio.run(sink.play(_chunk(pcm=b"")))
    assert calls == []


def test_a_wrong_format_is_refused_before_the_process_starts() -> None:
    sink, _made, calls = _sink()
    chunk = AudioChunk(request_id="r", sequence=0, pcm=_pcm(100), audio_format="opus")

    with pytest.raises(TTSError) as excinfo:
        asyncio.run(sink.play(chunk))
    assert excinfo.value.code == "format"
    assert calls == []


def test_a_wrong_sample_rate_is_refused() -> None:
    """The process was started for one fixed shape; anything else would play at
    the wrong speed instead of being refused."""
    sink, _made, calls = _sink()
    chunk = AudioChunk(request_id="r", sequence=0, pcm=_pcm(100), sample_rate=48_000)

    with pytest.raises(TTSError) as excinfo:
        asyncio.run(sink.play(chunk))
    assert excinfo.value.code == "format"
    assert calls == []


# --- the sink: teardown -----------------------------------------------------


def test_stop_ends_the_process_and_is_idempotent() -> None:
    sink, made, _calls = _sink()

    async def go():
        await sink.play(_chunk())
        await sink.stop()
        await sink.stop()

    asyncio.run(go())
    assert made[0].stdin.closed is True
    assert sink.active_request_id is None


def test_close_ends_the_process_and_is_idempotent() -> None:
    sink, made, _calls = _sink()

    async def go():
        await sink.play(_chunk())
        await sink.close()
        await sink.close()

    asyncio.run(go())
    assert made[0].stdin.closed is True
    assert sink.closed is True


def test_playing_after_close_is_refused() -> None:
    sink, _made, calls = _sink()

    async def go():
        await sink.close()
        with pytest.raises(TTSError) as excinfo:
            await sink.play(_chunk())
        return excinfo.value

    assert asyncio.run(go()).code == "closed"
    assert calls == []


def test_stop_terminates_instead_of_letting_the_buffer_play_out() -> None:
    """Regression: stop() used to close stdin and wait, so pw-cat played its
    whole buffer first — measured at 1.07 s of KIKI talking over a barge-in."""
    sink, created, _calls = _sink()

    async def go():
        await sink.play(_chunk())
        await sink.stop()

    asyncio.run(go())
    assert created[0].terminated >= 1, "stop() hat nicht terminiert"


def test_the_final_chunk_drains_rather_than_terminating() -> None:
    """The other half of the same decision: audio already handed over is audio
    the listener is meant to hear, so the end of an utterance waits."""
    sink, created, _calls = _sink()
    asyncio.run(sink.play(_chunk(final=True)))

    assert created[0].stdin.closed is True
    assert created[0].terminated == 0
    assert created[0].killed == 0


def test_a_hanging_process_is_terminated_rather_than_waited_on() -> None:
    """A player that will not exit must not hold the sink forever."""
    sink, created, _calls = _sink([FakeProcess(hang=True)])

    async def go():
        await sink.play(_chunk())
        await sink.stop()

    asyncio.run(go())
    assert created[0].terminated >= 1
    assert sink.active_request_id is None


def test_a_broken_pipe_is_translated_and_cleaned_up() -> None:
    sink, created, _calls = _sink([FakeProcess(fail_on_write=BrokenPipeError())])

    with pytest.raises(TTSError) as excinfo:
        asyncio.run(sink.play(_chunk()))
    assert excinfo.value.code == "playback"
    assert sink.active_request_id is None
    assert created[0].stdin.closed is True


def test_a_failure_does_not_block_the_next_request() -> None:
    sink, created, calls = _sink([FakeProcess(fail_on_write=BrokenPipeError()), FakeProcess()])

    async def go():
        with pytest.raises(TTSError):
            await sink.play(_chunk(request_id="a"))
        await sink.play(_chunk(request_id="b", final=True))

    asyncio.run(go())
    assert len(calls) == 2
    assert created[1].stdin.written


def test_a_new_request_id_supersedes_the_old_process() -> None:
    """Barge-in: the controller supersedes utterances, so switching ids ends the
    old player rather than raising."""
    sink, created, calls = _sink()

    async def go():
        await sink.play(_chunk(request_id="alt"))
        await sink.play(_chunk(request_id="neu", final=True))

    asyncio.run(go())
    assert len(calls) == 2
    assert created[0].stdin.closed is True
    assert created[1].stdin.written


def test_no_process_survives_a_close() -> None:
    sink, created, _calls = _sink()

    async def go():
        await sink.play(_chunk())
        await sink.close()

    asyncio.run(go())
    assert sink.active_request_id is None
    assert created[0].stdin.closed is True


def test_a_missing_pw_cat_is_a_named_category(monkeypatch) -> None:
    import kiki.voice.tts.streaming_adapters as sa

    monkeypatch.setattr(sa.shutil, "which", lambda _name: None)
    sink = PipeWirePcmSink()

    with pytest.raises(TTSError) as excinfo:
        asyncio.run(sink.play(_chunk()))
    assert excinfo.value.code == "pw_cat_unavailable"


# --- both halves under the controller ---------------------------------------


def test_provider_and_sink_fit_together_under_the_controller() -> None:
    payload = _pcm(48_000)                       # two seconds
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(pieces=_split(payload, 3000))])
    )
    sink, created, _calls = _sink()
    started: list[str] = []

    async def go():
        await provider.load()
        controller = VoicePlaybackController(
            provider, sink, on_audio_started=lambda event: started.append(event.request_id)
        )
        result = await controller.speak(TTSRequest(text="Alles zusammen.", id="req-1"))
        await controller.shutdown()
        return result

    result = asyncio.run(go())

    assert result.error == ""
    assert result.cancelled is False
    assert result.chunks == 5                    # 96000 Bytes / 19200 = 5
    assert bytes(created[0].stdin.written) == payload
    # on_audio_started still means what it says: the first real PCM went out.
    assert started == ["req-1"]


def test_no_temp_directory_is_created_by_the_streaming_path(temp_root) -> None:
    provider = StreamingServiceTTSProvider(
        client_factory=_factory([FakeResponse(pieces=[_pcm(24_000)])])
    )
    sink, _created, _calls = _sink()

    async def go():
        await provider.load()
        controller = VoicePlaybackController(provider, sink)
        await controller.speak(TTSRequest(text="Ohne Datei."))
        await controller.shutdown()

    asyncio.run(go())
    assert list(temp_root.iterdir()) == []
    assert list(temp_root.glob("kiki-tts-*")) == []
    assert list(temp_root.glob("kiki-sink-*")) == []


# --- import hygiene ---------------------------------------------------------


def test_importing_the_streaming_adapters_pulls_in_nothing_heavy() -> None:
    probe = """
import sys
started = []
sys.addaudithook(
    lambda event, args: started.append(event)
    if event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "socket.connect"}
    else None
)
import kiki.voice.tts.streaming_adapters  # noqa: F401
forbidden = [n for n in ("torch", "gi", "gi.repository", "numpy", "qwen_tts") if n in sys.modules]
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


def test_the_wav_adapters_are_untouched() -> None:
    """The old route must still be exactly the old route."""
    import kiki.voice.tts.adapters as wav

    assert hasattr(wav, "ServiceTTSProvider")
    assert hasattr(wav, "PipeWireAudioSink")
    source = (PROJECT_ROOT / "src/kiki/voice/tts/adapters.py").read_text(encoding="utf-8")
    # It mentions pw-cat in a comment about a future sink; what it must not do
    # is start one, or know about this module.
    assert "--playback" not in source
    assert "create_subprocess" not in source
    assert "streaming_adapters" not in source


def test_no_shell_is_ever_used() -> None:
    source = (
        PROJECT_ROOT / "src/kiki/voice/tts/streaming_adapters.py"
    ).read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "os.system" not in source
    assert "create_subprocess_shell" not in source


# --- what actually goes on the wire -----------------------------------------


def test_the_streaming_body_names_the_language_and_the_voice() -> None:
    """The acceptance criterion, checked on the JSON that really gets sent.

    The director builds `TTSRequest(text=...)` with no speaker and no language,
    so the values must come from the provider's configuration — the same
    `tts.speaker` / `tts.language` the WAV route is given.
    """
    log: list[dict] = []
    provider = StreamingServiceTTSProvider(
        "http://127.0.0.1:18765",
        speaker="Serena",
        language="German",
        client_factory=_factory([FakeResponse(pieces=[_pcm(9600)])], log=log),
    )
    _collect(provider, TTSRequest(text="Hallo Martin."))

    body = [entry for entry in log if entry["method"] == "POST"][0]["json"]
    assert body["language"] == "German"
    assert body["speaker"] == "Serena"


def test_both_routes_are_handed_the_same_language_and_voice() -> None:
    """No drift between the two providers: same configuration in, same values
    out, so the model is asked in German either way."""
    from kiki.voice.tts.adapters import ServiceTTSProvider

    seen: list[dict] = []

    async def _wav_synth(base_url, text, *, dest, language, speaker, timeout):
        seen.append({"text": text, "language": language, "speaker": speaker})
        return _write_wav_file(dest)

    async def _wav_health(base_url, **_kwargs):
        from kiki.voice.tts_client import TtsHealth

        return TtsHealth(ok=True, ready=True, detail="")

    wav = ServiceTTSProvider(
        "http://127.0.0.1:18765", speaker="Serena", language="German",
        synthesize=_wav_synth, health=_wav_health,
    )

    log: list[dict] = []
    streaming = StreamingServiceTTSProvider(
        "http://127.0.0.1:18765", speaker="Serena", language="German",
        client_factory=_factory([FakeResponse(pieces=[_pcm(9600)])], log=log),
    )

    request = TTSRequest(text="Status erledigt und 20 Euro.")

    async def go():
        await wav.load()
        async for _chunk in wav.synthesize(request):
            pass
        await streaming.load()
        async for _chunk in streaming.synthesize(TTSRequest(text=request.text)):
            pass

    asyncio.run(go())

    pcm_body = [entry for entry in log if entry["method"] == "POST"][0]["json"]
    assert seen[0]["language"] == pcm_body["language"] == "German"
    assert seen[0]["speaker"] == pcm_body["speaker"] == "Serena"
    assert seen[0]["text"] == pcm_body["text"] == request.text


def _write_wav_file(dest):
    import wave

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(_pcm(2400))
    return dest


def test_the_normalised_text_is_what_reaches_the_wire() -> None:
    """Emoji and symbols are resolved before either provider sees the text, so
    nothing decorative can reach the model and steer the voice."""
    from kiki.voice.tts_text import speakable

    raw = "Status ✅ und 20 € — siehe https://example.com"
    spoken = speakable(raw)
    # The words around the URL stay; only the address goes. A sentence can end
    # up dangling ("… siehe"), which is the honest cost of not reading URLs out.
    assert spoken == "Status erledigt und zwanzig Euro — siehe"

    log: list[dict] = []
    provider = StreamingServiceTTSProvider(
        "http://127.0.0.1:18765", speaker="Serena", language="German",
        client_factory=_factory([FakeResponse(pieces=[_pcm(9600)])], log=log),
    )
    _collect(provider, TTSRequest(text=spoken))

    sent = [entry for entry in log if entry["method"] == "POST"][0]["json"]["text"]
    assert sent == spoken
    assert "✅" not in sent
    assert "€" not in sent
    assert "http" not in sent
