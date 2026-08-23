"""Which voice route gets built, and why. No service, no player, no network.

Everything the composition root touches from the outside is injected here, so
what is under test is the decision itself: when streaming is chosen, when it is
not, and the guarantee that the two halves of a route are never mixed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from kiki.voice.tts.composition import (
    STREAMING_PREBUFFER_CHUNKS,
    WAV_PREBUFFER_CHUNKS,
    HealthProbe,
    VoiceRoute,
    build_controller_route,
    probe_health,
    why_not_streaming,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Recorder:
    """Stands in for one half of a route and remembers it was built."""

    def __init__(self, kind: str, role: str) -> None:
        self.kind = kind
        self.role = role


def _pairs():
    """Injected factories that record whether they were called at all."""
    built: list[str] = []

    def _streaming(base_url, speaker, language):
        built.append("streaming")
        return _Recorder("streaming", "provider"), _Recorder("streaming", "sink")

    def _wav(base_url, speaker, language):
        built.append("wav")
        return _Recorder("wav", "provider"), _Recorder("wav", "sink")

    return _streaming, _wav, built


def _build(*, health=None, pw_cat=True, streaming_pair=None, wav_pair=None, **kwargs):
    healthy = HealthProbe(reachable=True, available=True, streaming=True)
    return build_controller_route(
        base_url="http://127.0.0.1:18765",
        speaker="Serena",
        language="German",
        probe=lambda _url: health if health is not None else healthy,
        have_pw_cat=lambda: pw_cat,
        streaming_pair=streaming_pair,
        wav_pair=wav_pair,
        **kwargs,
    )


# --- streaming is chosen ----------------------------------------------------


def test_a_healthy_service_gets_the_streaming_route() -> None:
    streaming, wav, built = _pairs()
    route = _build(streaming_pair=streaming, wav_pair=wav)

    assert route.streaming is True
    assert route.provider.kind == "streaming"
    assert route.sink.kind == "streaming"
    assert route.prebuffer_chunks == 2
    assert route.reason == ""


def test_the_streaming_prebuffer_is_the_measured_two_chunks() -> None:
    """One chunk buys nothing — the gate opens as soon as the first lands.
    Two is what produces the ~0.8 s cushion the engine needs at RTF ~1.4."""
    assert STREAMING_PREBUFFER_CHUNKS == 2
    assert WAV_PREBUFFER_CHUNKS == 0


def test_the_wav_pair_is_never_built_when_streaming_wins() -> None:
    streaming, wav, built = _pairs()
    _build(streaming_pair=streaming, wav_pair=wav)
    assert built == ["streaming"]


# --- every reason to fall back ----------------------------------------------


@pytest.mark.parametrize(
    ("health", "pw_cat", "reason"),
    [
        (HealthProbe(reachable=False), True, "service_unreachable"),
        (HealthProbe(reachable=True, available=False), True, "service_not_ready"),
        (HealthProbe(reachable=True, available=True, streaming=False), True, "streaming_off"),
        (
            HealthProbe(reachable=True, available=True, streaming=False,
                        reason="runtime_incompatible"),
            True,
            "runtime_incompatible",
        ),
        (
            HealthProbe(reachable=True, available=True, streaming=True, reason="worker_stuck"),
            True,
            "worker_stuck",
        ),
        (HealthProbe(reachable=True, available=True, streaming=True), False, "pw_cat_missing"),
    ],
)
def test_anything_short_of_a_working_stream_falls_back(health, pw_cat, reason) -> None:
    streaming, wav, built = _pairs()
    route = _build(health=health, pw_cat=pw_cat, streaming_pair=streaming, wav_pair=wav)

    assert route.streaming is False
    assert route.reason == reason
    assert route.prebuffer_chunks == 0
    assert route.provider.kind == "wav"
    assert route.sink.kind == "wav"
    assert built == ["wav"], "die Streaming-Hälften dürfen nicht gebaut werden"


def test_a_missing_player_is_noticed_before_the_service_is_asked() -> None:
    """No point asking the service anything when nothing could play the answer."""
    asked: list[str] = []

    def _probe(url):
        asked.append(url)
        return HealthProbe(reachable=True, available=True, streaming=True)

    reason = why_not_streaming(
        "http://127.0.0.1:18765", probe=_probe, have_pw_cat=lambda: False
    )
    assert reason == "pw_cat_missing"
    assert asked == []


def test_a_failing_streaming_construction_falls_back() -> None:
    _streaming, wav, built = _pairs()

    def _boom(base_url, speaker, language):
        built.append("streaming")
        raise RuntimeError("Adapter kaputt")

    route = _build(streaming_pair=_boom, wav_pair=wav)

    assert route.streaming is False
    assert route.reason == "adapter_error"
    assert route.provider.kind == "wav"
    assert route.sink.kind == "wav"
    assert built == ["streaming", "wav"]


def test_the_streaming_pair_is_never_built_on_a_fallback() -> None:
    streaming, wav, built = _pairs()
    _build(health=HealthProbe(reachable=False), streaming_pair=streaming, wav_pair=wav)
    assert built == ["wav"]


# --- no hybrid route --------------------------------------------------------


@pytest.mark.parametrize(
    "health",
    [
        HealthProbe(reachable=True, available=True, streaming=True),
        HealthProbe(reachable=True, available=True, streaming=False),
        HealthProbe(reachable=False),
    ],
)
def test_both_halves_always_come_from_the_same_route(health) -> None:
    """A streaming provider with a WAV sink would write PCM into a file player.
    The pairs are built together so that combination cannot be expressed."""
    streaming, wav, _built = _pairs()
    route = _build(health=health, streaming_pair=streaming, wav_pair=wav)

    assert route.provider.kind == route.sink.kind
    assert (route.provider.kind == "streaming") is route.streaming


def test_the_real_pairs_are_the_documented_classes() -> None:
    """The default factories, checked without building anything else."""
    from kiki.voice.tts.adapters import PipeWireAudioSink, ServiceTTSProvider
    from kiki.voice.tts.composition import make_streaming_pair, make_wav_pair
    from kiki.voice.tts.streaming_adapters import PipeWirePcmSink, StreamingServiceTTSProvider

    provider, sink = make_streaming_pair("http://127.0.0.1:18765", "Serena", "German")
    assert isinstance(provider, StreamingServiceTTSProvider)
    assert isinstance(sink, PipeWirePcmSink)

    provider, sink = make_wav_pair("http://127.0.0.1:18765", "Serena", "German")
    assert isinstance(provider, ServiceTTSProvider)
    assert isinstance(sink, PipeWireAudioSink)


def test_the_route_is_immutable() -> None:
    route = VoiceRoute(provider=None, sink=None, prebuffer_chunks=2, streaming=True)
    with pytest.raises(Exception):  # noqa: B017 — frozen dataclass
        route.streaming = False


# --- the health probe itself ------------------------------------------------


def test_an_unreachable_service_is_not_an_exception() -> None:
    """Startup must not fail because the TTS service happens to be down."""
    result = probe_health("http://127.0.0.1:1", timeout=0.2)
    assert result.reachable is False
    assert result.streaming is False


def test_the_probe_reads_the_documented_health_fields(monkeypatch) -> None:
    import httpx

    class _Response:
        def json(self):
            return {
                "ok": True, "ready": True, "streaming": True, "streaming_reason": None,
            }

    monkeypatch.setattr(httpx, "get", lambda _url, timeout=None: _Response())
    result = probe_health("http://127.0.0.1:18765")

    assert result.reachable is True
    assert result.available is True
    assert result.streaming is True
    assert result.reason == ""


def test_a_service_that_is_up_but_not_ready_is_not_available(monkeypatch) -> None:
    import httpx

    class _Response:
        def json(self):
            return {"ok": True, "ready": False, "streaming": True}

    monkeypatch.setattr(httpx, "get", lambda _url, timeout=None: _Response())
    assert probe_health("http://127.0.0.1:18765").available is False


def test_a_nonsense_body_does_not_crash_startup(monkeypatch) -> None:
    import httpx

    class _Response:
        def json(self):
            return ["nicht", "was", "erwartet", "wurde"]

    monkeypatch.setattr(httpx, "get", lambda _url, timeout=None: _Response())
    result = probe_health("http://127.0.0.1:18765")
    assert result.reachable is True
    assert result.available is False


def test_the_reason_never_carries_a_url_or_a_body() -> None:
    """Categories only: these strings reach logs and user-facing text."""
    for health, _pw, reason in (
        (HealthProbe(reachable=False), True, "service_unreachable"),
        (HealthProbe(reachable=True, available=False), True, "service_not_ready"),
    ):
        streaming, wav, _built = _pairs()
        route = _build(health=health, streaming_pair=streaming, wav_pair=wav)
        assert "://" not in route.reason
        assert " " not in route.reason
        assert route.reason == reason


# --- import hygiene ---------------------------------------------------------


def test_the_composition_root_pulls_in_nothing_heavy() -> None:
    probe = """
import sys
started = []
sys.addaudithook(
    lambda event, args: started.append(event)
    if event in {"subprocess.Popen", "os.system", "os.exec", "socket.connect"}
    else None
)
import kiki.voice.tts.composition  # noqa: F401
forbidden = [
    n for n in ("gi", "gi.repository", "torch", "numpy", "qwen_tts", "httpx") if n in sys.modules
]
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


def test_the_application_holds_no_streaming_selection_of_its_own() -> None:
    """The whole point of this module: one place decides."""
    source = (PROJECT_ROOT / "src/kiki/application.py").read_text(encoding="utf-8")

    assert "build_controller_route" in source
    for name in ("StreamingServiceTTSProvider", "PipeWirePcmSink", "pw-cat", "streaming_adapters"):
        assert name not in source, name
    assert "prebuffer_chunks=route.prebuffer_chunks" in source
