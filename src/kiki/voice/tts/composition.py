"""Picks the voice route: live PCM streaming, or the WAV chunks it falls back to.

The one place that decides. `application.py` asks for a route and wires what it
gets; it contains no streaming-specific selection of its own, and this module
imports no GTK, no torch, no model and no PipeWire binding.

Why the probe is synchronous
----------------------------
`_build_voice_controller()` runs on the GTK thread during startup and returns
the controller, which is handed to `SpeechDirector` immediately and never
replaced. The choice therefore has to be made *before* construction — an async
probe would have to either block that thread or swap components underneath a
live controller, and swapping is exactly how a streaming provider ends up paired
with a WAV sink.

So the cheap half runs inline: `pw-cat` is a filesystem lookup, and `/health` is
one bounded loopback request. `provider.load()` still runs on the bridge
afterwards and still degrades the whole route if the service went away in
between — that path is unchanged.

The pairs are built together, never separately, so a hybrid route is not
something this module can express.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# Two 400 ms chunks held before playback starts. Measured: this is what buys the
# ~0.8 s cushion the engine needs at RTF ~1.4. One chunk buys nothing, because
# the gate opens as soon as the first one lands.
STREAMING_PREBUFFER_CHUNKS = 2
# The WAV route produces its chunks from a finished file, so there is nothing to
# run out of and nothing to hold back.
WAV_PREBUFFER_CHUNKS = 0

PW_CAT = "pw-cat"
HEALTH_TIMEOUT_S = 0.4


@dataclass(frozen=True)
class HealthProbe:
    """What one look at `/health` said. Never carries a body or a URL."""

    reachable: bool = False
    available: bool = False
    streaming: bool = False
    reason: str = ""


@dataclass(frozen=True)
class VoiceRoute:
    """One matched provider/sink pair and the prebuffer that belongs to it."""

    provider: Any
    sink: Any
    prebuffer_chunks: int
    streaming: bool
    # Why streaming was not chosen. A fixed category, safe to log and to show.
    reason: str = ""


def _pw_cat_present() -> bool:
    return shutil.which(PW_CAT) is not None


def probe_health(base_url: str, *, timeout: float = HEALTH_TIMEOUT_S) -> HealthProbe:
    """One bounded loopback request. Never raises, never quotes the response.

    `available` is `ok and ready`: the service reports those two separately and
    has no single field of that name.
    """
    try:
        import httpx

        response = httpx.get(base_url.rstrip("/") + "/health", timeout=timeout)
        payload = response.json()
    except Exception:
        return HealthProbe(reachable=False)
    if not isinstance(payload, dict):
        return HealthProbe(reachable=True)
    return HealthProbe(
        reachable=True,
        available=bool(payload.get("ok")) and bool(payload.get("ready", True)),
        streaming=bool(payload.get("streaming", False)),
        reason=str(payload.get("streaming_reason") or ""),
    )


def make_streaming_pair(base_url: str, speaker: str, language: str) -> tuple[Any, Any]:
    """Both halves of the PCM route, or neither."""
    from kiki.voice.tts.streaming_adapters import (
        PipeWirePcmSink,
        StreamingServiceTTSProvider,
    )

    provider = StreamingServiceTTSProvider(base_url, speaker=speaker, language=language)
    return provider, PipeWirePcmSink()


def make_wav_pair(base_url: str, speaker: str, language: str) -> tuple[Any, Any]:
    """Both halves of the WAV route, or neither."""
    from kiki.voice.tts.adapters import PipeWireAudioSink, ServiceTTSProvider

    provider = ServiceTTSProvider(base_url, speaker=speaker, language=language)
    return provider, PipeWireAudioSink()


def why_not_streaming(
    base_url: str,
    *,
    probe: Callable[[str], HealthProbe] | None = None,
    have_pw_cat: Callable[[], bool] | None = None,
) -> str:
    """Empty when the PCM route is usable; otherwise the category that stopped it.

    `pw-cat` is checked first: it costs a filesystem lookup, and without a player
    there is no point asking the service anything.
    """
    if not (have_pw_cat or _pw_cat_present)():
        return "pw_cat_missing"
    health = (probe or probe_health)(base_url)
    if not health.reachable:
        return "service_unreachable"
    if not health.available:
        return "service_not_ready"
    if not health.streaming:
        return health.reason or "streaming_off"
    if health.reason:
        # The service says streaming is on but names a problem anyway; believe
        # the problem.
        return health.reason
    return ""


def build_controller_route(
    *,
    base_url: str,
    speaker: str,
    language: str,
    probe: Callable[[str], HealthProbe] | None = None,
    have_pw_cat: Callable[[], bool] | None = None,
    streaming_pair: Callable[[str, str, str], tuple[Any, Any]] | None = None,
    wav_pair: Callable[[str, str, str], tuple[Any, Any]] | None = None,
) -> VoiceRoute:
    """Choose one route and build it whole.

    Everything that touches the outside world is injectable, so the decision can
    be tested without a service, a player or a network.
    """
    reason = why_not_streaming(base_url, probe=probe, have_pw_cat=have_pw_cat)
    if not reason:
        try:
            provider, sink = (streaming_pair or make_streaming_pair)(
                base_url, speaker, language
            )
        except Exception:
            # No traceback: it would carry the base URL through the frames, and
            # the only thing the caller can do about it is take the WAV route.
            log.warning("streaming adapters could not be built; using the WAV route")
            reason = "adapter_error"
        else:
            log.info("voice route: streaming PCM, prebuffer %d", STREAMING_PREBUFFER_CHUNKS)
            return VoiceRoute(
                provider=provider,
                sink=sink,
                prebuffer_chunks=STREAMING_PREBUFFER_CHUNKS,
                streaming=True,
            )
    log.info("voice route: WAV (%s)", reason)
    provider, sink = (wav_pair or make_wav_pair)(base_url, speaker, language)
    return VoiceRoute(
        provider=provider,
        sink=sink,
        prebuffer_chunks=WAV_PREBUFFER_CHUNKS,
        streaming=False,
        reason=reason,
    )
