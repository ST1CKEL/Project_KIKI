"""The provider contract, exercised without a model, a GPU or a sound card."""

from __future__ import annotations

import asyncio

import pytest

from kiki.voice.tts import (
    DEFAULT_SAMPLE_RATE,
    AudioChunk,
    FakeTTSProvider,
    NullTTSProvider,
    TTSError,
    TTSGenerationResult,
    TTSProvider,
    TTSProviderStatus,
    TTSRequest,
)


def _collect(provider, request) -> list[AudioChunk]:
    async def go():
        return [chunk async for chunk in provider.synthesize(request)]

    return asyncio.run(go())


# --- the contract itself ----------------------------------------------------


@pytest.mark.parametrize("provider", [FakeTTSProvider(), NullTTSProvider()])
def test_providers_satisfy_the_protocol(provider) -> None:
    assert isinstance(provider, TTSProvider)
    assert provider.provider_id
    assert provider.capabilities().provider_id == provider.provider_id


def test_lifecycle_moves_through_the_documented_states() -> None:
    provider = FakeTTSProvider()
    assert provider.status is TTSProviderStatus.UNLOADED

    asyncio.run(provider.load())
    assert provider.status is TTSProviderStatus.READY

    _collect(provider, TTSRequest(text="Hallo Martin."))
    assert provider.status is TTSProviderStatus.READY  # back to ready afterwards

    asyncio.run(provider.unload())
    assert provider.status is TTSProviderStatus.UNLOADED


def test_loading_twice_is_not_an_error() -> None:
    provider = FakeTTSProvider()
    asyncio.run(provider.load())
    asyncio.run(provider.load())
    assert provider.loads == 1
    assert provider.status is TTSProviderStatus.READY


def test_unloading_when_never_loaded_is_safe() -> None:
    provider = FakeTTSProvider()
    asyncio.run(provider.unload())
    assert provider.status is TTSProviderStatus.UNLOADED
    assert provider.unloads == 0


def test_synthesising_before_loading_fails_clearly() -> None:
    provider = FakeTTSProvider()
    with pytest.raises(TTSError) as excinfo:
        _collect(provider, TTSRequest(text="Zu früh."))
    assert excinfo.value.code == "not_ready"


def test_a_load_failure_leaves_the_provider_in_error() -> None:
    provider = FakeTTSProvider(fail_on_load=True)
    with pytest.raises(TTSError) as excinfo:
        asyncio.run(provider.load())
    assert excinfo.value.code == "load"
    assert provider.status is TTSProviderStatus.ERROR


def test_a_synthesis_failure_is_reported_as_ttserror() -> None:
    provider = FakeTTSProvider(fail_on_synthesize=True)
    asyncio.run(provider.load())
    with pytest.raises(TTSError) as excinfo:
        _collect(provider, TTSRequest(text="Kaputt."))
    assert excinfo.value.code == "synthesize"
    assert provider.status is TTSProviderStatus.ERROR


# --- chunks -----------------------------------------------------------------


def test_chunks_carry_their_request_and_are_ordered() -> None:
    provider = FakeTTSProvider(chunk_seconds=0.2)
    asyncio.run(provider.load())
    request = TTSRequest(text="Ein etwas längerer Satz für mehrere Stücke.")
    chunks = _collect(provider, request)

    assert len(chunks) > 1
    assert all(c.request_id == request.id for c in chunks)
    assert [c.sequence for c in chunks] == list(range(len(chunks)))
    assert chunks[-1].final is True
    assert all(c.final is False for c in chunks[:-1])


def test_chunk_audio_metadata_is_complete() -> None:
    provider = FakeTTSProvider()
    asyncio.run(provider.load())
    chunk = _collect(provider, TTSRequest(text="Kurz."))[0]
    assert chunk.sample_rate == DEFAULT_SAMPLE_RATE
    assert chunk.channels == 1
    assert chunk.audio_format == "pcm_s16le"
    assert chunk.duration_s > 0
    assert len(chunk.pcm) % 2 == 0  # PCM16


def test_cancellation_stops_the_stream_for_that_request_only() -> None:
    provider = FakeTTSProvider(chunk_seconds=0.1)
    asyncio.run(provider.load())
    request = TTSRequest(text="Ein langer Satz, der viele Stücke ergeben würde, wirklich viele.")

    async def go():
        seen = []
        async for chunk in provider.synthesize(request):
            seen.append(chunk)
            if len(seen) == 2:
                await provider.cancel(request.id)
        return seen

    seen = asyncio.run(go())
    assert len(seen) == 2  # stopped right after the cancel

    # A different request is unaffected and still runs to completion.
    other = _collect(provider, TTSRequest(text="Ein anderer Satz hier."))
    assert other and other[-1].final is True


def test_cancelling_an_unknown_request_is_ignored() -> None:
    provider = FakeTTSProvider()
    asyncio.run(provider.cancel("gibt-es-nicht"))


# --- null provider ----------------------------------------------------------


def test_the_null_provider_produces_no_audio_and_never_fails() -> None:
    provider = NullTTSProvider()
    asyncio.run(provider.load())
    assert _collect(provider, TTSRequest(text="Wird nicht gesprochen.")) == []
    health = asyncio.run(provider.health_check())
    assert health.ok is True
    assert provider.capabilities().streaming is False


# --- request validation -----------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_a_request_without_speakable_text_is_refused(text) -> None:
    with pytest.raises(ValueError):
        TTSRequest(text=text)


def test_a_non_positive_speed_is_refused() -> None:
    with pytest.raises(ValueError):
        TTSRequest(text="Hallo.", speed=0)


def test_every_request_gets_its_own_id() -> None:
    ids = {TTSRequest(text="Hallo.").id for _ in range(50)}
    assert len(ids) == 50


# --- result -----------------------------------------------------------------


def test_realtime_factor_is_none_without_audio() -> None:
    assert TTSGenerationResult(request_id="x").realtime_factor is None


def test_realtime_factor_reports_the_ratio() -> None:
    result = TTSGenerationResult(
        request_id="x", audio_seconds=4.0, synthesis_seconds=5.0
    )
    assert result.realtime_factor == pytest.approx(1.25)


# --- no side effects on import ---------------------------------------------


def test_importing_the_package_touches_no_gpu_or_audio() -> None:
    """The UI imports this; it must not pull in torch or open a device."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kiki.voice.tts;"
            " bad = [m for m in ('torch', 'gi', 'transformers') if m in sys.modules];"
            " print(','.join(bad))",
        ],
        env={"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"unerwartet geladen: {result.stdout.strip()}"
