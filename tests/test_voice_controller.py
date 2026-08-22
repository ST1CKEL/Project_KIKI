"""Voice queue, cancellation and shutdown. No audio device, no model, no GPU."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from kiki.voice.tts import (
    AudioChunk,
    AudioStartedEvent,
    FakeAudioSink,
    FakeTTSProvider,
    PlaybackState,
    TTSError,
    TTSProviderStatus,
    TTSRequest,
    VoicePlaybackController,
)

TEXT = "Ein Satz, der mehrere Stücke ergibt und damit die Queue beschäftigt."


async def _ready(**kwargs) -> FakeTTSProvider:
    provider = FakeTTSProvider(**kwargs)
    await provider.load()
    return provider


def _run(coro):
    return asyncio.run(coro)


# --- order and completion ---------------------------------------------------


def test_chunks_are_played_in_order_and_exactly_once() -> None:
    async def go():
        provider = await _ready(chunk_seconds=0.2)
        sink = FakeAudioSink()
        controller = VoicePlaybackController(provider, sink)
        result = await controller.speak(TTSRequest(text=TEXT))
        return sink, result

    sink, result = _run(go())
    assert sink.played_sequences == sorted(sink.played_sequences)
    assert sink.played_sequences == list(range(len(sink.played)))
    assert result.chunks == len(sink.played)
    assert result.cancelled is False
    assert result.error == ""
    assert result.time_to_first_audio is not None
    assert result.audio_seconds > 0


def test_playback_never_overlaps() -> None:
    """Only one chunk may be audible at a time."""

    async def go():
        provider = await _ready(chunk_seconds=0.1)
        sink = FakeAudioSink()
        await VoicePlaybackController(provider, sink).speak(TTSRequest(text=TEXT))
        return sink

    assert _run(go()).max_concurrent == 1


def test_the_state_returns_to_idle_after_speaking() -> None:
    async def go():
        provider = await _ready()
        controller = VoicePlaybackController(provider, FakeAudioSink())
        assert controller.state is PlaybackState.IDLE
        await controller.speak(TTSRequest(text="Kurz."))
        return controller

    controller = _run(go())
    assert controller.state is PlaybackState.IDLE
    assert controller.current_request_id is None
    assert controller.busy is False


# --- the queue bound --------------------------------------------------------


def test_at_most_one_chunk_is_buffered_ahead() -> None:
    """The whole point of the bound: synthesis must not run far ahead.

    The sink holds each chunk until released, so the producer can only be one
    chunk in front of the player.
    """

    async def go():
        provider = await _ready(chunk_seconds=0.05)
        gate = asyncio.Event()
        produced: list[int] = []
        played: list[int] = []

        class GatedSink(FakeAudioSink):
            async def play(self, chunk: AudioChunk) -> None:
                played.append(chunk.sequence)
                await gate.wait()
                await super().play(chunk)

        original = provider.synthesize

        async def counting(request):
            async for chunk in original(request):
                produced.append(chunk.sequence)
                yield chunk

        provider.synthesize = counting
        controller = VoicePlaybackController(provider, GatedSink(), prefetch=1)
        task = await controller.submit(TTSRequest(text=TEXT * 3))
        await asyncio.sleep(0.3)
        # One chunk is in the player, one waits in the queue, one is blocked on
        # put() — so at most two more than what has been played.
        ahead = len(produced) - len(played)
        gate.set()
        await task
        return ahead

    assert _run(go()) <= 2


# --- cancellation -----------------------------------------------------------


def test_cancel_stops_playback_and_clears_the_queue() -> None:
    async def go():
        provider = await _ready(chunk_seconds=0.05, latency_s=0.02)
        sink = FakeAudioSink()
        controller = VoicePlaybackController(provider, sink)
        request = TTSRequest(text=TEXT * 4)
        task = await controller.submit(request)
        await asyncio.sleep(0.08)
        stopped = await controller.cancel(request.id)
        with pytest.raises(asyncio.CancelledError):
            await task
        after = len(sink.played)
        await asyncio.sleep(0.15)
        return stopped, sink, after, controller

    stopped, sink, after, controller = _run(go())
    assert stopped is True
    assert sink.stops >= 1
    # Nothing plays after the cancel returned.
    assert len(sink.played) == after
    assert controller.state is PlaybackState.IDLE


def test_cancelling_twice_is_harmless() -> None:
    async def go():
        provider = await _ready(chunk_seconds=0.05, latency_s=0.02)
        controller = VoicePlaybackController(provider, FakeAudioSink())
        request = TTSRequest(text=TEXT * 3)
        task = await controller.submit(request)
        await asyncio.sleep(0.05)
        first = await controller.cancel(request.id)
        second = await controller.cancel(request.id)
        third = await controller.cancel(request.id)
        with pytest.raises(asyncio.CancelledError):
            await task
        return first, second, third, controller

    first, second, third, controller = _run(go())
    assert first is True
    assert second is False and third is False  # already gone, still no error
    assert controller.state is PlaybackState.IDLE


def test_cancelling_an_unknown_request_returns_false() -> None:
    async def go():
        provider = await _ready()
        controller = VoicePlaybackController(provider, FakeAudioSink())
        return await controller.cancel("gibt-es-nicht")

    assert _run(go()) is False


def test_cancel_addresses_exactly_one_request() -> None:
    """Cancelling a finished request must not stop the one running now."""

    async def go():
        provider = await _ready(chunk_seconds=0.05, latency_s=0.01)
        sink = FakeAudioSink()
        controller = VoicePlaybackController(provider, sink)
        old = TTSRequest(text="Alt.")
        await controller.speak(old)
        played_before = len(sink.played)

        new = TTSRequest(text=TEXT)
        task = await controller.submit(new)
        assert await controller.cancel(old.id) is False
        await task
        return played_before, sink

    played_before, sink = _run(go())
    assert len(sink.played) > played_before  # the new answer ran to the end


def test_interrupt_drops_the_current_answer() -> None:
    async def go():
        provider = await _ready(chunk_seconds=0.05, latency_s=0.02)
        sink = FakeAudioSink()
        controller = VoicePlaybackController(provider, sink)
        task = await controller.submit(TTSRequest(text=TEXT * 4))
        await asyncio.sleep(0.06)
        hit = await controller.interrupt()
        with pytest.raises(asyncio.CancelledError):
            await task
        return hit, controller

    hit, controller = _run(go())
    assert hit is True
    assert controller.state is PlaybackState.IDLE


def test_interrupt_when_silent_reports_nothing_to_do() -> None:
    async def go():
        provider = await _ready()
        return await VoicePlaybackController(provider, FakeAudioSink()).interrupt()

    assert _run(go()) is False


# --- a new answer during playback ------------------------------------------


def test_a_new_answer_supersedes_the_running_one() -> None:
    async def go():
        provider = await _ready(chunk_seconds=0.05, latency_s=0.02)
        sink = FakeAudioSink()
        controller = VoicePlaybackController(provider, sink)
        first = TTSRequest(text=TEXT * 4)
        first_task = await controller.submit(first)
        await asyncio.sleep(0.06)

        second = TTSRequest(text="Die neue Antwort.")
        second_result = await controller.speak(second)
        with pytest.raises(asyncio.CancelledError):
            await first_task
        return controller, second_result, sink

    controller, result, sink = _run(go())
    assert result.request_id != ""
    assert result.cancelled is False
    assert sink.stops >= 1
    assert controller.state is PlaybackState.IDLE
    # No chunk of the superseded answer was played after the switch.
    assert all(isinstance(c, AudioChunk) for c in sink.played)


# --- errors must not block the queue ---------------------------------------


def test_a_failing_chunk_is_skipped_and_the_rest_still_plays() -> None:
    async def go():
        provider = await _ready(chunk_seconds=0.1)
        sink = FakeAudioSink(fail_on_sequence={1})
        controller = VoicePlaybackController(provider, sink)
        result = await controller.speak(TTSRequest(text=TEXT * 2))
        return sink, result, controller

    sink, result, controller = _run(go())
    assert 1 not in sink.played_sequences
    assert len(sink.played) >= 2  # the queue kept moving
    assert result.error != ""
    assert controller.state is PlaybackState.IDLE
    assert controller.busy is False


def test_a_synthesis_error_ends_the_generation_cleanly() -> None:
    async def go():
        provider = await _ready(fail_on_synthesize=True)
        sink = FakeAudioSink()
        controller = VoicePlaybackController(provider, sink)
        result = await controller.speak(TTSRequest(text=TEXT))
        return result, sink, controller

    result, sink, controller = _run(go())
    assert result.error != ""
    assert sink.played == []
    assert controller.state is PlaybackState.IDLE
    assert controller.busy is False


def test_the_controller_still_works_after_an_error() -> None:
    """A failed answer must not wedge the queue for the next one."""

    async def go():
        provider = await _ready()
        sink = FakeAudioSink(fail_on_sequence={0})
        controller = VoicePlaybackController(provider, sink)
        await controller.speak(TTSRequest(text="Erste Antwort, die scheitert."))
        healthy = FakeAudioSink()
        controller._sink = healthy
        result = await controller.speak(TTSRequest(text="Zweite Antwort."))
        return healthy, result

    healthy, result = _run(go())
    assert healthy.played != []
    assert result.chunks > 0


# --- shutdown ---------------------------------------------------------------


def test_shutdown_stops_playback_and_closes_the_sink() -> None:
    async def go():
        provider = await _ready(chunk_seconds=0.05, latency_s=0.02)
        sink = FakeAudioSink()
        controller = VoicePlaybackController(provider, sink)
        task = await controller.submit(TTSRequest(text=TEXT * 4))
        await asyncio.sleep(0.06)
        await controller.shutdown()
        with pytest.raises(asyncio.CancelledError):
            await task
        return controller, sink

    controller, sink = _run(go())
    assert controller.state is PlaybackState.CLOSED
    assert sink.closed is True
    assert sink.stops >= 1


def test_shutdown_is_idempotent() -> None:
    async def go():
        provider = await _ready()
        controller = VoicePlaybackController(provider, FakeAudioSink())
        await controller.shutdown()
        await controller.shutdown()
        return controller

    assert _run(go()).state is PlaybackState.CLOSED


def test_speaking_after_shutdown_is_refused() -> None:
    async def go():
        provider = await _ready()
        controller = VoicePlaybackController(provider, FakeAudioSink())
        await controller.shutdown()
        with pytest.raises(TTSError) as excinfo:
            await controller.submit(TTSRequest(text="Zu spät."))
        return excinfo.value

    assert _run(go()).code == "closed"


def test_shutdown_while_idle_still_closes_the_sink() -> None:
    async def go():
        provider = await _ready()
        sink = FakeAudioSink()
        await VoicePlaybackController(provider, sink).shutdown()
        return sink

    assert _run(go()).closed is True


# --- isolation --------------------------------------------------------------


def test_the_controller_imports_nothing_heavy() -> None:
    """It must not drag torch, CUDA, GTK or a network client into the process.

    `socket` is deliberately not on this list: asyncio imports it itself. The
    rule is that the controller opens no connection, which is a property of the
    code below, not of the import graph.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kiki.voice.tts.controller;"
            " bad = [m for m in ('torch','gi','transformers','httpx','urllib.request',"
            "'kiki.voice.tts_player','kiki.voice.tts_client') if m in sys.modules];"
            " print(','.join(bad))",
        ],
        env={"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"unerwartet geladen: {result.stdout.strip()}"


def test_the_controller_source_opens_no_connection_and_runs_no_shell() -> None:
    """Checked against the source, since an import test cannot prove behaviour."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "src/kiki/voice/tts/controller.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("subprocess", "socket", "httpx", "urllib", "os.system", "popen"):
        assert forbidden not in source, forbidden


# --- robustness review ------------------------------------------------------


class _SteppedProvider:
    """Yields a fixed number of chunks, one per event-loop turn."""

    provider_id = "stepped"

    def __init__(self, count: int = 6) -> None:
        self._count = count
        self.cancelled: set[str] = set()

    @property
    def status(self):
        from kiki.voice.tts import TTSProviderStatus

        return TTSProviderStatus.READY

    def capabilities(self):
        from kiki.voice.tts import TTSProviderCapabilities

        return TTSProviderCapabilities(provider_id=self.provider_id)

    async def health_check(self):
        from kiki.voice.tts import TTSHealth, TTSProviderStatus

        return TTSHealth(ok=True, status=TTSProviderStatus.READY)

    async def load(self) -> None:
        return None

    async def unload(self) -> None:
        return None

    async def cancel(self, request_id: str) -> None:
        self.cancelled.add(request_id)

    async def synthesize(self, request):
        for sequence in range(self._count):
            yield AudioChunk(
                request_id=request.id,
                sequence=sequence,
                pcm=b"\x00\x00" * 2400,
                final=sequence == self._count - 1,
            )
            await asyncio.sleep(0)


def test_cancel_with_a_full_queue_terminates() -> None:
    """The queue-completion case from the robustness review.

    The consumer stops playing while a chunk still sits in Queue(maxsize=1) and
    the producer is about to write its sentinel. Exiting the consumer early
    deadlocked here: the producer blocked on a full queue and `_run` blocked on
    `await producer`. Proven in isolation before the fix.
    """

    async def go():
        provider = _SteppedProvider(count=6)
        gate = asyncio.Event()

        class GatedSink(FakeAudioSink):
            async def play(self, chunk: AudioChunk) -> None:
                if chunk.sequence == 0:
                    await gate.wait()  # let the producer fill the buffer
                await super().play(chunk)

        sink = GatedSink()
        controller = VoicePlaybackController(provider, sink, prefetch=1)
        request = TTSRequest(text="Sechs Stücke, die gecancelt werden.")
        task = await controller.submit(request)
        await asyncio.sleep(0.05)

        # Set the flag exactly as cancel() does, but leave the task alone: the
        # design must not depend on an external task.cancel() arriving.
        controller._cancelled.add(request.id)
        gate.set()
        result = await task
        return result, sink, controller

    result, sink, controller = _run(asyncio.wait_for(go(), timeout=5.0))
    assert result.cancelled is True
    # Only what was already playing was heard; the buffered rest was discarded.
    assert sink.played_sequences == [0]
    assert controller.state is PlaybackState.IDLE
    assert controller.busy is False


def test_cancel_with_a_full_queue_via_the_public_api_terminates() -> None:
    async def go():
        provider = _SteppedProvider(count=8)
        gate = asyncio.Event()

        class GatedSink(FakeAudioSink):
            async def play(self, chunk: AudioChunk) -> None:
                if chunk.sequence == 0:
                    await gate.wait()
                await super().play(chunk)

        controller = VoicePlaybackController(provider, GatedSink(), prefetch=1)
        request = TTSRequest(text="Acht Stücke.")
        task = await controller.submit(request)
        await asyncio.sleep(0.05)
        stopped = await controller.cancel(request.id)
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        return stopped, controller, provider

    stopped, controller, provider = _run(asyncio.wait_for(go(), timeout=5.0))
    assert stopped is True
    assert controller.state is PlaybackState.IDLE
    assert provider.cancelled  # the provider was told, not just the queue


def test_shutdown_with_a_full_queue_terminates() -> None:
    async def go():
        provider = _SteppedProvider(count=8)
        gate = asyncio.Event()

        class GatedSink(FakeAudioSink):
            async def play(self, chunk: AudioChunk) -> None:
                if chunk.sequence == 0:
                    await gate.wait()
                await super().play(chunk)

        sink = GatedSink()
        controller = VoicePlaybackController(provider, sink, prefetch=1)
        task = await controller.submit(TTSRequest(text="Acht Stücke."))
        await asyncio.sleep(0.05)
        await controller.shutdown()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        return controller, sink

    controller, sink = _run(asyncio.wait_for(go(), timeout=5.0))
    assert controller.state is PlaybackState.CLOSED
    assert sink.closed is True


def test_superseding_with_a_full_queue_terminates() -> None:
    async def go():
        provider = _SteppedProvider(count=8)
        gate = asyncio.Event()

        class GatedSink(FakeAudioSink):
            async def play(self, chunk: AudioChunk) -> None:
                if chunk.sequence == 0 and not gate.is_set():
                    await gate.wait()
                await super().play(chunk)

        controller = VoicePlaybackController(provider, GatedSink(), prefetch=1)
        first = await controller.submit(TTSRequest(text="Die erste Antwort."))
        await asyncio.sleep(0.05)
        gate.set()
        second = await controller.speak(TTSRequest(text="Die zweite Antwort."))
        with pytest.raises(asyncio.CancelledError):
            await first
        return second, controller

    second, controller = _run(asyncio.wait_for(go(), timeout=5.0))
    assert second.cancelled is False
    assert controller.state is PlaybackState.IDLE


def test_no_task_is_left_running_after_a_cancel() -> None:
    """An orphaned producer would block the loop at shutdown."""

    async def go():
        provider = _SteppedProvider(count=8)
        gate = asyncio.Event()

        class GatedSink(FakeAudioSink):
            async def play(self, chunk: AudioChunk) -> None:
                # Hold the generation open so the cancel lands mid-flight
                # instead of after it already finished.
                if chunk.sequence == 0:
                    await gate.wait()
                await super().play(chunk)

        controller = VoicePlaybackController(provider, GatedSink(), prefetch=1)
        request = TTSRequest(text="Acht Stücke.")
        task = await controller.submit(request)
        await asyncio.sleep(0.05)
        assert controller.busy is True
        await controller.cancel(request.id)
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        return [
            t
            for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]

    assert _run(asyncio.wait_for(go(), timeout=5.0)) == []


# --- prefetch -----------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_prefetch_below_one_is_refused(bad) -> None:
    """No silent max(1, ...): the API must not accept what it cannot honour."""
    with pytest.raises(ValueError, match="prefetch"):
        VoicePlaybackController(FakeTTSProvider(), FakeAudioSink(), prefetch=bad)


def test_prefetch_one_is_the_default_and_valid() -> None:
    controller = VoicePlaybackController(FakeTTSProvider(), FakeAudioSink())
    assert controller._prefetch == 1
    VoicePlaybackController(FakeTTSProvider(), FakeAudioSink(), prefetch=3)


# --- error sanitization -------------------------------------------------------


@pytest.mark.parametrize(
    ("leak", "must_not_appear"),
    [
        ("Modell abgelehnt: token=sk-abcdef1234567890", "sk-abcdef1234567890"),
        ("Fehler bei https://internal.example.com/v1/tts?key=abc", "internal.example.com"),
        ("Datei fehlt: /home/martin/.config/kiki/secrets.toml", "/home/martin"),
        ("api_key: hunter2supersecret ungültig", "hunter2supersecret"),
    ],
)
def test_provider_errors_are_redacted_before_they_reach_the_result(leak, must_not_appear) -> None:
    async def go():
        class LeakyProvider(FakeTTSProvider):
            async def synthesize(self, request):
                raise TTSError(leak, code="leak")
                yield  # pragma: no cover

        provider = LeakyProvider()
        await provider.load()
        controller = VoicePlaybackController(provider, FakeAudioSink())
        return await controller.speak(TTSRequest(text="Egal."))

    result = _run(asyncio.wait_for(go(), timeout=5.0))
    assert result.error != ""
    assert must_not_appear not in result.error


def test_a_playback_error_is_redacted_too() -> None:
    async def go():
        provider = await _ready()

        class LeakySink(FakeAudioSink):
            async def play(self, chunk: AudioChunk) -> None:
                raise RuntimeError("Gerät /home/martin/audio.sock, token=ghp_AAAABBBBCCCCDDDD")

        controller = VoicePlaybackController(provider, LeakySink())
        return await controller.speak(TTSRequest(text="Egal."))

    result = _run(asyncio.wait_for(go(), timeout=5.0))
    assert "/home/martin" not in result.error
    assert "ghp_" not in result.error
    assert "[pfad]" in result.error or "[entfernt]" in result.error


def test_a_very_long_error_is_shortened() -> None:
    from kiki.voice.tts.controller import MAX_ERROR_CHARS, _safe_error

    text = _safe_error(RuntimeError("x" * 5000))
    assert len(text) <= MAX_ERROR_CHARS + 1


def test_an_exception_without_a_message_still_names_its_type() -> None:
    from kiki.voice.tts.controller import _safe_error

    assert _safe_error(RuntimeError()) == "RuntimeError"


# --- on_audio_started -------------------------------------------------------


def _announcing(provider, sink, **kwargs):
    """A controller that records every AudioStartedEvent it emits."""
    seen: list[AudioStartedEvent] = []
    controller = VoicePlaybackController(
        provider, sink, on_audio_started=seen.append, **kwargs
    )
    return controller, seen


def test_the_first_playable_chunk_announces_itself_exactly_once() -> None:
    provider = FakeTTSProvider(chunk_seconds=0.1)
    sink = FakeAudioSink()
    controller, seen = _announcing(provider, sink)

    async def go():
        await provider.load()
        return await controller.speak(TTSRequest(text="Ein etwas längerer Satz zum Testen."))

    result = asyncio.run(go())

    assert result.chunks > 2          # several chunks were played
    assert len(seen) == 1             # one announcement all the same
    assert seen[0].sequence == 0
    assert seen[0].timestamp_monotonic > 0


def test_the_event_carries_the_requesting_id() -> None:
    provider = FakeTTSProvider()
    controller, seen = _announcing(provider, FakeAudioSink())

    async def go():
        await provider.load()
        await controller.speak(TTSRequest(text="Wer spricht?", id="req-4711"))

    asyncio.run(go())
    assert [event.request_id for event in seen] == ["req-4711"]


def test_it_fires_before_playback_of_that_chunk_finished() -> None:
    """Announcing only after play() returned would mean announcing after the
    chunk was already over."""
    provider = FakeTTSProvider(chunk_seconds=0.1)
    order: list[str] = []

    class _Slow:
        async def play(self, chunk):
            order.append(f"play-start-{chunk.sequence}")
            await asyncio.sleep(0.01)
            order.append(f"play-end-{chunk.sequence}")

        async def stop(self) -> None: ...

        async def close(self) -> None: ...

    controller = VoicePlaybackController(
        provider, _Slow(), on_audio_started=lambda e: order.append("announced")
    )

    async def go():
        await provider.load()
        await controller.speak(TTSRequest(text="Reihenfolge."))

    asyncio.run(go())

    assert order[0] == "play-start-0"
    assert order[1] == "announced"
    assert order.index("announced") < order.index("play-end-0")


def test_a_provider_failure_before_any_chunk_announces_nothing() -> None:
    provider = FakeTTSProvider(fail_on_synthesize=True)
    controller, seen = _announcing(provider, FakeAudioSink())

    async def go():
        await provider.load()
        return await controller.speak(TTSRequest(text="Geht nicht."))

    result = asyncio.run(go())
    assert result.error
    assert seen == []


def test_a_failing_first_chunk_does_not_announce_itself() -> None:
    """That chunk made no sound, so it may not claim it did — but the next one
    that really plays must, because by then KIKI genuinely is audible."""
    provider = FakeTTSProvider(chunk_seconds=0.1)
    sink = FakeAudioSink(fail_on_sequence={0})
    controller, seen = _announcing(provider, sink)

    async def go():
        await provider.load()
        return await controller.speak(TTSRequest(text="Ein Satz mit mehreren Stücken."))

    result = asyncio.run(go())

    assert result.error
    assert len(seen) == 1
    assert seen[0].sequence == 1          # not 0: that one never sounded
    assert 0 not in sink.played_sequences


def test_a_sink_that_fails_on_every_chunk_announces_nothing() -> None:
    """No sound was ever made at all."""
    provider = FakeTTSProvider(chunk_seconds=0.1)
    sink = FakeAudioSink(fail_on_sequence=set(range(50)))
    controller, seen = _announcing(provider, sink)

    async def go():
        await provider.load()
        return await controller.speak(TTSRequest(text="Nichts geht."))

    result = asyncio.run(go())

    assert result.error
    assert sink.played == []
    assert seen == []


def test_a_failure_on_a_later_chunk_does_not_announce_again() -> None:
    provider = FakeTTSProvider(chunk_seconds=0.1)
    sink = FakeAudioSink(fail_on_sequence={2})
    controller, seen = _announcing(provider, sink)

    async def go():
        await provider.load()
        return await controller.speak(TTSRequest(text="Ein längerer Satz mit mehreren Stücken."))

    asyncio.run(go())
    assert len(seen) == 1
    assert seen[0].sequence == 0


def test_an_empty_chunk_announces_nothing() -> None:
    """An empty chunk makes no sound and may not claim to."""

    class _EmptyThenReal:
        provider_id = "empty-first"

        def __init__(self) -> None:
            self._status = TTSProviderStatus.READY

        @property
        def status(self):
            return self._status

        def capabilities(self):
            return FakeTTSProvider().capabilities()

        async def health_check(self): ...

        async def load(self) -> None: ...

        async def unload(self) -> None: ...

        async def cancel(self, request_id: str) -> None: ...

        async def synthesize(self, request):
            yield AudioChunk(request_id=request.id, sequence=0, pcm=b"")
            yield AudioChunk(request_id=request.id, sequence=1, pcm=b"\x01\x02" * 240)

    provider = _EmptyThenReal()
    controller, seen = _announcing(provider, FakeAudioSink())

    asyncio.run(controller.speak(TTSRequest(text="Erst nichts, dann etwas.")))

    assert [event.sequence for event in seen] == [1]


def test_a_cancel_before_any_audio_announces_nothing() -> None:
    provider = FakeTTSProvider(chunk_seconds=0.1, latency_s=0.2)
    controller, seen = _announcing(provider, FakeAudioSink())

    async def go():
        await provider.load()
        task = await controller.submit(TTSRequest(text="Zu spät.", id="req-cancel"))
        await asyncio.sleep(0)
        await controller.cancel("req-cancel")
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert seen == []


def test_a_cancel_during_playback_produces_no_second_event() -> None:
    provider = FakeTTSProvider(chunk_seconds=0.05)
    sink = FakeAudioSink(realtime=True)
    controller, seen = _announcing(provider, sink)

    async def go():
        await provider.load()
        request = TTSRequest(text="Ein langer Satz, der unterbrochen wird.", id="req-mid")
        task = await controller.submit(request)
        for _ in range(200):
            if seen:
                break
            await asyncio.sleep(0.001)
        await controller.cancel("req-mid")
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert len(seen) == 1


def test_a_superseded_request_can_never_announce_itself() -> None:
    """The chunk of an answer the user already talked over must not reach the
    speakers, and must not report that it did."""
    provider = FakeTTSProvider(chunk_seconds=0.05)
    sink = FakeAudioSink(realtime=True)
    controller, seen = _announcing(provider, sink)

    async def go():
        await provider.load()
        first = await controller.submit(TTSRequest(text="Erste Antwort.", id="req-alt"))
        await asyncio.sleep(0)
        second = await controller.submit(TTSRequest(text="Zweite Antwort.", id="req-neu"))
        with contextlib.suppress(asyncio.CancelledError):
            await first
        await second

    asyncio.run(go())
    assert [event.request_id for event in seen] == ["req-neu"]


def test_the_next_request_after_a_cancel_announces_again() -> None:
    provider = FakeTTSProvider(chunk_seconds=0.05)
    controller, seen = _announcing(provider, FakeAudioSink())

    async def go():
        await provider.load()
        await controller.cancel("nie-gesehen")
        await controller.speak(TTSRequest(text="Erster.", id="a"))
        await controller.speak(TTSRequest(text="Zweiter.", id="b"))

    asyncio.run(go())
    assert [event.request_id for event in seen] == ["a", "b"]


def test_shutdown_produces_no_late_event() -> None:
    provider = FakeTTSProvider(chunk_seconds=0.05, latency_s=0.2)
    controller, seen = _announcing(provider, FakeAudioSink())

    async def go():
        await provider.load()
        task = await controller.submit(TTSRequest(text="Wird beendet."))
        await asyncio.sleep(0)
        await controller.shutdown()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)

    asyncio.run(go())
    assert seen == []


def test_a_listener_that_raises_does_not_silence_the_answer() -> None:
    provider = FakeTTSProvider(chunk_seconds=0.1)
    sink = FakeAudioSink()

    def _boom(_event):
        raise RuntimeError("Zuhörer kaputt")

    controller = VoicePlaybackController(provider, sink, on_audio_started=_boom)

    async def go():
        await provider.load()
        return await controller.speak(TTSRequest(text="Trotzdem sprechen."))

    result = asyncio.run(go())

    assert result.error == ""
    assert result.chunks > 0


def test_no_listener_at_all_is_fine() -> None:
    provider = FakeTTSProvider(chunk_seconds=0.1)
    controller = VoicePlaybackController(provider, FakeAudioSink())

    async def go():
        await provider.load()
        return await controller.speak(TTSRequest(text="Ohne Zuhörer."))

    assert asyncio.run(go()).chunks > 0


def test_cancelling_the_run_task_with_a_full_queue_terminates() -> None:
    """Regression: _run cancelled the producer and awaited it *before* draining,
    so the producer's own sentinel put blocked on a full queue and the await
    never returned. Surfaced when the consumer gained one extra loop turn."""
    provider = FakeTTSProvider(chunk_seconds=0.05)
    gate = asyncio.Event()

    class _Gated:
        async def play(self, chunk):
            await gate.wait()

        async def stop(self) -> None: ...

        async def close(self) -> None: ...

    controller = VoicePlaybackController(provider, _Gated(), prefetch=1)

    async def go():
        await provider.load()
        task = await controller.submit(TTSRequest(text="Ein langer Satz für viele Stücke."))
        for _ in range(50):
            await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)

    asyncio.run(go())
