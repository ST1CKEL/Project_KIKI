"""The opt-in controller route inside SpeechDirector.

The flag decides which of two routes a director drives. Everything here either
proves the default route is untouched, or that the new one behaves the same way
towards the UI while using none of the file-based machinery.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from kiki.config.settings import default_mapping, load_settings, settings_from_mapping
from kiki.voice.director import SpeechDirector
from kiki.voice.tts import TTSError, TTSGenerationResult, TTSRequest

# --- doubles ----------------------------------------------------------------


class _FakePlayer:
    def __init__(self) -> None:
        self.played: list[Path] = []
        self.texts: list[str] = []
        self.stopped = 0
        self._eos = None

    def play(self, path: Path, *, on_eos=None, on_error=None) -> None:
        self.played.append(path)
        # Read now: the director unlinks the WAV the moment EOS arrives.
        try:
            self.texts.append(path.read_text(encoding="utf-8"))
        except OSError:
            self.texts.append("")
        self._eos = on_eos

    def stop(self) -> None:
        self.stopped += 1
        self._eos = None

    def finish(self) -> None:
        callback, self._eos = self._eos, None
        if callback is not None:
            callback()


class _FakeController:
    """Records requests and lets a test decide when an utterance finishes."""

    def __init__(self, *, auto_finish: bool = True, error: str = "",
                 raises: BaseException | None = None) -> None:
        self.requests: list[TTSRequest] = []
        self.interrupts = 0
        self._auto = auto_finish
        self._error = error
        self._raises = raises
        self._gate: asyncio.Event | None = None

    async def speak(self, request: TTSRequest) -> TTSGenerationResult:
        self.requests.append(request)
        if self._raises is not None:
            raise self._raises
        if not self._auto:
            self._gate = asyncio.Event()
            await self._gate.wait()
        return TTSGenerationResult(request_id=request.id, chunks=1, error=self._error)

    async def interrupt(self) -> bool:
        self.interrupts += 1
        gate, self._gate = self._gate, None
        if gate is not None:
            gate.set()
        return True

    @property
    def texts(self) -> list[str]:
        return [request.text for request in self.requests]


class _Handle:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def _sync_submit(coro, *, on_success=None, on_error=None, on_complete=None) -> _Handle:
    """Stands in for AsyncBridge.submit: runs the coroutine, calls back inline."""
    handle = _Handle()
    try:
        result = asyncio.run(coro)
    except Exception as exc:
        if on_error is not None:
            on_error(exc)
        if on_complete is not None:
            on_complete()
        return handle
    if on_success is not None:
        on_success(result)
    if on_complete is not None:
        on_complete()
    return handle


class _DeferredSubmit:
    """Queues coroutines instead of running them, so a test can stop() first."""

    def __init__(self, *, timeout: float = 5.0) -> None:
        self.pending: list[tuple] = []
        self.handles: list[_Handle] = []
        self._timeout = timeout

    def __call__(self, coro, *, on_success=None, on_error=None, on_complete=None) -> _Handle:
        handle = _Handle()
        self.handles.append(handle)
        self.pending.append((coro, on_success, on_error, on_complete))
        return handle

    def run_all(self) -> None:
        """Run everything queued in **one** loop.

        Sequentially would deadlock the way the bridge never does: a speak()
        that waits for an interrupt would never see the interrupt that is still
        sitting behind it in the queue.
        """

        async def _one(coro, on_success, on_error, on_complete) -> None:
            try:
                result = await coro
            except Exception as exc:
                if on_error is not None:
                    on_error(exc)
            else:
                if on_success is not None:
                    on_success(result)
            if on_complete is not None:
                on_complete()

        async def _go() -> None:
            running: list[asyncio.Task] = []
            while self.pending or running:
                while self.pending:
                    running.append(asyncio.create_task(_one(*self.pending.pop(0))))
                _done, rest = await asyncio.wait(
                    running, timeout=self._timeout, return_when=asyncio.FIRST_COMPLETED
                )
                if not _done:
                    for task in rest:
                        task.cancel()
                    raise AssertionError("Eine übergebene Coroutine wurde nie fertig")
                running = list(rest)

        asyncio.run(_go())

    def discard(self) -> None:
        """Close what a test deliberately never ran."""
        for coro, *_rest in self.pending:
            coro.close()
        self.pending.clear()


async def _synth_ok(text: str, dest: Path) -> Path:
    dest.write_text(text, encoding="utf-8")
    return dest


def _director(tmp_path: Path, *, controller=None, flag: bool = False, submit=_sync_submit,
              events: list[str] | None = None, player=None) -> tuple[SpeechDirector, _FakePlayer]:
    player = player or _FakePlayer()
    log = events if events is not None else []
    director = SpeechDirector(
        synthesize=_synth_ok,
        player=player,
        submit=submit,
        wav_dir=tmp_path,
        on_speaking=lambda: log.append("speaking"),
        on_idle=lambda: log.append("idle"),
        on_error=lambda exc: log.append(f"error:{exc}"),
        controller=controller,
        use_controller_route=flag,
    )
    return director, player


# --- the flag ---------------------------------------------------------------


def test_the_flag_defaults_to_false() -> None:
    assert settings_from_mapping(default_mapping()).tts.use_controller_route is False


def test_the_shipped_defaults_file_says_false() -> None:
    assert default_mapping()["tts"]["use_controller_route"] is False


def test_a_user_config_without_the_key_still_gets_the_default(tmp_path: Path) -> None:
    """The failure that bit ai.system_prompt: a saved config shadowing a default.
    A table merges key by key, so an older config must keep working."""
    config = tmp_path / "config.toml"
    config.write_text(
        '[tts]\nspeaker = "Vivian"\nfallback_to_system = false\n', encoding="utf-8"
    )
    settings = load_settings(config)

    assert settings.tts.use_controller_route is False
    assert settings.tts.speaker == "Vivian"          # the user's key survives
    assert settings.tts.base_url == "http://127.0.0.1:18765"  # the default too


def test_a_user_config_can_switch_the_route_on(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[tts]\nuse_controller_route = true\n", encoding="utf-8")

    assert load_settings(config).tts.use_controller_route is True


def test_a_mapping_that_predates_the_key_does_not_switch_it_on() -> None:
    mapping = default_mapping()
    del mapping["tts"]["use_controller_route"]

    assert settings_from_mapping(mapping).tts.use_controller_route is False


def test_the_flag_survives_a_save_and_reload(tmp_path: Path) -> None:
    from kiki.config.settings import save_settings

    settings = settings_from_mapping(default_mapping())
    settings.tts.use_controller_route = True
    target = tmp_path / "config.toml"
    save_settings(settings, target)

    assert load_settings(target).tts.use_controller_route is True


# --- routing ----------------------------------------------------------------


def test_with_the_flag_off_the_file_route_still_runs(tmp_path: Path) -> None:
    controller = _FakeController()
    director, player = _director(tmp_path, controller=controller, flag=False)
    director.begin()
    director.feed("Hallo Welt. Zweiter Satz!")
    director.flush()

    assert controller.requests == []
    assert player.texts == ["Hallo Welt."]


def test_a_flag_without_a_controller_keeps_the_file_route(tmp_path: Path) -> None:
    """Half a configuration must not turn speech off."""
    director, player = _director(tmp_path, controller=None, flag=True)
    director.say("Hallo")

    assert len(player.played) == 1


def test_with_the_flag_on_the_controller_speaks(tmp_path: Path) -> None:
    controller = _FakeController()
    director, player = _director(tmp_path, controller=controller, flag=True)
    director.say("Hallo Martin.")

    assert controller.texts == ["Hallo Martin."]
    assert player.played == []


def test_one_request_per_sentence_never_one_per_answer(tmp_path: Path) -> None:
    """The whole point of keeping split_ready() in the director: the controller
    must get sentences, not a collected answer."""
    controller = _FakeController()
    director, _player = _director(tmp_path, controller=controller, flag=True)
    director.begin()
    director.feed("Erster Satz. Zweiter Satz! Dritter Satz?")
    director.flush()

    assert controller.texts == ["Erster Satz.", "Zweiter Satz!", "Dritter Satz?"]
    assert len({request.id for request in controller.requests}) == 3


def test_a_trailing_fragment_is_flushed_as_its_own_request(tmp_path: Path) -> None:
    controller = _FakeController()
    director, _player = _director(tmp_path, controller=controller, flag=True)
    director.begin()
    director.feed("Ein Satz. Und ein Rest ohne Punkt")
    director.flush()

    assert controller.texts == ["Ein Satz.", "Und ein Rest ohne Punkt"]


def test_the_controller_route_touches_no_wav_machinery(tmp_path: Path) -> None:
    controller = _FakeController()
    director, player = _director(tmp_path, controller=controller, flag=True)
    director.begin()
    director.feed("Erster Satz. Zweiter Satz!")
    director.flush()

    assert list(director._play_queue) == []
    assert director._synth_busy is False
    assert director._playing is False
    assert director._active_synth_path is None
    assert player.played == []
    assert list(tmp_path.glob("*.wav")) == []


def test_sentences_are_serialised_not_superseded(tmp_path: Path) -> None:
    """submit() on the controller aborts whatever runs, so a director that
    dispatched sentence two early would cut sentence one off mid-word."""
    controller = _FakeController(auto_finish=False)
    submit = _DeferredSubmit()
    director, _player = _director(tmp_path, controller=controller, flag=True, submit=submit)
    director.begin()
    director.feed("Erster Satz. Zweiter Satz!")
    director.flush()

    assert len(submit.pending) == 1  # only one utterance is in flight
    submit.discard()


# --- UI signals -------------------------------------------------------------


def test_the_signal_sequence_matches_the_file_route(tmp_path: Path) -> None:
    controller = _FakeController()
    events: list[str] = []
    director, _player = _director(tmp_path, controller=controller, flag=True, events=events)
    director.begin()
    director.feed("Hallo Welt.")
    assert events == ["speaking"]
    assert director.active is True

    director.flush()
    assert events == ["speaking", "idle"]
    assert director.active is False


def test_active_stays_true_while_an_utterance_runs(tmp_path: Path) -> None:
    controller = _FakeController()
    submit = _DeferredSubmit()
    director, _player = _director(tmp_path, controller=controller, flag=True, submit=submit)
    director.say("Ein langer Satz.")

    assert director.active is True
    submit.run_all()
    assert director.active is False


def test_a_failing_utterance_reports_and_returns_to_idle(tmp_path: Path) -> None:
    controller = _FakeController(raises=TTSError("TTS-Dienst nicht erreichbar", code="unreachable"))
    events: list[str] = []
    director, _player = _director(tmp_path, controller=controller, flag=True, events=events)
    director.say("Hallo")

    assert events == ["speaking", "error:TTS-Dienst nicht erreichbar", "idle"]
    assert director.active is False


def test_an_error_inside_the_result_is_reported_too(tmp_path: Path) -> None:
    """The controller returns failures in the result rather than raising."""
    controller = _FakeController(error="Wiedergabe fehlgeschlagen")
    events: list[str] = []
    director, _player = _director(tmp_path, controller=controller, flag=True, events=events)
    director.say("Hallo")

    assert "error:Wiedergabe fehlgeschlagen" in events
    assert director.active is False


def test_a_single_bad_sentence_does_not_stop_the_rest(tmp_path: Path) -> None:
    calls = {"n": 0}

    class _FlakyController(_FakeController):
        async def speak(self, request):
            self.requests.append(request)
            calls["n"] += 1
            if calls["n"] == 1:
                raise TTSError("Unerwarteter TTS-Inhalt: text/html", code="format")
            return TTSGenerationResult(request_id=request.id, chunks=1)

    controller = _FlakyController()
    director, _player = _director(tmp_path, controller=controller, flag=True)
    director.begin()
    director.feed("Erster Satz. Zweiter Satz!")
    director.flush()

    assert controller.texts == ["Erster Satz.", "Zweiter Satz!"]


# --- stop / barge-in --------------------------------------------------------


def test_stop_right_after_submit_starts_nothing_further(tmp_path: Path) -> None:
    controller = _FakeController()
    submit = _DeferredSubmit()
    director, _player = _director(tmp_path, controller=controller, flag=True, submit=submit)
    director.begin()
    director.feed("Erster Satz. Zweiter Satz!")

    director.stop()
    submit.run_all()

    # The first utterance was already handed over; nothing after it may run.
    assert controller.texts == ["Erster Satz."]
    assert director.active is False


def test_stop_during_synthesis_cancels_and_goes_idle(tmp_path: Path) -> None:
    controller = _FakeController(auto_finish=False)
    submit = _DeferredSubmit()
    events: list[str] = []
    director, _player = _director(
        tmp_path, controller=controller, flag=True, submit=submit, events=events
    )
    director.say("Ein langer Satz.")
    assert director.active is True

    director.stop()

    assert submit.handles[0].cancelled is True
    assert "idle" in events
    assert director.active is False
    submit.discard()


def test_stop_hands_the_interrupt_to_the_bridge(tmp_path: Path) -> None:
    """Synchronously: generation invalidated. Asynchronously: the sink is
    silenced through the bridge, so the GTK thread never waits on audio."""
    controller = _FakeController(auto_finish=False)
    submit = _DeferredSubmit()
    director, _player = _director(tmp_path, controller=controller, flag=True, submit=submit)
    director.say("Ein langer Satz.")

    director.stop()

    assert director.active is False        # synchronous half, before any await
    assert controller.interrupts == 0      # asynchronous half, not yet run
    submit.run_all()
    assert controller.interrupts == 1


def test_stop_during_playback_lets_the_late_result_die(tmp_path: Path) -> None:
    """The utterance finishes after stop(); its callback must change nothing."""
    controller = _FakeController(auto_finish=False)
    submit = _DeferredSubmit()
    events: list[str] = []
    director, _player = _director(
        tmp_path, controller=controller, flag=True, submit=submit, events=events
    )
    director.begin()
    director.feed("Erster Satz. Zweiter Satz!")
    director.stop()
    before = list(events)

    submit.run_all()

    assert controller.texts == ["Erster Satz."]
    assert events.count("idle") == before.count("idle")
    assert director.active is False


def test_stop_when_nothing_speaks_is_harmless(tmp_path: Path) -> None:
    controller = _FakeController()
    director, _player = _director(tmp_path, controller=controller, flag=True)
    director.stop()
    director.stop()

    assert controller.requests == []
    assert director.active is False


def test_say_after_stop_speaks_again(tmp_path: Path) -> None:
    controller = _FakeController()
    director, _player = _director(tmp_path, controller=controller, flag=True)
    director.say("Erster.")
    director.stop()
    director.say("Zweiter.")

    assert controller.texts == ["Erster.", "Zweiter."]
    assert director.active is False


# --- against the real controller and the real adapters ----------------------


def test_the_real_chain_runs_inside_one_event_loop(tmp_path: Path) -> None:
    """Everything above uses a fake controller. This one wires the actual
    VoicePlaybackController, ServiceTTSProvider and PipeWireAudioSink together,
    with only the HTTP call and the player replaced."""
    import wave

    from kiki.voice.tts.adapters import PipeWireAudioSink, ServiceTTSProvider
    from kiki.voice.tts.controller import VoicePlaybackController
    from kiki.voice.tts_client import TtsHealth

    spoken: list[str] = []

    async def _synth(base_url, text, *, dest, language, speaker, timeout):
        spoken.append(text)
        with wave.open(str(dest), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24_000)
            wav.writeframes(b"\x01\x02" * 2400)
        return Path(dest)

    async def _health(base_url, **_kwargs):
        return TtsHealth(ok=True, ready=True, detail="")

    class _Player:
        def __init__(self) -> None:
            self.count = 0

        def play(self, path, *, on_eos=None, on_error=None):
            self.count += 1
            if on_eos is not None:
                on_eos()

        def stop(self) -> None:
            pass

    async def go():
        provider = ServiceTTSProvider(
            synthesize=_synth, health=_health, wav_dir=tmp_path / "tts", chunk_seconds=0.05
        )
        await provider.load()
        player = _Player()
        controller = VoicePlaybackController(
            provider, PipeWireAudioSink(player, wav_dir=tmp_path / "sink")
        )
        events: list[str] = []
        loop = asyncio.get_running_loop()
        finished = asyncio.Event()

        def _on_idle() -> None:
            events.append("idle")
            finished.set()

        def _submit(coro, *, on_success=None, on_error=None, on_complete=None):
            async def _run():
                try:
                    result = await coro
                except Exception as exc:
                    if on_error is not None:
                        on_error(exc)
                else:
                    if on_success is not None:
                        on_success(result)
                if on_complete is not None:
                    on_complete()

            loop.create_task(_run())
            return _Handle()

        director = SpeechDirector(
            synthesize=_synth_ok,
            player=_FakePlayer(),
            submit=_submit,
            wav_dir=tmp_path / "unused",
            on_speaking=lambda: events.append("speaking"),
            on_idle=_on_idle,
            on_error=lambda exc: events.append(f"error:{exc}"),
            controller=controller,
            use_controller_route=True,
        )
        director.say("Guten Abend.")
        await asyncio.wait_for(finished.wait(), timeout=5)
        await controller.shutdown()
        return spoken, player.count, events

    texts, played, events = asyncio.run(go())

    assert texts == ["Guten Abend."]
    assert played == 2                      # 0.1 s of audio in 0.05 s chunks
    assert events[0] == "speaking"
    assert not any(event.startswith("error") for event in events)
    assert list((tmp_path / "sink").glob("*.wav")) == []


# --- the file route is provably untouched -----------------------------------


@pytest.mark.parametrize("controller", [None, _FakeController()])
def test_the_file_route_behaves_identically_with_and_without_a_controller(
    tmp_path: Path, controller
) -> None:
    events: list[str] = []
    director, player = _director(tmp_path, controller=controller, flag=False, events=events)
    director.begin()
    director.feed("Hallo Welt. Zweiter Satz!")
    director.flush()

    assert player.texts == ["Hallo Welt."]
    assert events == ["speaking"]
    player.finish()
    assert player.texts == ["Hallo Welt.", "Zweiter Satz!"]
    player.finish()
    assert events[-1] == "idle"
    assert director.active is False


# --- what the signals actually promise --------------------------------------


def test_on_speaking_precedes_the_first_audio_chunk(tmp_path: Path) -> None:
    """On the controller route `on_speaking` means "accepted", not "audible".

    The order is pinned here so the day someone wires the character animation to
    real audio (`on_audio_started`, see docs/ARCHITECTURE.md) the change is
    visible as a failing test rather than a silent behaviour shift.
    """
    import wave

    from kiki.voice.tts.adapters import PipeWireAudioSink, ServiceTTSProvider
    from kiki.voice.tts.controller import VoicePlaybackController
    from kiki.voice.tts_client import TtsHealth

    timeline: list[str] = []

    async def _synth(base_url, text, *, dest, language, speaker, timeout):
        timeline.append("synthesis")
        with wave.open(str(dest), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24_000)
            wav.writeframes(b"\x01\x02" * 2400)
        return Path(dest)

    async def _health(base_url, **_kwargs):
        return TtsHealth(ok=True, ready=True, detail="")

    class _Player:
        def play(self, path, *, on_eos=None, on_error=None):
            timeline.append("audio")
            if on_eos is not None:
                on_eos()

        def stop(self) -> None:
            pass

    async def go():
        provider = ServiceTTSProvider(
            synthesize=_synth, health=_health, wav_dir=tmp_path / "tts", chunk_seconds=0.05
        )
        await provider.load()
        controller = VoicePlaybackController(
            provider, PipeWireAudioSink(_Player(), wav_dir=tmp_path / "sink")
        )
        loop = asyncio.get_running_loop()
        finished = asyncio.Event()

        def _submit(coro, *, on_success=None, on_error=None, on_complete=None):
            async def _run():
                try:
                    result = await coro
                except Exception as exc:
                    if on_error is not None:
                        on_error(exc)
                else:
                    if on_success is not None:
                        on_success(result)
                if on_complete is not None:
                    on_complete()

            loop.create_task(_run())
            return _Handle()

        def _idle() -> None:
            timeline.append("idle")
            finished.set()

        director = SpeechDirector(
            synthesize=_synth_ok,
            player=_FakePlayer(),
            submit=_submit,
            wav_dir=tmp_path / "unused",
            on_speaking=lambda: timeline.append("speaking"),
            on_idle=_idle,
            controller=controller,
            use_controller_route=True,
        )
        director.say("Guten Abend.")
        await asyncio.wait_for(finished.wait(), timeout=5)
        await controller.shutdown()

    asyncio.run(go())

    assert timeline[0] == "speaking"
    # Synthesis has not even started when the signal fires — claiming audio
    # exists at this point would be false.
    assert timeline.index("speaking") < timeline.index("synthesis")
    assert timeline.index("synthesis") < timeline.index("audio")
    assert timeline[-1] == "idle"


def test_on_idle_waits_for_the_utterance_to_finish(tmp_path: Path) -> None:
    controller = _FakeController()
    submit = _DeferredSubmit()
    events: list[str] = []
    director, _player = _director(
        tmp_path, controller=controller, flag=True, submit=submit, events=events
    )
    director.say("Ein Satz.")

    assert events == ["speaking"]          # accepted, not finished
    submit.run_all()
    assert events == ["speaking", "idle"]


def test_on_idle_also_follows_a_cancel(tmp_path: Path) -> None:
    controller = _FakeController(auto_finish=False)
    submit = _DeferredSubmit()
    events: list[str] = []
    director, _player = _director(
        tmp_path, controller=controller, flag=True, submit=submit, events=events
    )
    director.say("Ein Satz.")
    director.stop()

    assert events == ["speaking", "idle"]
    submit.run_all()
    assert events == ["speaking", "idle"]  # the late result adds nothing


def test_a_late_result_never_announces_speaking_again(tmp_path: Path) -> None:
    """A callback for a superseded generation must stay silent — otherwise the
    character would start talking after the user already interrupted."""
    controller = _FakeController()
    submit = _DeferredSubmit()
    events: list[str] = []
    director, _player = _director(
        tmp_path, controller=controller, flag=True, submit=submit, events=events
    )
    director.begin()
    director.feed("Erster Satz. Zweiter Satz! Dritter Satz?")
    director.stop()
    after_stop = list(events)

    submit.run_all()

    assert events == after_stop
    assert events.count("speaking") == 1
    assert director.active is False


def test_the_file_route_announces_speaking_at_playback(tmp_path: Path) -> None:
    """The contrast the docs describe: here the WAV exists before the signal."""
    order: list[str] = []

    class _Watching(_FakePlayer):
        def play(self, path, *, on_eos=None, on_error=None):
            order.append(f"audio:{path.is_file()}")
            super().play(path, on_eos=on_eos, on_error=on_error)

    director, _player = _director(
        tmp_path, controller=None, flag=False, player=_Watching(),
        events=order,
    )
    director.say("Hallo")

    assert order == ["audio:True", "speaking"]


# --- shutdown ---------------------------------------------------------------


def test_the_route_can_be_disabled_after_a_failed_load(tmp_path: Path) -> None:
    """A provider that never loaded fails every sentence. Falling back beats
    losing speech, and the flag must not switch itself back on."""
    controller = _FakeController()
    director, player = _director(tmp_path, controller=controller, flag=True)
    assert director.uses_controller_route is True

    director.disable_controller_route()

    assert director.uses_controller_route is False
    director.say("Hallo")
    assert controller.requests == []
    assert len(player.played) == 1


def test_disabling_the_route_is_idempotent(tmp_path: Path) -> None:
    controller = _FakeController()
    director, _player = _director(tmp_path, controller=controller, flag=True)
    director.disable_controller_route()
    director.disable_controller_route()

    assert director.uses_controller_route is False


def test_the_application_closes_the_controller_at_shutdown() -> None:
    """The bridge cancels stray tasks, but only shutdown() closes the sink —
    and an unclosed sink leaves its temporary directory behind on every run."""
    from kiki.runtime.async_bridge import AsyncBridge

    class _Recording:
        def __init__(self) -> None:
            self.shutdowns = 0

        async def shutdown(self) -> None:
            self.shutdowns += 1

    controller = _Recording()
    bridge = AsyncBridge()
    bridge.start()
    try:
        closer = _make_closer(bridge, controller)
        closer()
        deadline = time.monotonic() + 3
        while controller.shutdowns == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert controller.shutdowns == 1
        closer()  # idempotent: the reference is already gone
        time.sleep(0.1)
        assert controller.shutdowns == 1
    finally:
        bridge.stop()


def _make_closer(bridge, controller):
    """Runs the real `_close_voice_controller` body against a bare object.

    Importing KikiApplication would pull in GTK and construct a window, so the
    method is bound to a stand-in that carries only the two attributes it uses.
    """
    from kiki.application import KikiApplication

    class _Stub:
        pass

    stub = _Stub()
    stub._bridge = bridge
    stub._voice_controller = controller
    return lambda: KikiApplication._close_voice_controller(stub)


def test_closing_without_a_controller_does_nothing() -> None:
    from kiki.runtime.async_bridge import AsyncBridge

    bridge = AsyncBridge()
    closer = _make_closer(bridge, None)
    closer()  # the bridge was never started; this must not raise


# --- the flag must never take the app down ----------------------------------


class _AppStub:
    """Only the attributes `_build_voice_controller` touches."""

    def __init__(self, *, url: str = "http://127.0.0.1:18765") -> None:
        self._voice_controller = None
        self._speech = None
        self._submitted: list[object] = []
        outer = self

        class _Tts:
            use_controller_route = True
            base_url = url
            speaker = "Serena"
            language = "German"

        class _Settings:
            tts = _Tts()

        class _Bridge:
            def submit(self, coro, **_kwargs):
                outer._submitted.append(coro)
                coro.close()
                return None

        self._settings = _Settings()
        self._bridge = _Bridge()


def _build(stub) -> object | None:
    from kiki.application import KikiApplication

    return KikiApplication._build_voice_controller(stub)


def test_a_broken_adapter_import_does_not_stop_the_app(monkeypatch, caplog) -> None:
    import sys
    import types

    monkeypatch.setitem(
        sys.modules, "kiki.voice.tts.adapters", types.ModuleType("kiki.voice.tts.adapters")
    )
    stub = _AppStub()
    with caplog.at_level("ERROR"):
        controller = _build(stub)

    assert controller is None
    assert stub._voice_controller is None
    assert "staying on the old one" in caplog.text


def test_the_failure_log_carries_no_configuration_values(monkeypatch, caplog) -> None:
    import sys
    import types

    monkeypatch.setitem(
        sys.modules, "kiki.voice.tts.adapters", types.ModuleType("kiki.voice.tts.adapters")
    )
    stub = _AppStub(url="https://secret.internal:18765/v1?token=sk-live-4711")
    with caplog.at_level("DEBUG"):
        _build(stub)

    assert "secret.internal" not in caplog.text
    assert "sk-live-4711" not in caplog.text


def test_a_controller_that_never_built_leaves_the_file_route_running(tmp_path: Path) -> None:
    """The end of that chain: controller=None plus flag=True is the file route,
    not silence."""
    director, player = _director(tmp_path, controller=None, flag=True)
    assert director.uses_controller_route is False
    director.say("Hallo")

    assert player.texts == ["Hallo"]


def test_the_flag_is_never_switched_on_again_by_a_failure(tmp_path: Path) -> None:
    controller = _FakeController()
    director, _player = _director(tmp_path, controller=controller, flag=True)
    director.disable_controller_route()
    director.begin()
    director.feed("Erster Satz. Zweiter Satz!")
    director.flush()

    assert director.uses_controller_route is False
    assert controller.requests == []
