"""Wake-word matching and the listener state machine. No audio hardware."""

from __future__ import annotations

import pytest

from kiki.voice.wake import (
    DEFAULT_PHRASES,
    ListenerState,
    UtteranceStream,
    WakeError,
    WakeWordListener,
    missing_words,
    normalize,
    phrase_matches,
)


class FakeStream(UtteranceStream):
    """Replays scripted utterances instead of decoding audio."""

    def __init__(self, utterances: list[str | None]) -> None:
        super().__init__(model_dir="/nonexistent")
        self._queue = list(utterances)
        self.opened = False
        self.resets = 0

    def open(self, *, phrases=()) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def reset(self) -> None:
        self.resets += 1

    def feed(self, pcm: bytes) -> str | None:
        return self._queue.pop(0) if self._queue else None


def _listener(**kwargs) -> tuple[WakeWordListener, list[str], list[str]]:
    detected: list[str] = []
    commands: list[str] = []
    params = dict(
        stream=FakeStream([]),
        microphone=object(),
        on_detect=lambda: detected.append("wake"),
        on_command=commands.append,
    )
    params.update(kwargs)
    return WakeWordListener(**params), detected, commands


# --- matching ---------------------------------------------------------------


@pytest.mark.parametrize(
    "heard",
    ["kiki", "KIKI", "  kiki  ", "hey kiki", "hallo kiki", "o k kiki", "kiki bitte hilf mir"],
)
def test_wake_phrase_is_recognized(heard) -> None:
    assert phrase_matches(heard, DEFAULT_PHRASES) is True


@pytest.mark.parametrize(
    "heard",
    [
        "",
        "ich habe eine kiwi gekauft",
        "der schlüssel liegt auf dem tisch",
        "die kita öffnet um acht uhr",
        "kikiriki",
        "makaki",
        "der key ist abgelaufen",
    ],
)
def test_ordinary_speech_does_not_wake(heard) -> None:
    """Regression guard for the corpus that ruled out Vosk grammar mode."""
    assert phrase_matches(heard, DEFAULT_PHRASES) is False


def test_multi_word_phrase_needs_all_words_in_order() -> None:
    phrases = ("hey kiki",)
    assert phrase_matches("hey kiki", phrases) is True
    assert phrase_matches("sag mal hey kiki bitte", phrases) is True
    assert phrase_matches("kiki hey", phrases) is False
    assert phrase_matches("hey", phrases) is False


def test_normalize_collapses_case_and_whitespace() -> None:
    assert normalize("  Hey   KIKI \n") == "hey kiki"
    assert normalize(None) == ""


def test_missing_words_reports_unknown_lexicon_entries() -> None:
    class Lexicon:
        def FindWord(self, word):  # noqa: N802 - Vosk compatibility
            return 1 if word in {"kiki", "hey"} else -1

    assert missing_words(Lexicon(), ["hey kiki"]) == []
    assert missing_words(Lexicon(), ["hey zyzzyx"]) == ["zyzzyx"]


def test_missing_words_tolerates_a_runtime_without_the_symbol() -> None:
    class Old:
        def FindWord(self, word):  # noqa: N802
            raise AttributeError("not available")

    assert missing_words(Old(), ["kiki"]) == []


def test_empty_phrase_list_is_refused() -> None:
    with pytest.raises(WakeError):
        _listener(phrases=())
    with pytest.raises(WakeError):
        _listener(phrases=("", "   "))


# --- state machine ----------------------------------------------------------


def test_wake_then_command_is_handed_over() -> None:
    listener, detected, commands = _listener()

    listener.handle("wie ist das wetter", now=0.0)
    assert detected == [] and commands == []
    assert listener.state is ListenerState.WAITING

    listener.handle("kiki", now=1.0)
    assert detected == ["wake"]
    assert listener.state is ListenerState.CAPTURING

    listener.handle("wie voll ist die platte", now=2.0)
    assert commands == ["wie voll ist die platte"]
    assert listener.state is ListenerState.WAITING


def test_non_wake_speech_never_escapes_the_listener() -> None:
    listener, detected, commands = _listener()
    for text in ["guten morgen", "ich habe eine kiwi gekauft", "der key ist abgelaufen"]:
        listener.handle(text, now=0.0)
    assert detected == []
    assert commands == []


def test_cooldown_suppresses_an_immediate_second_wake() -> None:
    listener, detected, commands = _listener(cooldown_ms=2000)

    listener.handle("kiki", now=0.0)
    listener.handle("mach das licht an", now=0.5)
    assert commands == ["mach das licht an"]

    # Inside the cooldown that follows the command.
    listener.handle("kiki", now=1.0)
    assert detected == ["wake"]

    listener.handle("kiki", now=5.0)
    assert detected == ["wake", "wake"]


def test_capture_times_out_back_to_waiting() -> None:
    listener, detected, commands = _listener(command_timeout_s=10.0, cooldown_ms=0)

    listener.handle("kiki", now=0.0)
    assert listener.state is ListenerState.CAPTURING

    # Silence ticks the machine without delivering an utterance.
    listener.handle(None, now=20.0)
    assert listener.state is ListenerState.WAITING

    # A late utterance is treated as ordinary speech, not as a command.
    listener.handle("das war nur ein gespräch", now=21.0)
    assert commands == []


def test_empty_command_is_still_delivered() -> None:
    """The app decides how to report "nothing understood"; the listener does not."""
    listener, _detected, commands = _listener()
    listener.handle("kiki", now=0.0)
    listener.handle("", now=1.0)
    assert commands == [""]
    assert listener.state is ListenerState.WAITING


def test_resume_clears_capture_state_and_resets_the_stream() -> None:
    stream = FakeStream([])
    listener, _detected, commands = _listener(stream=stream)

    listener.handle("kiki", now=0.0)
    assert listener.state is ListenerState.CAPTURING

    listener.pause()
    assert listener.paused is True
    listener.resume()

    assert listener.paused is False
    assert stream.resets == 1
    # Audio heard while KIKI was speaking must not become a command.
    assert listener.state is ListenerState.WAITING
    assert commands == []


def test_none_utterance_does_not_wake() -> None:
    listener, detected, _commands = _listener()
    listener.handle(None, now=0.0)
    listener.handle(None, now=1.0)
    assert detected == []


def test_custom_phrase_replaces_the_default() -> None:
    listener, detected, _commands = _listener(phrases=("computer",))
    listener.handle("kiki", now=0.0)
    assert detected == []
    listener.handle("computer", now=1.0)
    assert detected == ["wake"]
