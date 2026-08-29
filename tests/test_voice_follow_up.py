"""Follow-up opens only for the completed voice turn that requested it."""

from __future__ import annotations

from types import SimpleNamespace

from kiki.character.state_machine import CharacterState, CharacterStateMachine
from kiki.config.settings import Settings
from kiki.voice.follow_up import FollowUpTurn


def test_turn_needs_terminal_and_delivered_answer() -> None:
    turn = FollowUpTurn()
    turn.begin(enabled=True)

    assert not turn.consume_ready()
    assert not turn.mark_response_delivered()

    turn.mark_terminal()
    assert turn.mark_response_delivered()
    assert turn.consume_ready()
    assert not turn.consume_ready()


def test_cancelled_or_disabled_turn_never_opens_follow_up() -> None:
    disabled = FollowUpTurn()
    disabled.begin(enabled=False)
    disabled.mark_terminal()
    disabled.mark_response_delivered()
    assert not disabled.consume_ready()

    cancelled = FollowUpTurn()
    cancelled.begin(enabled=True)
    cancelled.mark_terminal(cancelled=True)
    cancelled.mark_response_delivered()
    assert not cancelled.consume_ready()


class _Wake:
    def __init__(self) -> None:
        self.captures = 0

    def capture_next(self) -> bool:
        self.captures += 1
        return True


class _Chat:
    def __init__(self) -> None:
        self.listening: list[bool] = []

    def set_listening(self, value: bool) -> None:
        self.listening.append(value)


class _AppStub:
    from kiki.application import KikiApplication

    _follow_up_allowed = KikiApplication._follow_up_allowed
    _try_arm_follow_up = KikiApplication._try_arm_follow_up

    def __init__(self, *, follow_up: bool = True) -> None:
        self._settings = Settings()
        self._settings.voice.wake.enabled = True
        self._settings.voice.wake.follow_up = follow_up
        self._assistant_pause = SimpleNamespace(paused=False)
        self._follow_up = FollowUpTurn()
        self._wake = _Wake()
        self._chat = _Chat()
        self._machine = CharacterStateMachine()
        self.resumes = 0
        self.toasts: list[str] = []

    def _resume_wake(self) -> None:
        self.resumes += 1

    def _toast(self, text: str) -> None:
        self.toasts.append(text)


def _ready(stub: _AppStub) -> None:
    stub._follow_up.begin(enabled=True)
    stub._follow_up.mark_terminal()
    stub._follow_up.mark_response_delivered()


def test_tts_idle_arms_one_follow_up_and_keeps_listening_state() -> None:
    from kiki.application import KikiApplication

    stub = _AppStub()
    _ready(stub)
    stub._machine.set(CharacterState.SPEAKING, hold_ms=0)

    KikiApplication._on_tts_idle(stub)

    assert stub._wake.captures == 1
    assert stub.resumes == 1
    assert stub._chat.listening == [True]
    assert stub._machine.state is CharacterState.LISTENING
    assert "direkt weiterreden" in stub.toasts[-1]


def test_unrelated_tts_idle_never_opens_the_microphone() -> None:
    from kiki.application import KikiApplication

    stub = _AppStub()
    stub._machine.set(CharacterState.SPEAKING, hold_ms=0)

    KikiApplication._on_tts_idle(stub)

    assert stub._wake.captures == 0
    assert stub.resumes == 1
    assert stub._machine.state is CharacterState.IDLE


def test_disabling_follow_up_before_answer_finishes_prevents_capture() -> None:
    from kiki.application import KikiApplication

    stub = _AppStub(follow_up=False)
    _ready(stub)

    assert KikiApplication._try_arm_follow_up(stub) is False
    assert stub._wake.captures == 0


def test_intermediate_harness_prompt_does_not_open_follow_up() -> None:
    from kiki.application import KikiApplication

    class Speech:
        active = False

        def __init__(self) -> None:
            self.spoken: list[str] = []

        def say(self, text: str) -> None:
            self.spoken.append(text)

    stub = _AppStub()
    stub._harness = object()
    stub._speech = Speech()
    stub._follow_up.begin(enabled=True)

    KikiApplication._apply_harness_speak(stub, "Bitte bestätige die Aktion.")
    assert stub._wake.captures == 0

    stub._follow_up.mark_terminal()
    KikiApplication._apply_harness_speak(stub, "Die Aktion ist abgeschlossen.")
    assert stub._wake.captures == 1


def test_chat_completion_without_tts_does_not_open_a_silent_follow_up() -> None:
    from kiki.application import KikiApplication

    stub = _AppStub()
    stub._settings.tts.enabled = False
    stub._speech = None
    stub._follow_up.begin(enabled=True)
    stub._machine.set(CharacterState.THINKING, hold_ms=0)

    KikiApplication._on_stream_done(stub, ok=True, text="Hier ist die Antwort.")

    assert stub._wake.captures == 0
    assert stub._machine.state is CharacterState.IDLE
