from __future__ import annotations

from kiki.character.state_machine import (
    MVP_STATES,
    CharacterState,
    CharacterStateMachine,
    resolve_state,
)


def test_starts_idle() -> None:
    machine = CharacterStateMachine(clock=lambda: 0.0)
    assert machine.state is CharacterState.IDLE


def test_mvp_states_exist() -> None:
    assert MVP_STATES <= frozenset(CharacterState)


def test_unknown_name_resolves_to_idle() -> None:
    assert resolve_state("not-a-state") is CharacterState.IDLE
    machine = CharacterStateMachine()
    machine.set("totally-unknown")
    assert machine.state is CharacterState.IDLE


def test_transitory_returns_to_idle() -> None:
    now = {"t": 0.0}
    machine = CharacterStateMachine(clock=lambda: now["t"])
    machine.set(CharacterState.GREET, hold_ms=500)
    assert machine.state is CharacterState.GREET
    now["t"] = 0.4
    machine.tick()
    assert machine.state is CharacterState.GREET
    now["t"] = 0.6
    machine.tick()
    assert machine.state is CharacterState.IDLE


def test_transitory_returns_to_previous_base_state() -> None:
    now = {"t": 0.0}
    machine = CharacterStateMachine(clock=lambda: now["t"])
    machine.set(CharacterState.SPEAKING, hold_ms=0)
    machine.set(CharacterState.HAPPY, hold_ms=500)
    assert machine.state is CharacterState.HAPPY
    assert machine.base_state is CharacterState.SPEAKING

    now["t"] = 0.6
    machine.tick()
    assert machine.state is CharacterState.SPEAKING


def test_nested_reactions_keep_original_base_state() -> None:
    now = {"t": 0.0}
    machine = CharacterStateMachine(clock=lambda: now["t"])
    machine.set(CharacterState.LISTENING, hold_ms=0)
    machine.set(CharacterState.SURPRISED, hold_ms=500)
    machine.set(CharacterState.HAPPY, hold_ms=500)
    assert machine.base_state is CharacterState.LISTENING

    now["t"] = 0.6
    machine.tick()
    assert machine.state is CharacterState.LISTENING


def test_thinking_to_speaking_is_allowed() -> None:
    machine = CharacterStateMachine(clock=lambda: 0.0)
    machine.set(CharacterState.THINKING, hold_ms=0)
    assert machine.state is CharacterState.THINKING
    machine.set(CharacterState.SPEAKING, hold_ms=0)
    assert machine.state is CharacterState.SPEAKING
    machine.set(CharacterState.IDLE, hold_ms=0)
    assert machine.state is CharacterState.IDLE


def test_pause_is_own_state_and_blocks_chat() -> None:
    machine = CharacterStateMachine()
    machine.pause()
    assert machine.state is CharacterState.PAUSED
    assert machine.paused is True
    machine.set(CharacterState.THINKING)
    assert machine.state is CharacterState.PAUSED
    machine.set(CharacterState.SPEAKING)
    assert machine.state is CharacterState.PAUSED
    machine.resume()
    assert machine.state is CharacterState.IDLE


def test_cannot_set_paused_except_via_pause() -> None:
    machine = CharacterStateMachine()
    machine.set(CharacterState.THINKING, hold_ms=0)
    machine.set(CharacterState.PAUSED)
    assert machine.state is CharacterState.THINKING


def test_listeners_fire_on_change() -> None:
    seen: list[CharacterState] = []
    machine = CharacterStateMachine()
    machine.subscribe(seen.append)
    machine.set(CharacterState.HAPPY, hold_ms=0)
    machine.set(CharacterState.HAPPY, hold_ms=0)
    assert seen == [CharacterState.HAPPY]
