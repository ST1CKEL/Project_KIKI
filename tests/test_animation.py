from __future__ import annotations

from pathlib import Path

import pytest

from kiki.character.animation_engine import AnimationClip, AnimationEngine, Frame
from kiki.character.state_machine import CharacterState


def _clip(state: CharacterState, n: int, ms: int, loop: bool) -> AnimationClip:
    frames = tuple(Frame(path=Path(f"{state}-{i}.png"), duration_ms=ms) for i in range(n))
    return AnimationClip(state=state, frames=frames, loop=loop)


def test_looping_idle() -> None:
    clips = {
        CharacterState.IDLE: _clip(CharacterState.IDLE, 3, 100, True),
        CharacterState.THINKING: _clip(CharacterState.THINKING, 2, 50, True),
    }
    engine = AnimationEngine(clips)
    engine.play(CharacterState.IDLE)
    assert engine.frame.path.name == "idle-0.png"
    assert engine.advance(100)
    assert engine.frame.path.name == "idle-1.png"
    engine.advance(100)
    engine.advance(100)
    assert engine.frame.path.name == "idle-0.png"


def test_switch_resets_index() -> None:
    clips = {
        CharacterState.IDLE: _clip(CharacterState.IDLE, 2, 40, True),
        CharacterState.THINKING: _clip(CharacterState.THINKING, 2, 40, True),
    }
    engine = AnimationEngine(clips)
    engine.advance(40)
    engine.play(CharacterState.THINKING)
    assert engine.frame.path.name == "thinking-0.png"


def test_unknown_state_plays_idle() -> None:
    clips = {CharacterState.IDLE: _clip(CharacterState.IDLE, 1, 100, True)}
    engine = AnimationEngine(clips)
    engine.play("not-real")
    assert engine.state is CharacterState.IDLE
    assert engine.frame.path.name == "idle-0.png"


def test_large_elapsed_time_skips_complete_cycles() -> None:
    clips = {CharacterState.IDLE: _clip(CharacterState.IDLE, 3, 100, True)}
    engine = AnimationEngine(clips)

    assert engine.advance(10_150)
    assert engine.frame.path.name == "idle-2.png"


def test_non_looping_clip_stops_on_last_frame_after_large_delta() -> None:
    clips = {
        CharacterState.IDLE: _clip(CharacterState.IDLE, 1, 100, True),
        CharacterState.GREET: _clip(CharacterState.GREET, 3, 75, False),
    }
    engine = AnimationEngine(clips)
    engine.play(CharacterState.GREET)

    assert engine.advance(10_000)
    assert engine.frame.path.name == "greet-2.png"
    assert not engine.advance(10_000)


def test_frame_duration_must_be_positive() -> None:
    with pytest.raises(ValueError, match="duration"):
        Frame(path=Path("bad.png"), duration_ms=0)
