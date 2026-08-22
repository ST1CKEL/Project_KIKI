"""Frame-sequence player. Future renderers (Lottie, Spine, Live2D) implement the same clip API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kiki.character.state_machine import CharacterState, resolve_state


@dataclass(frozen=True)
class Frame:
    path: Path
    duration_ms: int

    def __post_init__(self) -> None:
        if self.duration_ms <= 0:
            raise ValueError("frame duration must be positive")


@dataclass(frozen=True)
class AnimationClip:
    state: CharacterState
    frames: tuple[Frame, ...]
    loop: bool

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError(f"clip {self.state} has no frames")


class AnimationEngine:
    """Time-based clip player. The GTK layer supplies real frame-clock deltas."""

    def __init__(self, clips: dict[CharacterState, AnimationClip]) -> None:
        if CharacterState.IDLE not in clips:
            raise ValueError("character pack must provide an idle clip")
        self._clips = clips
        self._state = CharacterState.IDLE
        self._index = 0
        self._elapsed = 0
        self._clip = clips[CharacterState.IDLE]

    @property
    def state(self) -> CharacterState:
        return self._state

    @property
    def frame(self) -> Frame:
        return self._clip.frames[self._index]

    def available(self) -> frozenset[CharacterState]:
        return frozenset(self._clips)

    def play(self, state: CharacterState | str) -> Frame:
        target = resolve_state(state)
        clip = self._clips.get(target) or self._clips[CharacterState.IDLE]
        if target != self._state or clip is not self._clip:
            self._state = target
            self._clip = clip
            self._index = 0
            self._elapsed = 0
        return self.frame

    def advance(self, dt_ms: int) -> bool:
        """Advance time. Returns True if the visible frame changed."""
        if dt_ms <= 0 or not self._clip.frames:
            return False
        start_index = self._index
        self._elapsed += dt_ms

        # A frame-clock delta may be large after the window was hidden or the
        # desktop resumed from suspend. Skip complete loops in O(1), then walk
        # at most one cycle to find the frame that is actually visible now.
        if self._clip.loop:
            cycle_ms = sum(frame.duration_ms for frame in self._clip.frames)
            if self._elapsed >= cycle_ms:
                self._elapsed %= cycle_ms

        while self._elapsed >= self.frame.duration_ms:
            self._elapsed -= self.frame.duration_ms
            nxt = self._index + 1
            if nxt >= len(self._clip.frames):
                if self._clip.loop:
                    nxt = 0
                else:
                    nxt = len(self._clip.frames) - 1
                    self._elapsed = 0
                    break
            self._index = nxt
        return self._index != start_index
