"""Load a character pack from data/character/<id>/manifest.toml.

Renderer is "frames" today. The manifest `renderer` field is the extension
point for lottie / spine / live2d / godot later — the rest of the app talks
to CharacterPack + AnimationEngine only.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

from kiki.character.animation_engine import AnimationClip, AnimationEngine, Frame
from kiki.character.state_machine import CharacterState
from kiki.paths import cache_dir, character_dir

log = logging.getLogger(__name__)


class CharacterPackError(Exception):
    """The character pack is missing or invalid."""


@dataclass(frozen=True)
class CharacterPack:
    id: str
    name: str
    renderer: str
    root: Path
    scale_anchor_px: int
    clips: dict[CharacterState, AnimationClip]
    aspect: float  # width / height of the first idle frame, used for window sizing

    def engine(self) -> AnimationEngine:
        return AnimationEngine(self.clips)


def load_character_pack(character_id: str = "kiki-adult-v3", root: Path | None = None) -> CharacterPack:
    base = root or character_dir(character_id)
    manifest_path = base / "manifest.toml"
    if not manifest_path.is_file():
        raise CharacterPackError(f"missing character manifest: {manifest_path}")
    with manifest_path.open("rb") as fh:
        data = tomllib.load(fh)
    renderer = str(data.get("renderer", "frames"))
    if renderer != "frames":
        log.warning("character renderer %s is not implemented; using frames fallback", renderer)
    states = data.get("states") or {}
    clips: dict[CharacterState, AnimationClip] = {}
    for raw_name, spec in states.items():
        try:
            state = CharacterState(raw_name)
        except ValueError:
            log.warning("ignoring unknown character state %s", raw_name)
            continue
        frames_spec = spec.get("frames") or []
        durations = spec.get("frame_ms") or []
        default_ms = int(spec.get("frame_ms_default", 400))
        frames: list[Frame] = []
        for idx, rel in enumerate(frames_spec):
            path = base / str(rel)
            if not path.is_file():
                log.warning("missing frame %s for %s", path, state)
                continue
            ms = int(durations[idx]) if idx < len(durations) else default_ms
            frames.append(Frame(path=path, duration_ms=max(50, ms)))
        if not frames:
            continue
        clips[state] = AnimationClip(
            state=state,
            frames=tuple(frames),
            loop=bool(spec.get("loop", state in {CharacterState.IDLE, CharacterState.THINKING, CharacterState.SPEAKING, CharacterState.LISTENING, CharacterState.SLEEPING})),
        )
    if CharacterState.IDLE not in clips:
        raise CharacterPackError(f"{base} has no idle frames")
    # Missing clips must not crash the UI. paused reuses sleeping when present.
    for state in CharacterState:
        if state in clips:
            continue
        if state is CharacterState.PAUSED and CharacterState.SLEEPING in clips:
            sleeping = clips[CharacterState.SLEEPING]
            clips[state] = AnimationClip(state=state, frames=sleeping.frames, loop=True)
            continue
        idle = clips[CharacterState.IDLE]
        clips[state] = AnimationClip(state=state, frames=idle.frames[:1], loop=False)
    from PIL import Image

    with Image.open(clips[CharacterState.IDLE].frames[0].path) as probe:
        width, height = probe.size
    aspect = width / max(height, 1)
    return CharacterPack(
        id=str(data.get("id", character_id)),
        name=str(data.get("name", character_id)),
        renderer=renderer,
        root=base,
        scale_anchor_px=int(data.get("scale_anchor_px", 260)),
        clips=clips,
        aspect=aspect,
    )


def write_placeholder_pack(root: Path) -> CharacterPack:
    """Minimal original silhouette so the pet window still opens without art."""
    root.mkdir(parents=True, exist_ok=True)
    idle_dir = root / "idle"
    think_dir = root / "thinking"
    speak_dir = root / "speaking"
    idle_dir.mkdir(exist_ok=True)
    think_dir.mkdir(exist_ok=True)
    speak_dir.mkdir(exist_ok=True)
    _draw_placeholder(idle_dir / "00.png", body=(46, 98, 168, 255))
    _draw_placeholder(think_dir / "00.png", body=(72, 72, 150, 255))
    _draw_placeholder(speak_dir / "00.png", body=(58, 120, 176, 255))
    (root / "manifest.toml").write_text(
        """
id = "placeholder"
name = "KIKI Placeholder"
renderer = "frames"
scale_anchor_px = 260

[states.idle]
loop = true
frames = ["idle/00.png"]
frame_ms = [800]

[states.thinking]
loop = true
frames = ["thinking/00.png"]
frame_ms = [400]

[states.speaking]
loop = true
frames = ["speaking/00.png"]
frame_ms = [280]
""".lstrip(),
        encoding="utf-8",
    )
    log.warning("using placeholder character pack at %s", root)
    return load_character_pack("placeholder", root=root)


def ensure_character_pack(character_id: str = "kiki-adult-v3") -> CharacterPack:
    try:
        return load_character_pack(character_id)
    except CharacterPackError as exc:
        log.warning("character pack unavailable (%s); drawing placeholder", exc)
        return write_placeholder_pack(cache_dir() / "placeholder-character")


def _draw_placeholder(path: Path, *, body: tuple[int, int, int, int]) -> None:
    from PIL import Image, ImageDraw

    im = Image.new("RGBA", (200, 360), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.ellipse((70, 18, 130, 82), fill=(88, 62, 128, 255))
    draw.rounded_rectangle((58, 78, 142, 210), radius=22, fill=body)
    draw.rectangle((72, 208, 94, 330), fill=(36, 42, 58, 255))
    draw.rectangle((106, 208, 128, 330), fill=(36, 42, 58, 255))
    draw.ellipse((68, 322, 98, 348), fill=(180, 190, 205, 255))
    draw.ellipse((102, 322, 132, 348), fill=(180, 190, 205, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "PNG")
