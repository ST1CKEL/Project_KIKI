from __future__ import annotations

import tomllib
from pathlib import Path

from PIL import Image

from kiki.character.assets import load_character_pack, write_placeholder_pack
from kiki.character.state_machine import CharacterState

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_pack(tmp_path: Path) -> None:
    idle = tmp_path / "idle"
    idle.mkdir()
    Image.new("RGBA", (40, 80), (10, 20, 200, 255)).save(idle / "00.png")
    (tmp_path / "manifest.toml").write_text(
        """
id = "test"
name = "Test"
renderer = "frames"
scale_anchor_px = 80

[states.idle]
loop = true
frames = ["idle/00.png"]
frame_ms = [400]
""",
        encoding="utf-8",
    )
    pack = load_character_pack("test", root=tmp_path)
    assert pack.renderer == "frames"
    assert CharacterState.IDLE in pack.clips
    assert CharacterState.THINKING in pack.clips  # fallback
    engine = pack.engine()
    engine.play(CharacterState.THINKING)
    assert engine.frame.path.name == "00.png"


def test_unknown_manifest_state_is_ignored(tmp_path: Path) -> None:
    idle = tmp_path / "idle"
    idle.mkdir()
    Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(idle / "00.png")
    (tmp_path / "manifest.toml").write_text(
        """
id = "test"
renderer = "frames"

[states.idle]
frames = ["idle/00.png"]
frame_ms = [200]

[states.not_a_real_state]
frames = ["idle/00.png"]
""",
        encoding="utf-8",
    )
    pack = load_character_pack("test", root=tmp_path)
    assert CharacterState.IDLE in pack.clips
    assert CharacterState.PAUSED in pack.clips


def test_missing_idle_raises(tmp_path: Path) -> None:
    (tmp_path / "manifest.toml").write_text(
        """
id = "empty"
renderer = "frames"
""",
        encoding="utf-8",
    )
    import pytest

    from kiki.character.assets import CharacterPackError

    with pytest.raises(CharacterPackError):
        load_character_pack("empty", root=tmp_path)


def test_placeholder_pack_has_mvp_clips(tmp_path: Path) -> None:
    pack = write_placeholder_pack(tmp_path / "ph")
    assert CharacterState.IDLE in pack.clips
    assert CharacterState.THINKING in pack.clips
    assert CharacterState.SPEAKING in pack.clips
    engine = pack.engine()
    engine.play(CharacterState.ERROR)
    assert engine.frame.path.is_file()


def test_shipped_adult_pack_has_complete_production_frames() -> None:
    root = PROJECT_ROOT / "data/character/kiki-adult-v3"
    manifest = tomllib.loads((root / "manifest.toml").read_text(encoding="utf-8"))
    pack = load_character_pack("kiki-adult-v3", root=root)

    assert pack.id == "kiki-adult-v3"
    assert pack.name == "KIKI Adult v3"
    assert set(manifest["states"]) == {state.value for state in CharacterState}
    assert set(pack.clips) == set(CharacterState)

    relative_frames = {
        relative
        for state in manifest["states"].values()
        for relative in state["frames"]
    }
    assert len(relative_frames) == 13
    for relative in sorted(relative_frames):
        path = root / relative
        assert path.is_file(), relative
        with Image.open(path) as image:
            assert image.mode == "RGBA", relative
            assert image.size == (512, 512), relative
            assert image.getbbox() is not None, relative
            assert image.getpixel((0, 0))[3] == 0, relative
            assert image.getpixel((511, 511))[3] == 0, relative
            alpha = image.getchannel("A")
            assert alpha.getextrema() == (0, 255), relative
