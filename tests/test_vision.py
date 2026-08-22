from __future__ import annotations

from pathlib import Path

from PIL import Image

from kiki.ai.prompts import build_messages
from kiki.ai.provider import ChatMessage
from kiki.ai.vision import (
    VisionEncodeError,
    encode_image_file,
    ollama_message_payload,
    openai_message_payload,
)


def test_encode_jpeg_base64(tmp_path: Path) -> None:
    src = tmp_path / "tiny.png"
    Image.new("RGB", (64, 48), (10, 80, 200)).save(src)
    raw = encode_image_file(src)
    assert len(raw) > 40
    assert "\n" not in raw
    assert not raw.startswith("data:")


def test_encode_rejects_non_image(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text("hello", encoding="utf-8")
    try:
        encode_image_file(src)
        raise AssertionError("expected VisionEncodeError")
    except VisionEncodeError:
        pass


def test_ollama_payload_includes_images() -> None:
    msg = ChatMessage(role="user", content="Was ist das?", images=("abc",))
    payload = ollama_message_payload(msg)
    assert payload["images"] == ["abc"]
    assert "data:" not in payload["images"][0]


def test_openai_payload_uses_data_url() -> None:
    msg = ChatMessage(role="user", content="x", images=("abc",))
    payload = openai_message_payload(msg)
    assert payload["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_history_does_not_keep_images() -> None:
    messages = build_messages(
        system_prompt="sys",
        history=[ChatMessage("user", "alt", images=("old",))],
        user_text="neu",
        history_limit=10,
        images=("fresh",),
    )
    assert messages[-1].images == ("fresh",)
    assert messages[1].images == ()
