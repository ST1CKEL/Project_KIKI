"""Encode user-picked images for vision models. Never captures the desktop."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image

MAX_EDGE = 1280
JPEG_QUALITY = 85
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


class VisionEncodeError(ValueError):
    """The file is not a usable still image."""


def encode_image_file(path: Path | str, *, max_edge: int = MAX_EDGE) -> str:
    """Return JPEG-base64 suitable for Ollama `images` and OpenAI image_url."""
    src = Path(path)
    if src.suffix.lower() not in ALLOWED_SUFFIXES:
        raise VisionEncodeError(f"unsupported image type: {src.suffix}")
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            longest = max(w, h)
            if longest > max_edge:
                scale = max_edge / longest
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    except OSError as exc:
        raise VisionEncodeError(f"could not read image {src}: {exc}") from exc
    return base64.b64encode(buf.getvalue()).decode("ascii")


def ollama_message_payload(message: object) -> dict:
    """Build an Ollama /api/chat message dict, including optional images."""
    role = getattr(message, "role")
    content = getattr(message, "content")
    images = getattr(message, "images", ()) or ()
    tool_calls = getattr(message, "tool_calls", ()) or ()
    payload: dict = {"role": role, "content": content}
    if images:
        payload["images"] = list(images)
    if tool_calls:
        payload["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments}} for call in tool_calls
        ]
    if role == "tool":
        # Ollama matches results by name, not by id.
        name = getattr(message, "tool_name", None)
        if name:
            payload["tool_name"] = name
    return payload


def openai_message_payload(message: object) -> dict:
    """Build an OpenAI-compatible chat message, with image_url parts when needed."""
    role = getattr(message, "role")
    content = getattr(message, "content")
    images = getattr(message, "images", ()) or ()
    tool_calls = getattr(message, "tool_calls", ()) or ()
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": getattr(message, "tool_call_id", None) or "",
            "content": content,
        }
    if tool_calls:
        return {
            "role": role,
            "content": content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in tool_calls
            ],
        }
    if not images:
        return {"role": role, "content": content}
    parts: list[dict] = [{"type": "text", "text": content}]
    for raw in images:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{raw}"},
            }
        )
    return {"role": role, "content": parts}
