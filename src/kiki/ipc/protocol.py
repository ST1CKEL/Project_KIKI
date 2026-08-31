"""Newline-delimited JSON. One object per line, UTF-8, no binary on the socket."""

from __future__ import annotations

import json
from typing import Any


class ProtocolError(ValueError):
    """A socket line was not a JSON object."""


def dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


def loads(line: bytes | str) -> dict[str, Any]:
    if isinstance(line, bytes):
        text = line.decode("utf-8")
    else:
        text = line
    text = text.strip()
    if not text:
        raise ProtocolError("empty line")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("payload is not an object")
    return payload
