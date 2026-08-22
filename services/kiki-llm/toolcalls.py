"""Pull tool calls out of a token stream.

Qwen3 emits them as ``<tool_call>{"name": …, "arguments": {…}}</tool_call>``
inline with ordinary prose. Because the harness owns generation, the stream can
be split as it arrives instead of waiting for the whole answer.

The hard part is not the JSON, it is the boundaries: ``<tool_call>`` regularly
arrives across two chunks as ``"<tool"`` and ``"_call>"``. Text that *might* be
the start of a tag is therefore held back until it is clear either way — a
parser that forwards optimistically leaks half-tags into the chat window.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

OPEN = "<tool_call>"
CLOSE = "</tool_call>"


@dataclass
class ParsedCall:
    id: str
    name: str
    arguments: dict = field(default_factory=dict)
    parse_error: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "parse_error": self.parse_error,
        }


def _longest_partial_suffix(text: str, marker: str) -> int:
    """Length of the trailing run of `text` that could still become `marker`."""
    limit = min(len(text), len(marker) - 1)
    for size in range(limit, 0, -1):
        if marker.startswith(text[-size:]):
            return size
    return 0


def parse_call(raw: str) -> ParsedCall:
    """One tool_call body. Malformed JSON is reported, never guessed."""
    call_id = f"call_{uuid.uuid4().hex[:12]}"
    text = raw.strip()
    if not text:
        return ParsedCall(id=call_id, name="", parse_error="leerer Aufruf")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return ParsedCall(id=call_id, name="", parse_error=f"ungültiges JSON: {exc}")
    if not isinstance(payload, dict):
        return ParsedCall(id=call_id, name="", parse_error="Aufruf ist kein Objekt")
    name = str(payload.get("name") or "").strip()
    if not name:
        return ParsedCall(id=call_id, name="", parse_error="Aufruf ohne Namen")
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return ParsedCall(
                id=call_id, name=name, parse_error=f"ungültige Argumente: {exc}"
            )
    if not isinstance(arguments, dict):
        return ParsedCall(id=call_id, name=name, parse_error="Argumente sind kein Objekt")
    return ParsedCall(id=call_id, name=name, arguments=arguments)


class ToolCallStreamParser:
    """Feed raw chunks, get back safe text and finished tool calls."""

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False
        self._call = ""

    def feed(self, chunk: str) -> tuple[str, list[ParsedCall]]:
        """Return (text safe to show, calls completed by this chunk)."""
        calls: list[ParsedCall] = []
        out: list[str] = []
        self._buffer += chunk or ""

        while self._buffer:
            if self._inside:
                end = self._buffer.find(CLOSE)
                if end < 0:
                    # Keep accumulating; a closing tag may still be split.
                    hold = _longest_partial_suffix(self._buffer, CLOSE)
                    if hold:
                        self._call += self._buffer[:-hold]
                        self._buffer = self._buffer[-hold:]
                    else:
                        self._call += self._buffer
                        self._buffer = ""
                    break
                self._call += self._buffer[:end]
                self._buffer = self._buffer[end + len(CLOSE) :]
                calls.append(parse_call(self._call))
                self._call = ""
                self._inside = False
                continue

            start = self._buffer.find(OPEN)
            if start >= 0:
                out.append(self._buffer[:start])
                self._buffer = self._buffer[start + len(OPEN) :]
                self._inside = True
                continue

            # No tag in sight. Emit everything except a possible partial tag.
            hold = _longest_partial_suffix(self._buffer, OPEN)
            if hold:
                out.append(self._buffer[:-hold])
                self._buffer = self._buffer[-hold:]
            else:
                out.append(self._buffer)
                self._buffer = ""
            break

        return "".join(out), calls

    def finish(self) -> tuple[str, list[ParsedCall]]:
        """Flush what is left when generation ends."""
        if self._inside:
            # The model stopped mid-call; report it rather than inventing one.
            leftover = ParsedCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                name="",
                parse_error="Aufruf wurde nicht abgeschlossen",
            )
            self._call = ""
            self._inside = False
            self._buffer = ""
            return "", [leftover]
        text, self._buffer = self._buffer, ""
        return text, []
