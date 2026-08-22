"""Turn assistant markdown into spoken German sentences."""

from __future__ import annotations

import html
import re

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]*`")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HEADING = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_EMPHASIS = re.compile(r"[*_~]{1,3}")
_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_NUMBERED = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
_SPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"([.!?…]+)(\s+|$)")
# Clause boundaries, used only for the very first chunk of an answer.
_CLAUSE_END = re.compile(r"([,;:–—]+)(\s+)")

MIN_SPEAK_CHARS = 2
# Below this a clause is not worth its own synthesis request; above it, cutting
# early pays off. Synthesis runs at about 1.24x realtime on the reference GPU,
# so the wait before KIKI starts speaking is roughly the first chunk's length —
# halving that chunk halves the silence at the start of every answer.
FIRST_CHUNK_MIN_CHARS = 24


def speakable(text: str) -> str:
    """Strip markdown/code so only words remain for TTS."""
    if not text:
        return ""
    cleaned = _FENCE.sub(" ", text)
    cleaned = _INLINE_CODE.sub(" ", cleaned)
    cleaned = _IMAGE.sub(r"\1", cleaned)
    cleaned = _LINK.sub(r"\1", cleaned)
    cleaned = _HEADING.sub("", cleaned)
    cleaned = _BULLET.sub("", cleaned)
    cleaned = _NUMBERED.sub("", cleaned)
    cleaned = _EMPHASIS.sub("", cleaned)
    cleaned = html.unescape(cleaned)
    return _SPACE.sub(" ", cleaned).strip()


def split_ready(buffer: str, *, first: bool = False) -> tuple[list[str], str]:
    """Return completed speakable sentences and the leftover raw buffer.

    With `first`, a clause boundary also ends a chunk. Only the opening chunk
    of an answer uses this: it cuts the silence before KIKI starts talking,
    while later chunks keep whole sentences so the prosody stays natural.
    """
    if not buffer:
        return [], ""
    if buffer.count("```") % 2 == 1:
        return [], buffer
    if first:
        early = _first_clause(buffer)
        if early is not None:
            head, rest = early
            return [head], rest
    ready: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(buffer):
        piece = buffer[start : match.end()]
        spoken = speakable(piece)
        if len(spoken) >= MIN_SPEAK_CHARS:
            ready.append(spoken)
        start = match.end()
    return ready, buffer[start:]


def _first_clause(buffer: str) -> tuple[str, str] | None:
    """Cut at the earliest clause break that yields a worthwhile chunk."""
    sentence = _SENTENCE_END.search(buffer)
    limit = sentence.end() if sentence else len(buffer)
    for match in _CLAUSE_END.finditer(buffer):
        if match.end() > limit:
            break
        spoken = speakable(buffer[: match.end()])
        if len(spoken) >= FIRST_CHUNK_MIN_CHARS:
            return spoken, buffer[match.end() :]
    return None


def flush_buffer(buffer: str) -> str:
    """Last leftover after the stream ends."""
    spoken = speakable(buffer)
    return spoken if len(spoken) >= MIN_SPEAK_CHARS else ""
