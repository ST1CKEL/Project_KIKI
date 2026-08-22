"""Cut a streaming German answer into chunks that are worth synthesising.

Synthesis on the reference GPU runs at roughly 1.3x realtime, so the silence
before KIKI starts talking is essentially the length of the first chunk. The
first chunk is therefore cut as early as a sensible boundary allows, and later
chunks are kept longer so prosody stays natural.

The rules that matter are the negative ones: a period is not a sentence end when
it sits inside "z.B.", "1.33", "Fedora 44.1", "192.168.0.1", a URL, a path or a
markdown link. Splitting there produces audible nonsense.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from kiki.voice.tts.policy import VoiceResponsePolicy

# Spans that must never be cut, and never spoken piecemeal.
_PROTECTED = re.compile(
    r"```.*?```"                                  # fenced code
    r"|`[^`]*`"                                   # inline code
    r"|\[[^\]]*\]\([^)]*\)"                       # markdown link
    r"|\b(?:https?://|www\.)\S+"                  # url
    r"|(?<![\w.])(?:~|\.{1,2})?/[\w.\-/]{2,}"     # posix path
    r"|\b[A-Za-z]:\\[\w\\.\-]+"                   # windows path
    r"|\b\d{1,3}(?:\.\d{1,3}){3}\b"               # ipv4
    r"|\b\d+(?:[.,]\d+)+\b"                       # decimals and versions
    r"|\b\d+[.,]\b",                              # trailing decimal marker
    re.DOTALL,
)

# German abbreviations whose period is not a sentence end.
_ABBREVIATIONS = (
    "z.b.", "u.a.", "d.h.", "i.d.r.", "s.o.", "s.u.", "u.u.", "z.t.", "v.a.",
    "ca.", "bzw.", "etc.", "evtl.", "ggf.", "inkl.", "exkl.", "usw.", "vgl.",
    "nr.", "abb.", "tab.", "bspw.", "mio.", "mrd.", "dr.", "prof.", "st.",
    "sog.", "max.", "min.", "bzgl.", "insb.", "urspr.",
)

_SENTENCE_END = re.compile(r"[.!?…]+(?=\s|$)")
_CLAUSE_END = re.compile(r"[,;:–—](?=\s)")
_SPEAKABLE = re.compile(r"[\wäöüÄÖÜß]")


@dataclass(frozen=True)
class ChunkerConfig:
    first_chunk_target_chars: int = 80
    min_chunk_chars: int = 55
    max_chunk_chars: int = 180
    max_wait_ms: int = 500
    prefetch_chunks: int = 1
    cancel_pending_on_interrupt: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> ChunkerConfig:
        raw = dict(data or {})

        def number(name: str, default: int, low: int, high: int) -> int:
            try:
                value = int(raw.get(name, default))
            except (TypeError, ValueError):
                return default
            return max(low, min(high, value))

        return cls(
            first_chunk_target_chars=number("first_chunk_target_chars", 80, 10, 1000),
            min_chunk_chars=number("min_chunk_chars", 55, 1, 1000),
            max_chunk_chars=number("max_chunk_chars", 180, 20, 4000),
            max_wait_ms=number("max_wait_ms", 500, 0, 10_000),
            prefetch_chunks=number("prefetch_chunks", 1, 0, 8),
            cancel_pending_on_interrupt=bool(raw.get("cancel_pending_on_interrupt", True)),
        )


def is_speakable(text: str) -> bool:
    """A chunk of pure punctuation or whitespace must never be synthesised."""
    return bool(_SPEAKABLE.search(text or ""))


def _protected_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _PROTECTED.finditer(text)]


def _inside(index: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in spans)


def _ends_abbreviation(text: str, end: int) -> bool:
    """True when the period at `end` closes a known abbreviation."""
    tail = text[max(0, end - 12) : end].lower()
    return any(tail.endswith(abbr) for abbr in _ABBREVIATIONS)


def boundaries(text: str, *, clauses: bool = False) -> list[int]:
    """Offsets just past every usable break, in order."""
    spans = _protected_spans(text)
    found: list[int] = []
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        if _inside(match.start(), spans) or _ends_abbreviation(text, end):
            continue
        found.append(end)
    if clauses:
        for match in _CLAUSE_END.finditer(text):
            if not _inside(match.start(), spans):
                found.append(match.end())
    return sorted(set(found))


class StreamingChunker:
    """Feed streamed text, get chunks that are ready to synthesise."""

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        *,
        policy: VoiceResponsePolicy | None = None,
    ) -> None:
        self._config = config or ChunkerConfig()
        self._policy = policy or VoiceResponsePolicy()
        self.reset()

    def reset(self) -> None:
        self._buffer = ""
        self._emitted_first = False
        self._waiting_since: float | None = None

    @property
    def pending(self) -> str:
        return self._buffer

    @property
    def emitted_first(self) -> bool:
        return self._emitted_first

    def feed(self, delta: str, *, now: float | None = None) -> list[str]:
        """Append streamed text and return whatever became speakable."""
        if delta:
            self._buffer += delta
        if self._waiting_since is None and self._buffer:
            self._waiting_since = now if now is not None else 0.0
        return self._drain(now=now, final=False)

    def flush(self) -> list[str]:
        """Everything left when the answer ended."""
        chunks = self._drain(now=None, final=True)
        rest = self._clean(self._buffer)
        self._buffer = ""
        if rest and is_speakable(rest):
            chunks.append(rest)
            self._emitted_first = True
        self._waiting_since = None
        return chunks

    # --- internals ---------------------------------------------------------

    def _clean(self, raw: str) -> str:
        """Strip anything the policy forbids, then tidy for speech."""
        plan = self._policy.plan(raw, mode=None)
        if plan.text:
            return plan.text
        # The policy also caps length; for a chunk we only need its redaction,
        # so fall back to tidying when the cap emptied it.
        cleaned, _removed = self._policy._redact(raw)
        return VoiceResponsePolicy._tidy(cleaned)

    def _target(self) -> int:
        return (
            self._config.first_chunk_target_chars
            if not self._emitted_first
            else self._config.min_chunk_chars
        )

    def _drain(self, *, now: float | None, final: bool) -> list[str]:
        out: list[str] = []
        while True:
            cut = self._next_cut(now=now, final=final)
            if cut is None:
                break
            head, self._buffer = self._buffer[:cut], self._buffer[cut:]
            spoken = self._clean(head)
            if spoken and is_speakable(spoken):
                out.append(spoken)
                self._emitted_first = True
                self._waiting_since = now if now is not None else 0.0
        return out

    def _next_cut(self, *, now: float | None, final: bool) -> int | None:
        buffer = self._buffer
        if not buffer.strip():
            return None
        # Never cut inside an unbalanced code fence: the closing ``` may still
        # be streaming, and half a fence is not speakable.
        if buffer.count("```") % 2 == 1:
            return None

        target = self._target()
        sentence_cuts = boundaries(buffer, clauses=False)
        for cut in sentence_cuts:
            if len(self._clean(buffer[:cut])) >= target:
                return cut

        waited = self._waited(now)
        allow_clause = (not self._emitted_first) or waited or final
        if allow_clause:
            for cut in boundaries(buffer, clauses=True):
                if len(self._clean(buffer[:cut])) >= target:
                    return cut

        # Hard ceiling: an answer without any boundary must still be spoken.
        if len(self._clean(buffer)) >= self._config.max_chunk_chars:
            forced = self._force_cut(buffer)
            if forced is not None:
                return forced
        # A sentence that ended but stayed below target still goes out once the
        # stream is finished or the wait elapsed.
        if (final or waited) and sentence_cuts:
            return sentence_cuts[0]
        return None

    def _waited(self, now: float | None) -> bool:
        if self._config.max_wait_ms <= 0:
            return True
        if now is None or self._waiting_since is None:
            return False
        return (now - self._waiting_since) * 1000.0 >= self._config.max_wait_ms

    def _force_cut(self, buffer: str) -> int | None:
        """Last resort: cut at a word boundary outside protected spans."""
        spans = _protected_spans(buffer)
        limit = min(len(buffer), self._config.max_chunk_chars)
        for index in range(limit, 0, -1):
            if buffer[index - 1].isspace() and not _inside(index - 1, spans):
                return index
        return None
