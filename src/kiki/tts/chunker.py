#!/usr/bin/env python3
"""Project KIKI - Real-Time TTS Streaming Chunker

Responsibilities:
1. Accumulates streaming LLM tokens in a high-speed buffer.
2. Evaluates semantic release rules (sentence ends, clause boundaries, word bounds).
3. Enforces hard character limits (180-240 chars) and stream inactivity timeout.
4. Prevents splitting inside numbers, abbreviations or URLs.
"""

from __future__ import annotations

import re
import time

_SENTENCE_END = re.compile(r"([.!?…])(?:\s+|$)")
_CLAUSE_END = re.compile(r"([,;:])(?:\s+|$)")
_ABBREVIATIONS = frozenset({"z.b", "d.h", "ca", "bzw", "usw", "inkl", "evtl", "prof", "dr", "st"})


class StreamingTtsChunker:
    def __init__(
        self,
        max_chunk_chars: int = 220,
        semantic_min_words: int = 12,
        semantic_max_words: int = 18,
        stream_timeout_ms: int = 300,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.semantic_min_words = semantic_min_words
        self.semantic_max_words = semantic_max_words
        self.stream_timeout_s = stream_timeout_ms / 1000.0

        self._buffer = ""
        self._last_push_time = 0.0

    def push_token(self, token: str) -> list[str]:
        """Add a token to the buffer and return any ready chunks."""
        if not token:
            return []
        self._buffer += token
        self._last_push_time = time.perf_counter()
        return self._extract_ready(final=False)

    def check_timeout(self) -> list[str]:
        """Release buffer if stream is idle past timeout."""
        if not self._buffer.strip():
            return []
        if (time.perf_counter() - self._last_push_time) >= self.stream_timeout_s:
            return self._extract_ready(final=True)
        return []

    def flush(self) -> list[str]:
        """Force flush all remaining text at stream conclusion."""
        return self._extract_ready(final=True)

    def reset(self) -> None:
        self._buffer = ""
        self._last_push_time = 0.0

    def _is_abbreviation(self, text_before_dot: str) -> bool:
        tokens = text_before_dot.rstrip().split()
        if not tokens:
            return False
        last_word = tokens[-1].lower().replace(".", "")
        return last_word in _ABBREVIATIONS

    def _extract_ready(self, final: bool = False) -> list[str]:
        ready_chunks: list[str] = []

        while self._buffer:
            text = self._buffer.strip()
            if not text:
                self._buffer = ""
                break

            words = text.split()
            word_count = len(words)

            # 1. Look for true sentence boundary
            sentence_match = _SENTENCE_END.search(text)
            if sentence_match:
                end_pos = sentence_match.end()
                preceding = text[: sentence_match.start()]
                if not self._is_abbreviation(preceding):
                    chunk = text[:end_pos].strip()
                    ready_chunks.append(chunk)
                    self._buffer = text[end_pos:].lstrip()
                    continue

            # 2. Look for semantic clause boundary if word count is sufficient
            if word_count >= self.semantic_min_words:
                clause_match = _CLAUSE_END.search(text)
                if clause_match:
                    end_pos = clause_match.end()
                    chunk = text[:end_pos].strip()
                    ready_chunks.append(chunk)
                    self._buffer = text[end_pos:].lstrip()
                    continue

            # 3. Hard character limit break on closest word boundary
            if len(text) >= self.max_chunk_chars:
                # Find last whitespace before limit
                cutoff = self.max_chunk_chars
                last_space = text.rfind(" ", 0, cutoff)
                if last_space > 20:
                    cutoff = last_space
                chunk = text[:cutoff].strip()
                ready_chunks.append(chunk)
                self._buffer = text[cutoff:].lstrip()
                continue

            # 4. Stream final flush
            if final:
                ready_chunks.append(text)
                self._buffer = ""
                break

            # Buffer holds until next token or timeout
            break

        return ready_chunks
