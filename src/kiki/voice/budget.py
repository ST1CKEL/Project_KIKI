"""Incremental sentence budget for a spoken answer.

The assistant's full answer keeps streaming into chat untouched. This filter
decides, while deltas arrive, what may already be *fed* to speech: whole
sentences (a clause for the very first, so KIKI starts talking sooner), up to
a sentence and character limit. Everything past the budget stays silent —
the chat keeps it as the complete transcript.

Redaction is NOT this module's job: the speech director redacts every chunk
on its way into the queue (`_enqueue_locked`), typed chat and voice alike.
"""

from __future__ import annotations

from kiki.voice.tts_text import ends_with_abbreviation

_SENTENCE_PUNCT = ".!?…"
# The first fed chunk may end at a clause once this many characters exist, so
# the first words leave the gate while the sentence is still being generated.
FIRST_CLAUSE_MIN_CHARS = 12


def _word_before(text: str, pos: int) -> str:
    """The dotted word ending at `pos` (index of the punctuation)."""
    start = pos
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in ".-"):
        start -= 1
    return text[start:pos]


def _is_true_sentence_end(text: str, pos: int) -> bool:
    """Is the punctuation at `pos` a real sentence end for streaming purposes?

    A stop only counts when followed by whitespace: a period as the very last
    character may still grow into an abbreviation or an ordinal, so it waits
    for the next delta (or the final flush).
    """
    if pos + 1 < len(text) and not text[pos + 1].isspace():
        return False
    if ends_with_abbreviation(text[:pos]):
        return False
    word = _word_before(text, pos)
    if word.isdigit():  # ordinals: "den 3." — wait for what follows
        return False
    if len(word) == 1 and word.isupper():  # initials: "Bei A. Müller"
        return False
    return True


class StreamingVoiceBudget:
    """Pass through what fits; keep the rest for the transcript only."""

    def __init__(self, *, max_sentences: int, max_characters: int) -> None:
        self._max_sentences = max(0, max_sentences)
        self._max_characters = max(0, max_characters)
        self._pending = ""
        self._sentences = 0
        self._characters = 0

    @property
    def exhausted(self) -> bool:
        if self._max_sentences and self._sentences >= self._max_sentences:
            return True
        if self._max_characters and self._characters >= self._max_characters:
            return True
        return False

    @property
    def spoken_characters(self) -> int:
        return self._characters

    def push(self, delta: str) -> str:
        """Return the portion of `delta` (plus held-back text) within budget."""
        if not delta:
            return ""
        self._pending += delta
        if self.exhausted:
            return ""
        return self._release(final=False)

    def final_flush(self) -> str:
        """Release the last in-budget sentence when the answer is complete.

        A sentence that ends exactly with the stream has no trailing
        whitespace, so `push` held it back; the budget still has room.
        """
        if self.exhausted:
            return ""
        return self._release(final=True)

    def _release(self, *, final: bool) -> str:
        emitted = ""
        while not self.exhausted:
            piece, end_is_sentence = self._next_piece(final=final)
            if piece is None:
                break
            room = self._max_characters - self._characters if self._max_characters else len(piece)
            if room <= 0:
                break
            if len(piece) > room:
                # Character budget ends mid-sentence: speak what fits, once.
                emitted += piece[:room].rstrip()
                self._characters += room
                self._pending = ""
                break
            emitted += piece
            self._characters += len(piece)
            self._pending = self._pending[len(piece) :]
            if end_is_sentence:
                self._sentences += 1
        return emitted

    def _next_piece(self, *, final: bool) -> tuple[str | None, bool]:
        """(next speakable unit, whether it ends a sentence) from the buffer."""
        for pos, char in enumerate(self._pending):
            if char not in _SENTENCE_PUNCT:
                continue
            at_stream_end = pos == len(self._pending) - 1
            if _is_true_sentence_end(self._pending, pos) or (
                final and at_stream_end and not self._ambiguous_stop(pos)
            ):
                piece = self._pending[: pos + 1]
                return piece, True
        # No sentence end yet: the very first unit may end at a clause so the
        # first words are out early while the sentence finishes generating.
        if self._sentences == 0 and self._max_sentences != 0:
            for pos, char in enumerate(self._pending):
                if char in ",;:" and pos + 1 >= FIRST_CLAUSE_MIN_CHARS:
                    return self._pending[: pos + 1], False
        if final and self._pending.strip():
            return self._pending, False
        return None, False

    def _ambiguous_stop(self, pos: int) -> bool:
        """A trailing stop that would be misread as a sentence end."""
        return not _is_true_sentence_end(self._pending + "\n", pos)

    def drop_pending(self) -> str:
        """Forget held-back text (used when the turn ends without flush)."""
        leftover = self._pending
        self._pending = ""
        return leftover
