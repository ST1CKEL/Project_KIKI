"""Turn assistant markdown into spoken German sentences.

The normalisation contract
--------------------------
`speakable()` is the single place both voice routes pass through — the
file-based WAV route and the PCM streaming route call it via `split_ready`,
`flush_buffer` and `SpeechDirector.say`. Whatever it returns is what reaches the
service, so the two routes cannot drift apart by construction.

What it guarantees, in this order:

1. **Markdown** — fences and inline code drop out entirely; images and links
   keep their label; headings, bullets, numbering and emphasis markers go.
2. **URLs** — a bare `https://…` or `www.…` is removed. Read aloud it is noise,
   and a markdown link has already become its label by this point.
3. **Control characters** — C0 and C1 are removed. Tab and newline survive as
   whitespace and fold into single spaces at the end.
4. **Invisible characters** — zero-width spaces, joiners, soft hyphens and the
   byte-order mark are removed; they are layout, not speech.
5. **Emoji** — a short, deliberate table becomes German words where the symbol
   carries meaning ("Status ✅" → "Status erledigt"); everything else in the
   emoji ranges is dropped. Decorative emoji are not read out, and none of them
   reaches the model, where a stray pictograph has previously pulled the voice
   into the wrong language.
6. **Symbols** — a small table becomes words: € Euro, $ Dollar, £ Pfund,
   % Prozent, ° Grad, & und, × mal.

Finally whitespace folds to single spaces and is tightened before `, . ; : ! ?`
so a replaced symbol cannot leave "20 Euro ." behind. Dashes and the ellipsis are
left alone — German writes "Also gut …" with the space.

Everything else, German punctuation and umlauts included, is passed through
untouched: a plain German sentence comes out of this function exactly as it went
in, so the existing WAV audio does not change.
"""

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
# The word tables pad both sides, which would otherwise leave "20 Euro ." — an
# audible stumble, and a space the sentence splitter has to reason about. Only
# the marks German writes tight: dashes and the ellipsis keep their space, so
# "Also gut …" comes through exactly as written.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
# Same shape the voice policy uses, so both agree on what counts as a URL.
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
# C0 and C1 minus tab and newline, which the whitespace fold handles.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Layout, not speech: zero-width marks, joiners, soft hyphen, BOM.
_INVISIBLE = re.compile("[\u200b-\u200f\u2060\ufeff\u00ad]")
# Emoji and pictographs. Applied *after* the word table below, so the few that
# carry meaning have already become words.
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"      # emoticons, pictographs, transport, symbols
    "\u2190-\u21ff"              # arrows
    "\u2300-\u23ff"              # misc technical: ⌚ ⏰ ⏳
    "\u2460-\u24ff"              # enclosed alphanumerics
    "\u25a0-\u27bf"              # geometric shapes, misc symbols, dingbats
    "\u2b00-\u2bff"
    "\ufe00-\ufe0f"              # variation selectors
    "\u00a9\u00ae\u2122"        # © ® ™
    "]+"
)

# Deliberately short German words, and only where the symbol is information
# rather than decoration. Spaces on both sides: "Status✅fertig" must not become
# one word, and neither must "€20".
EMOJI_WORDS: dict[str, str] = {
    "✅": " erledigt ",
    "☑": " erledigt ",
    "✔": " erledigt ",
    "❌": " fehlgeschlagen ",
    "✖": " fehlgeschlagen ",
    "⚠": " Achtung ",
    "❗": " Achtung ",
    "❓": " Frage ",
    "👍": " gut ",
    "👎": " schlecht ",
}

# Order matters: the compound units come before the bare degree sign, so
# "30 °C" says "30 Grad Celsius" rather than "30 Grad C".
SYMBOL_WORDS: dict[str, str] = {
    "°C": " Grad Celsius ",
    "°F": " Grad Fahrenheit ",
    "€": " Euro ",
    "$": " Dollar ",
    "£": " Pfund ",
    "%": " Prozent ",
    "°": " Grad ",
    "&": " und ",
    "×": " mal ",
}
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
    """Normalise assistant text to what the TTS service should say.

    The contract is documented at the top of this module. Both routes call this
    and nothing else, so both send the identical string.
    """
    if not text:
        return ""
    cleaned = _FENCE.sub(" ", text)
    cleaned = _INLINE_CODE.sub(" ", cleaned)
    cleaned = _IMAGE.sub(r"\1", cleaned)
    # Links become their label first, so the URL rule below cannot eat the words.
    cleaned = _LINK.sub(r"\1", cleaned)
    cleaned = _URL.sub(" ", cleaned)
    cleaned = _HEADING.sub("", cleaned)
    cleaned = _BULLET.sub("", cleaned)
    cleaned = _NUMBERED.sub("", cleaned)
    cleaned = _EMPHASIS.sub("", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = _CONTROL.sub("", cleaned)
    cleaned = _INVISIBLE.sub("", cleaned)
    for symbol, word in EMOJI_WORDS.items():
        if symbol in cleaned:
            cleaned = cleaned.replace(symbol, word)
    cleaned = _EMOJI.sub(" ", cleaned)
    for symbol, word in SYMBOL_WORDS.items():
        if symbol in cleaned:
            cleaned = cleaned.replace(symbol, word)
    cleaned = _SPACE.sub(" ", cleaned)
    return _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned).strip()


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
