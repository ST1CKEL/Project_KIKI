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

# --- German pronunciation: numbers, times, abbreviations, units -------------
#
# The model leaves "14:37", "z.B." and "1.234" as written; a TTS voice then
# improvises differently on every generation. These tables pin the spoken
# form. They run inside `speakable()`, which both voice routes share.

# Expand rather than spell: the stop would otherwise become the letter soup
# "zett Punkt Beh" and can cut a sentence in half (see `_ends_abbreviation`).
ABBREVIATION_WORDS: dict[str, str] = {
    "z.b.": "zum Beispiel",
    "zb.": "zum Beispiel",
    "bspw.": "beispielsweise",
    "ca.": "circa",
    "ggf.": "gegebenenfalls",
    "evtl.": "eventuell",
    "etc.": "et cetera",
    "usw.": "und so weiter",
    "inkl.": "inklusive",
    "bzw.": "beziehungsweise",
    "max.": "maximal",
    "min.": "minimal",
    "u.a.": "unter anderem",
    "d.h.": "das heißt",
    "nr.": "Nummer",
    "dr.": "Doktor",
    "prof.": "Professor",
}
# Words whose trailing dot never ends a sentence — the splitter must skip
# these boundaries, or "Das sind ca. 5 km." arrives as two broken chunks.
_ABBREVIATION_STOPS = frozenset(
    {"z", "b", "zb", "bspw", "ca", "ggf", "evtl", "etc", "usw", "inkl", "bzw",
     "max", "min", "u", "a", "d", "h", "nr", "dr", "prof", "ggfs", "inkl"}
)
_MONTHS = (
    "Januar Februar März April Mai Juni Juli August September Oktober "
    "November Dezember"
).split()

_UNIT_WORDS: dict[str, str] = {
    "km": "Kilometer", "kg": "Kilogramm", "g": "Gramm", "t": "Tonnen",
    "m": "Meter", "cm": "Zentimeter", "mm": "Millimeter",
    "l": "Liter", "ml": "Milliliter",
    "s": "Sekunden", "ms": "Millisekunden", "h": "Stunden",
    "MB": "Megabyte", "GB": "Gigabyte", "KB": "Kilobyte", "TB": "Terabyte",
    "GHz": "Gigahertz", "MHz": "Megahertz", "Hz": "Hertz",
    "W": "Watt", "kW": "Kilowatt", "PS": "Pferdestärken",
    "km/h": "Kilometer pro Stunde",
}
_UNIT_RE = re.compile(
    r"(?<![\w])(\d+(?:\.\d+)?(?:,\d+)?)\s?(km/h|kHz|GHz|MHz|MB|GB|KB|TB|Hz|km|kg|cm|mm|ml|ms|kW|PS|W|t|g|m|l|s|h)(?![\w])"
)

_ONES = {
    1: "ein", 2: "zwei", 3: "drei", 4: "vier", 5: "fünf", 6: "sechs",
    7: "sieben", 8: "acht", 9: "neun", 10: "zehn", 11: "elf", 12: "zwölf",
    13: "dreizehn", 14: "vierzehn", 15: "fünfzehn", 16: "sechzehn",
    17: "siebzehn", 18: "achtzehn", 19: "neunzehn",
}
_TENS = {
    2: "zwanzig", 3: "dreißig", 4: "vierzig", 5: "fünfzig", 6: "sechzig",
    7: "siebzig", 8: "achtzig", 9: "neunzig",
}
_ORDINAL_STEMS = {1: "erst", 2: "zweit", 3: "dritt", 7: "siebt", 8: "acht"}

_TIME_RE = re.compile(r"(?<!\d)([0-9]{1,2}):([0-9]{2})(?![0-9])(\s*Uhr)?")
_DECIMAL_RE = re.compile(r"(?<!\d)(\d+),(\d+)(?!\d)")
# German numerals: dot groups thousands, comma starts the fraction. One
# combined pass so "1.234,56" never half-converts.
_NUMERAL_RE = re.compile(r"(?<![\d])(\d{1,3}(?:\.\d{3})+)(?:,(\d+))?(?![\d])")
_ORDINAL_RE = re.compile(
    r"(?<![\w])(\d{1,2})\.\s+(?=(" + "|".join(_MONTHS) + r"|Monat|Platz|Jahrhundert)\b)"
)
_INT_RE = re.compile(r"(?<![\w.,])(\d{1,6})(?!\.?\d)")


def _number_words(n: int) -> str:
    """0..999999 as German cardinal words ("eins" for the standalone one)."""
    if n == 0:
        return "null"
    parts: list[str] = []
    if n >= 1000:
        thousands, n = divmod(n, 1000)
        parts.append((_number_words_compound(thousands) or "ein") + "tausend")
    if n >= 100:
        hundreds, n = divmod(n, 100)
        parts.append((_ONES.get(hundreds, "ein") if hundreds > 1 else "ein") + "hundert")
    if n >= 20:
        ones, decade = n % 10, n // 10
        parts.append(f"{_ONES[ones]}und{_TENS[decade]}" if ones else _TENS[decade])
    elif n > 0:
        parts.append(_ONES[n])
    word = "".join(parts)
    return "eins" if word == "ein" else word


def _number_words_compound(n: int) -> str:
    """Cardinal words for use inside a compound ("eintausend", not "einstausend")."""
    return _number_words(n).replace("eins", "ein", 1) if _number_words(n) == "eins" else _number_words(n)


def _ordinal_words(n: int, *, ending: str) -> str:
    stem = _ORDINAL_STEMS.get(n)
    if stem is not None:
        return stem + ending
    cardinal = _number_words_compound(n)
    # 4→vierte, 14→vierzehnte — but 23→dreiundzwanzig+st+e.
    return cardinal + ("st" if n >= 20 else "") + ending


def ends_with_abbreviation(head: str) -> bool:
    """True when a stop after `head` would belong to an abbreviation word.

    `head` is the text *before* the stop: the check looks at its last word,
    so "Das sind ca" (before the dot) qualifies just like a checked tail.
    """
    tokens = head.rstrip().rsplit(maxsplit=1)
    if not tokens:
        return False
    word = tokens[-1].rstrip(".")
    if word in _ABBREVIATION_STOPS:
        return True
    # "z.B." keeps its inner dots in the token; compare them stripped.
    return word.replace(".", "").lower() in _ABBREVIATION_STOPS


def _spoken_numeral(match: re.Match) -> str:
    whole = int(match.group(1).replace(".", ""))
    words = _number_words(whole)
    if match.group(2):
        words += f" Komma {_number_words(int(match.group(2)))}"
    return words


def _spoken_numbers(text: str) -> str:
    """Times, units, decimals, thousands, ordinals, then plain integers."""
    text = _TIME_RE.sub(
        lambda m: f"{_number_words(int(m.group(1)))} Uhr {_number_words(int(m.group(2)))}",
        text,
    )
    text = _UNIT_RE.sub(
        lambda m: f"{_spoken_number_text(m.group(1))} {_UNIT_WORDS[m.group(2)]}",
        text,
    )
    text = _NUMERAL_RE.sub(_spoken_numeral, text)
    text = _DECIMAL_RE.sub(
        lambda m: f"{_number_words(int(m.group(1)))} Komma {_number_words(int(m.group(2)))}",
        text,
    )

    def _ordinal(match: re.Match) -> str:
        preceding = text[: match.start()].rstrip().lower()
        ending = "e" if preceding.endswith("der") else "en"
        return f"{_ordinal_words(int(match.group(1)), ending=ending)} "

    text = _ORDINAL_RE.sub(_ordinal, text)
    return _INT_RE.sub(lambda m: _number_words(int(m.group(1))), text)


def _spoken_number_text(raw: str) -> str:
    value = raw.replace(".", "")
    if "," in value:
        whole, frac = value.split(",", 1)
        return f"{_number_words(int(whole))} Komma {_number_words(int(frac))}"
    return _number_words(int(value))

MIN_SPEAK_CHARS = 2
# Below this a clause is not worth its own synthesis request; above it, cutting
# early pays off. With Kokoro fast synthesis, starting on the first few words cuts latency.
FIRST_CHUNK_MIN_CHARS = 12


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
    # "€20" is amount-first in German; fix it while the amount is still
    # digits, before the symbol table pads the currency into words.
    cleaned = _CURRENCY_PREFIX_RE.sub(
        lambda m: f"{m.group(2)} {_CURRENCY_NAMES[m.group(1)]} ", cleaned
    )
    for symbol, word in SYMBOL_WORDS.items():
        if symbol in cleaned:
            cleaned = cleaned.replace(symbol, word)
    # Spoken German for what a voice would otherwise improvise: times,
    # numbers, ordinals, units.
    cleaned = _spoken_numbers(cleaned)
    cleaned = _currency_order(cleaned)
    for short, long in ABBREVIATION_WORDS.items():
        stem = re.escape(short[:-1])
        cleaned = re.sub(
            r"(?<![\w.])" + stem + r"\.((?=\s)|$)",
            long + " ",
            cleaned,
            flags=re.IGNORECASE,
        )
    cleaned = _SPACE.sub(" ", cleaned)
    return _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned).strip()


_CURRENCY_NAMES = {"€": "Euro", "$": "Dollar", "£": "Pfund"}
_CURRENCY_PREFIX_RE = re.compile(r"(?<![\w])([€$£])\s*(\d[\d.,]*)")
_CURRENCY_ORDER_RE = re.compile(r"\b(Euro|Dollar|Pfund)\s+(\d[\w]*)", re.IGNORECASE)


def _currency_order(text: str) -> str:
    """„ Euro 20" reads wrong in German — put the amount first."""
    return _CURRENCY_ORDER_RE.sub(lambda m: f"{m.group(2)} {m.group(1)}", text)


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
        # "Das sind ca. 5 km." must not arrive as "Das sind ca." plus "5 km.":
        # a stop behind a known abbreviation is part of the word, not a
        # sentence end. The next real boundary collects the whole sentence.
        if ends_with_abbreviation(buffer[: match.start()]):
            continue
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
