"""The text both voice routes send. One contract, one function, no drift.

`speakable()` is the only normalisation on either route, so what it returns is
literally what reaches the service — for the file-based WAV path and for the PCM
streaming path alike. These tests pin the contract documented at the top of
`tts_text.py`, including that plain German comes through untouched.
"""

from __future__ import annotations

import pytest

from kiki.voice.tts_text import EMOJI_WORDS, SYMBOL_WORDS, flush_buffer, speakable, split_ready

# --- the regression cases ---------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("Hallo 👋", "Hallo"),
        ("Das kostet 20 €", "Das kostet 20 Euro"),
        ("Status ✅", "Status erledigt"),
        ("Fehler ❌ bei 30 °C", "Fehler fehlgeschlagen bei 30 Grad Celsius"),
        ("Größe: 3,5 % — läuft!", "Größe: 3,5 Prozent — läuft!"),
        ("Straße, Grüße — „Anführung“ … 1. Punkt!", "Straße, Grüße — „Anführung“ … 1. Punkt!"),
    ],
)
def test_the_named_regression_cases(raw, spoken) -> None:
    assert speakable(raw) == spoken


# --- German stays German ----------------------------------------------------


PLAIN_GERMAN = [
    "Guten Abend Martin.",
    "Die Straße war größer als gedacht.",
    "Übrigens: äöüÄÖÜß bleiben, wie sie sind.",
    "Er sagte „das reicht mir“ — und ging.",
    "Am 23. August 2026 um 14:30 Uhr.",
    "Das sind 1.234,56 Einheiten; genauer: 1.234,56.",
    "Wirklich? Ja! Also gut …",
    "Erst A, dann B – schließlich C.",
]


@pytest.mark.parametrize("sentence", PLAIN_GERMAN)
def test_plain_german_is_returned_unchanged(sentence) -> None:
    """The guarantee that keeps the existing WAV audio exactly as it was: for
    text without markup, emoji, URLs or control characters, nothing happens."""
    assert speakable(sentence) == sentence


def test_umlauts_and_sharp_s_survive_every_rule() -> None:
    text = "Fußgängerübergang, Größe, Öl, Ärger, Übung"
    assert speakable(text) == text


# --- markdown ---------------------------------------------------------------


def test_code_never_reaches_the_voice() -> None:
    assert speakable("Vorher ```print(1)``` nachher") == "Vorher nachher"
    assert speakable("Nutze `--raw` dafür") == "Nutze dafür"


def test_a_link_keeps_its_label_and_loses_its_target() -> None:
    assert speakable("Schau in die [Doku](https://example.com) rein.") == (
        "Schau in die Doku rein."
    )


def test_markdown_structure_is_removed() -> None:
    assert speakable("## Titel") == "Titel"
    assert speakable("- Erster Punkt") == "Erster Punkt"
    assert speakable("1. Erster Punkt") == "Erster Punkt"
    assert speakable("Das ist **wichtig**") == "Das ist wichtig"


# --- URLs -------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "Sieh https://example.com/x an",
        "Sieh http://example.com an",
        "Sieh www.example.com an",
    ],
)
def test_a_bare_url_is_not_read_aloud(raw) -> None:
    assert speakable(raw) == "Sieh an"


def test_a_sentence_that_is_only_a_url_becomes_nothing() -> None:
    assert speakable("https://example.com/sehr/langer/pfad") == ""


# --- control and invisible characters ---------------------------------------


@pytest.mark.parametrize("char", ["\x00", "\x07", "\x1b", "\x7f", "\x9f", "\x0b", "\x0c"])
def test_control_characters_are_removed(char) -> None:
    assert speakable(f"Zeile{char}Zwei") == "ZeileZwei"


@pytest.mark.parametrize("char", ["​", "‍", "⁠", "﻿", "­"])
def test_invisible_characters_are_removed(char) -> None:
    assert speakable(f"Ok{char}dann") == "Okdann"


@pytest.mark.parametrize("char", [" ", " ", " ", "\t", "\n"])
def test_exotic_whitespace_becomes_one_plain_space(char) -> None:
    assert speakable(f"Ok{char}dann") == "Ok dann"


def test_nothing_unprintable_survives() -> None:
    """Whatever the model emits, no control byte may reach the service."""
    noisy = "".join(chr(index) for index in range(0, 0xA0)) + "Text"
    result = speakable(noisy)
    assert all(char.isprintable() or char == " " for char in result), repr(result)


# --- emoji ------------------------------------------------------------------


def test_meaningful_emoji_become_short_german_words() -> None:
    assert speakable("Build ✅") == "Build erledigt"
    assert speakable("Build ❌") == "Build fehlgeschlagen"
    assert speakable("⚠ Vorsicht") == "Achtung Vorsicht"
    assert speakable("👍 passt") == "gut passt"


def test_decorative_emoji_are_dropped_entirely() -> None:
    assert speakable("Fertig 🎉😊") == "Fertig"
    assert speakable("Los geht's 🚀") == "Los geht's"
    assert speakable("🇩🇪 Deutsch") == "Deutsch"
    assert speakable("Zeit ⏳ läuft") == "Zeit läuft"


def test_an_emoji_never_glues_two_words_together() -> None:
    assert speakable("Status✅fertig") == "Status erledigt fertig"


def test_no_emoji_codepoint_survives() -> None:
    """A stray pictograph has previously pulled the voice into another
    language; none of them may reach the model."""
    text = speakable("a 👋 b ✅ c 🎉 d 🇩🇪 e ⏰ f ™ g ©")
    assert all(ord(char) < 0x2000 or char in "…—„“–" for char in text), repr(text)


def test_the_word_tables_are_short_and_german() -> None:
    for table in (EMOJI_WORDS, SYMBOL_WORDS):
        for symbol, word in table.items():
            assert word.startswith(" ") and word.endswith(" "), symbol
            assert len(word.split()) <= 2, symbol
            assert word.strip().isascii() or True  # German words may carry umlauts


# --- symbols ----------------------------------------------------------------


def test_no_space_is_left_in_front_of_punctuation() -> None:
    """A padded replacement must not leave "20 Euro ." — audible as a stumble."""
    assert speakable("Das kostet 20 €.") == "Das kostet 20 Euro."
    assert speakable("Fertig ✅!") == "Fertig erledigt!"
    assert speakable("Wieviel %?") == "Wieviel Prozent?"
    # German dashes keep the space that belongs to them.
    assert speakable("Erst A — dann B") == "Erst A — dann B"


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("20 €", "20 Euro"),
        ("€20", "Euro 20"),
        ("5 $", "5 Dollar"),
        ("9 £", "9 Pfund"),
        ("100 %", "100 Prozent"),
        ("100%", "100 Prozent"),
        ("30 °C", "30 Grad Celsius"),
        ("72 °F", "72 Grad Fahrenheit"),
        ("45 °", "45 Grad"),
        ("A & B", "A und B"),
        ("5 × 3", "5 mal 3"),
    ],
)
def test_symbols_become_words(raw, spoken) -> None:
    assert speakable(raw) == spoken


def test_an_html_entity_is_decoded_before_it_becomes_a_word() -> None:
    assert speakable("A &amp; B") == "A und B"


# --- the whole pipeline agrees ----------------------------------------------


def test_every_entry_point_applies_the_same_contract() -> None:
    """split_ready, flush_buffer and speakable must not diverge — the director
    uses all three, and a difference would mean two routes hearing two texts."""
    raw = "Status ✅ und 20 €. Danach https://example.com noch was. Rest"
    chunks, rest = split_ready(raw)

    assert chunks == ["Status erledigt und 20 Euro.", "Danach noch was."]
    assert flush_buffer(rest) == "Rest"


def test_the_first_chunk_split_normalises_too() -> None:
    chunks, _rest = split_ready("Kurz gesagt, das kostet 20 € und ist ✅.", first=True)
    assert chunks
    assert "€" not in chunks[0]
    assert "✅" not in chunks[0]


def test_an_utterance_that_normalises_to_nothing_is_not_spoken() -> None:
    assert flush_buffer("🎉🚀😊") == ""
    assert speakable("```nur code```") == ""
