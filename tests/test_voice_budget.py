"""The incremental sentence budget for streaming spoken answers."""

from __future__ import annotations

from kiki.voice.budget import StreamingVoiceBudget


def _stream(budget: StreamingVoiceBudget, text: str, step: int = 5) -> str:
    out = ""
    for i in range(0, len(text), step):
        out += budget.push(text[i : i + step])
    return out


def test_budget_releases_whole_sentences_while_streaming() -> None:
    budget = StreamingVoiceBudget(max_sentences=3, max_characters=450)
    out = _stream(budget, "Erstens. Zweitens. Drittens. Vierter bleibt stumm.")
    assert out == "Erstens. Zweitens. Drittens."
    assert budget.exhausted


def test_first_clause_leaves_early_for_fast_first_words() -> None:
    budget = StreamingVoiceBudget(max_sentences=3, max_characters=450)
    early = budget.push("Das E-Mail-Programm Thunderbird wurde soeben,")
    assert early == "Das E-Mail-Programm Thunderbird wurde soeben,"
    rest = budget.push(" nach deiner Anfrage, beendet.")
    assert rest.endswith("beendet.")


def test_final_flush_releases_a_period_less_tail() -> None:
    budget = StreamingVoiceBudget(max_sentences=3, max_characters=450)
    fed = _stream(budget, "Erstens. Zweitens")  # stream ends without a stop
    assert fed == "Erstens."
    assert budget.final_flush() == " Zweitens"


def test_character_budget_cuts_mid_sentence_once() -> None:
    budget = StreamingVoiceBudget(max_sentences=5, max_characters=40)
    out = _stream(budget, "Dieser eine Satz ist deutlich zu lang für das Budget.")
    assert len(out) <= 40
    assert budget.exhausted


def test_abbreviation_stops_do_not_count_as_sentences() -> None:
    budget = StreamingVoiceBudget(max_sentences=2, max_characters=450)
    out = _stream(budget, "Das sind ca. 5 km. Und z.B. noch ein Satz. Dritter nicht.")
    assert out == "Das sind ca. 5 km. Und z.B. noch ein Satz."


def test_nothing_is_released_after_exhaustion() -> None:
    budget = StreamingVoiceBudget(max_sentences=1, max_characters=450)
    out = _stream(budget, "Nur dieser. Nicht dieser.")
    assert out == "Nur dieser."
    assert budget.push(" Mehr Text.") == ""
    assert budget.final_flush() == ""


def test_zero_limits_mean_unlimited_house_convention() -> None:
    """Zero is the policy's word for "unlimited", not for "mute"."""
    budget = StreamingVoiceBudget(max_sentences=0, max_characters=0)
    assert _stream(budget, "Alles. Und noch eines.") == "Alles. Und noch eines."
    assert not budget.exhausted
