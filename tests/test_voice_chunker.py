"""German streaming chunker: cut early, but never in the wrong place."""

from __future__ import annotations

import pytest

from kiki.config.settings import default_mapping
from kiki.voice.tts import ChunkerConfig, StreamingChunker, boundaries, is_speakable


def _chunker(**overrides) -> StreamingChunker:
    base = {"first_chunk_target_chars": 30, "min_chunk_chars": 30, "max_wait_ms": 0}
    base.update(overrides)
    return StreamingChunker(ChunkerConfig(**base))


def _stream(text: str, *, size: int = 7, **overrides) -> list[str]:
    """Feed text in fixed-size pieces, as a token stream would arrive."""
    chunker = _chunker(**overrides)
    out: list[str] = []
    for index in range(0, len(text), size):
        out.extend(chunker.feed(text[index : index + size], now=0.0))
    out.extend(chunker.flush())
    return out


# --- boundaries that must not be used ---------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Nutze z.B. den Editor",
        "Das kostet ca. 5 Euro",
        "Der Wert liegt bei 1.33 Sekunden",
        "Wir nutzen Fedora 44.1 hier",
        "Der Server hat 192.168.0.1 als Adresse",
        "Siehe https://example.com/pfad hier",
        "Die Datei /etc/hosts liegt dort",
        "Lies die [Doku](https://x.io/y) durch",
        "Ein Wert von 3,5 Prozent",
    ],
)
def test_no_sentence_boundary_inside_protected_text(text) -> None:
    """A period inside an abbreviation, number, URL or path is not an end."""
    assert boundaries(text, clauses=False) == []


def test_a_real_sentence_end_is_found() -> None:
    assert boundaries("Das ist fertig. Und weiter") == [len("Das ist fertig.")]


def test_an_abbreviation_does_not_hide_the_following_end() -> None:
    text = "Nutze z.B. den Editor. Danach weiter"
    assert boundaries(text) == [len("Nutze z.B. den Editor.")]


def test_clause_boundaries_are_optional() -> None:
    text = "Der Speicher reicht, es sind viele Gigabyte frei"
    assert boundaries(text, clauses=False) == []
    assert boundaries(text, clauses=True) == [len("Der Speicher reicht,")]


# --- chunking ---------------------------------------------------------------


def test_a_url_is_never_split_across_chunks() -> None:
    text = "Die Anleitung steht unter https://docs.example.com/a/b/c und hilft dir weiter."
    for chunk in _stream(text, first_chunk_target_chars=20, min_chunk_chars=20):
        assert "https" not in chunk  # dropped by policy, never half-spoken
        assert "docs.example" not in chunk


def test_a_decimal_survives_a_chunk_boundary() -> None:
    text = "Der Faktor liegt bei 1,33 fach Echtzeit und das reicht uns fürs Erste aus."
    joined = " ".join(_stream(text))
    assert "1,33" in joined


def test_the_first_chunk_comes_early() -> None:
    text = "Der Speicherplatz ist ausreichend, es sind noch fünfhundert Gigabyte frei."
    chunks = _stream(text, first_chunk_target_chars=25, min_chunk_chars=60)
    assert len(chunks) >= 2
    assert chunks[0].endswith(",")
    assert len(chunks[0]) < len(text)


def test_later_chunks_prefer_whole_sentences() -> None:
    text = "Kurzer Start hier. Danach kommt ein längerer Satz, der ein Komma enthält."
    chunks = _stream(text, first_chunk_target_chars=10, min_chunk_chars=10, max_wait_ms=10_000)
    assert chunks[0] == "Kurzer Start hier."
    assert any("," in c for c in chunks[1:])


def test_nothing_unspeakable_is_ever_emitted() -> None:
    for chunk in _stream("Fertig. ... !!! ---   . Weiter geht es hier."):
        assert is_speakable(chunk)


def test_an_open_code_fence_holds_everything_back() -> None:
    chunker = _chunker()
    assert chunker.feed("Beispiel:\n```bash\ndf -h", now=0.0) == []
    assert chunker.pending != ""


def test_a_closed_code_fence_is_dropped_but_the_prose_survives() -> None:
    chunks = _stream("Erst Text hier drin. ```bash\ndf -h\n``` Und danach noch mehr Text.")
    joined = " ".join(chunks)
    assert "df -h" not in joined
    assert "Erst Text hier drin." in joined
    assert "danach" in joined


def test_flush_returns_the_remainder() -> None:
    chunker = _chunker(first_chunk_target_chars=500, min_chunk_chars=500)
    assert chunker.feed("Ein kurzer Rest ohne Ende", now=0.0) == []
    assert chunker.flush() == ["Ein kurzer Rest ohne Ende"]


def test_flush_on_an_empty_stream_yields_nothing() -> None:
    assert _chunker().flush() == []
    assert _chunker().feed("", now=0.0) == []


def test_a_very_long_run_without_punctuation_is_still_cut() -> None:
    text = "wort " * 120
    chunks = _stream(text, max_chunk_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 160 for c in chunks), [len(c) for c in chunks]


def test_reset_starts_a_fresh_answer() -> None:
    chunker = _chunker()
    chunker.feed("Ein erster fertiger Satz steht hier.", now=0.0)
    assert chunker.emitted_first is True
    chunker.reset()
    assert chunker.emitted_first is False
    assert chunker.pending == ""


# --- waiting ----------------------------------------------------------------


def test_a_clause_is_only_used_after_the_wait_elapsed() -> None:
    """Later chunks hold out for a sentence until max_wait_ms passes."""
    chunker = StreamingChunker(
        ChunkerConfig(first_chunk_target_chars=5, min_chunk_chars=5, max_wait_ms=500)
    )
    assert chunker.feed("Erster Satz fertig.", now=0.0) == ["Erster Satz fertig."]
    # Second chunk: a clause exists, but the wait has not elapsed yet.
    assert chunker.feed(" Dann kommt mehr, und noch mehr", now=0.1) == []
    late = chunker.feed("", now=1.0)
    assert late and late[0].endswith(",")


# --- configuration ----------------------------------------------------------


def test_defaults_toml_matches_the_documented_streaming_values() -> None:
    config = ChunkerConfig.from_mapping(default_mapping()["tts"]["streaming"])
    assert config.first_chunk_target_chars == 80
    assert config.min_chunk_chars == 55
    assert config.max_chunk_chars == 180
    assert config.max_wait_ms == 500
    assert config.prefetch_chunks == 1
    assert config.cancel_pending_on_interrupt is True


def test_unreadable_streaming_values_keep_their_defaults() -> None:
    config = ChunkerConfig.from_mapping({"min_chunk_chars": "viele", "max_wait_ms": None})
    assert config.min_chunk_chars == 55
    assert config.max_wait_ms == 500
