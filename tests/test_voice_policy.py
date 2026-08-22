"""Voice policy: how much KIKI says, and what she must never say aloud."""

from __future__ import annotations

import pytest

from kiki.config.settings import default_mapping
from kiki.voice.tts import VoiceMode, VoicePolicyConfig, VoiceResponsePolicy


def _policy(**overrides) -> VoiceResponsePolicy:
    return VoiceResponsePolicy(VoicePolicyConfig(**overrides))


# --- modes ------------------------------------------------------------------


def test_silent_speaks_nothing() -> None:
    plan = _policy().plan("Alles bestens.", mode=VoiceMode.SILENT)
    assert plan.speaks is False
    assert plan.text == ""


def test_concise_stops_after_two_sentences() -> None:
    answer = "Erstens. Zweitens. Drittens. Viertens."
    plan = _policy().plan(answer, mode=VoiceMode.CONCISE)
    assert plan.text == "Erstens. Zweitens."
    assert plan.truncated is True


def test_normal_allows_three() -> None:
    answer = "Erstens. Zweitens. Drittens. Viertens."
    plan = _policy().plan(answer, mode=VoiceMode.NORMAL)
    assert plan.text == "Erstens. Zweitens. Drittens."


def test_detailed_stays_capped_unless_explicitly_enabled() -> None:
    answer = "Eins. Zwei. Drei. Vier. Fünf."
    capped = _policy().plan(answer, mode=VoiceMode.DETAILED)
    assert capped.text == "Eins. Zwei. Drei."

    full = _policy(detailed_speech=True).plan(answer, mode=VoiceMode.DETAILED)
    assert full.text == answer
    assert full.truncated is False


def test_character_cap_ends_on_a_sentence_when_it_can() -> None:
    answer = "Das ist ein erster Satz. " + "Und noch viel mehr Text. " * 20
    plan = _policy(concise_max_sentences=0, concise_max_characters=60).plan(
        answer, mode=VoiceMode.CONCISE
    )
    assert len(plan.text) <= 60
    assert plan.text.endswith(".")
    assert plan.truncated is True


def test_the_default_mode_is_used_when_none_is_given() -> None:
    plan = _policy(default_mode=VoiceMode.SILENT).plan("Hallo.")
    assert plan.speaks is False


def test_empty_input_never_speaks() -> None:
    for text in ("", "   ", "\n\n"):
        assert _policy().plan(text).speaks is False


# --- what must not be spoken ------------------------------------------------


def test_code_fences_are_never_spoken() -> None:
    plan = _policy().plan("Nutze das hier:\n```bash\nsudo rm -rf /tmp/x\n```\nFertig.")
    assert "sudo" not in plan.text
    assert "rm" not in plan.text
    assert "code" in plan.removed
    assert "Fertig." in plan.text


def test_an_unclosed_fence_is_also_dropped() -> None:
    """A still-streaming code block must not leak its first lines."""
    plan = _policy().plan("Beispiel:\n```python\nimport os\nos.system(")
    assert "import" not in plan.text
    assert "os.system" not in plan.text


def test_inline_code_is_dropped() -> None:
    plan = _policy().plan("Führe `systemctl restart nginx` aus.")
    assert "systemctl" not in plan.text
    assert "code" in plan.removed


@pytest.mark.parametrize(
    "secret",
    [
        "sk-abc123def456ghi789",
        "ghp_AAAABBBBCCCCDDDDEEEEFFFFGGGG",
        "api_key: hunter2supersecret",
        "token = eyJhbGciOiJIUzI1NiJ9abcdef",
        "AKIAIOSFODNN7EXAMPLE",
    ],
)
def test_secrets_are_never_spoken(secret) -> None:
    plan = _policy().plan(f"Dein Zugang lautet {secret} und ist gültig.")
    assert secret.split()[-1] not in plan.text
    assert "secrets" in plan.removed


def test_a_secret_inside_a_code_block_is_removed_too() -> None:
    plan = _policy().plan("```\nexport API_KEY=sk-abcdef123456\n```")
    assert "sk-" not in plan.text
    assert plan.speaks is False


def test_urls_are_not_spoken_but_the_link_text_survives() -> None:
    plan = _policy().plan("Lies die [GTK-Doku](https://docs.gtk.org/gtk4/) durch.")
    assert "https" not in plan.text
    assert "docs.gtk.org" not in plan.text
    assert "GTK-Doku" in plan.text


def test_bare_urls_are_dropped() -> None:
    plan = _policy().plan("Siehe https://example.com/pfad?x=1 für Details.")
    assert "example.com" not in plan.text
    assert "urls" in plan.removed
    assert "Details" in plan.text


def test_paths_are_not_spoken() -> None:
    plan = _policy().plan("Die Datei liegt in /home/martin/.config/kiki/config.toml dort.")
    assert "/home/martin" not in plan.text
    assert "paths" in plan.removed


def test_tables_are_not_spoken() -> None:
    answer = "Ergebnis:\n| Name | Wert |\n|---|---|\n| RTF | 1.33 |\nSoweit."
    plan = _policy().plan(answer)
    assert "RTF" not in plan.text
    assert "tables" in plan.removed
    assert "Soweit." in plan.text


def test_log_lines_are_not_spoken() -> None:
    """Real logs sit on their own lines, which is what the rule anchors on."""
    answer = "Fehler gefunden.\n2026-08-22 10:11 ERROR kiki.voice: boom\nIch prüfe das."
    plan = _policy().plan(answer)
    assert "boom" not in plan.text
    assert "logs" in plan.removed
    assert "Ich prüfe das." in plan.text


def test_a_bullet_list_is_not_mistaken_for_a_diff() -> None:
    """A leading "-" is a markdown bullet unless a diff header says otherwise."""
    plan = _policy().plan("- **Wichtig**: das zählt.\n- Und das auch.")
    assert "Wichtig" in plan.text
    assert "diff" not in plan.removed


def test_diff_output_is_not_spoken() -> None:
    answer = "Änderung:\n--- a/x.py\n+++ b/x.py\n+    neue_zeile()\nFertig."
    plan = _policy().plan(answer)
    assert "neue_zeile" not in plan.text
    assert "diff" in plan.removed


def test_switching_a_flag_on_allows_that_category() -> None:
    plan = _policy(speak_urls=True).plan("Siehe https://example.com dort.")
    assert "example.com" in plan.text
    assert "urls" not in plan.removed


def test_removed_reports_categories_never_content() -> None:
    """A diagnostic field must not become the place a secret leaks."""
    plan = _policy().plan("Key sk-abcdef123456 und ```code``` und https://x.io")
    assert set(plan.removed) <= {"secrets", "code", "logs", "diff", "tables", "urls", "paths"}
    for entry in plan.removed:
        assert "sk-" not in entry


def test_markdown_markup_is_stripped_from_speech() -> None:
    plan = _policy().plan("## Titel\n- **Wichtig**: das hier zählt.")
    assert "#" not in plan.text
    assert "*" not in plan.text
    assert "Wichtig" in plan.text


# --- configuration ----------------------------------------------------------


def test_defaults_toml_matches_the_documented_policy() -> None:
    config = VoicePolicyConfig.from_mapping(default_mapping()["voice"]["response_policy"])
    assert config.default_mode is VoiceMode.CONCISE
    assert config.concise_max_sentences == 2
    assert config.concise_max_characters == 300
    assert config.normal_max_sentences == 3
    assert config.normal_max_characters == 500
    assert config.detailed_speech is False
    for flag in ("speak_code", "speak_logs", "speak_urls", "speak_paths", "speak_tables", "speak_secrets"):
        assert getattr(config, flag) is False, flag


@pytest.mark.parametrize("bad", ["", "laut", "LOUD", None, 5])
def test_an_unreadable_mode_falls_back_to_concise(bad) -> None:
    """A broken config must not make KIKI talk more than intended."""
    config = VoicePolicyConfig.from_mapping({"default_mode": bad})
    assert config.default_mode is VoiceMode.CONCISE


def test_unreadable_numbers_keep_their_defaults() -> None:
    config = VoicePolicyConfig.from_mapping(
        {"concise_max_sentences": "viele", "normal_max_characters": None}
    )
    assert config.concise_max_sentences == 2
    assert config.normal_max_characters == 500
