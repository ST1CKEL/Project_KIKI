"""A complete answer stays in chat while TTS receives its safe companion."""

from __future__ import annotations

from kiki.voice.answer import CHAT_NOTICE, plan_voice_answer
from kiki.voice.tts.policy import VoicePolicyConfig, VoiceResponsePolicy


def _policy(**overrides) -> VoiceResponsePolicy:
    return VoiceResponsePolicy(VoicePolicyConfig(**overrides))


def test_long_answer_is_shortened_once_as_a_whole() -> None:
    answer = "Erstens. Zweitens. Drittens. Viertens. Fünftens. Sechstens."

    delivery = plan_voice_answer(answer, policy=_policy())

    assert delivery.truncated
    assert delivery.open_chat
    assert delivery.spoken_text == f"Erstens. Zweitens. Drittens. {CHAT_NOTICE}"
    assert answer == "Erstens. Zweitens. Drittens. Viertens. Fünftens. Sechstens."


def test_short_plain_answer_does_not_open_chat() -> None:
    delivery = plan_voice_answer("Alles erledigt.", policy=_policy())

    assert delivery.spoken_text == "Alles erledigt."
    assert not delivery.truncated
    assert not delivery.open_chat


def test_privacy_redaction_opens_the_full_answer_in_chat() -> None:
    answer = "Die Datei liegt in /home/martin/privat.txt."

    delivery = plan_voice_answer(answer, policy=_policy())

    assert "/home/martin" not in delivery.spoken_text
    assert "paths" in delivery.removed
    assert delivery.open_chat
    assert delivery.spoken_text.endswith(CHAT_NOTICE)


def test_answer_that_is_entirely_unspoken_still_points_to_chat() -> None:
    delivery = plan_voice_answer("`geheimer_code()`", policy=_policy())

    assert delivery.spoken_text == CHAT_NOTICE
    assert delivery.open_chat


def test_disabling_concise_mode_keeps_all_safe_prose() -> None:
    answer = "Erstens. Zweitens. Drittens. Viertens."

    delivery = plan_voice_answer(answer, policy=_policy(), concise=False)

    assert delivery.spoken_text == answer
    assert not delivery.truncated
    assert not delivery.open_chat


def test_chat_auto_open_can_be_disabled_without_restoring_omitted_speech() -> None:
    answer = "Erstens. Zweitens. Drittens. Viertens."

    delivery = plan_voice_answer(
        answer,
        policy=_policy(),
        open_chat_for_details=False,
    )

    assert delivery.spoken_text == "Erstens. Zweitens. Drittens."
    assert delivery.truncated
    assert not delivery.open_chat
    assert CHAT_NOTICE not in delivery.spoken_text
