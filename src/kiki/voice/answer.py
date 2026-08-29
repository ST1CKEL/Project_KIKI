"""Plan the spoken companion to a complete voice answer.

The model's full answer remains untouched in chat.  This module only decides
what reaches TTS and whether omitted detail should bring the chat to the front.
Planning happens once the complete answer exists, never per streamed chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from kiki.voice.tts.policy import VoiceMode, VoiceResponsePolicy

CHAT_NOTICE = "Die vollständige Antwort ist im Chat geöffnet."


@dataclass(frozen=True)
class VoiceAnswerDelivery:
    spoken_text: str
    truncated: bool
    open_chat: bool
    removed: tuple[str, ...] = ()


def plan_voice_answer(
    answer: str,
    *,
    policy: VoiceResponsePolicy,
    concise: bool = True,
    open_chat_for_details: bool = True,
) -> VoiceAnswerDelivery:
    """Return a safe spoken answer and whether its full text needs the chat."""
    mode = VoiceMode.CONCISE if concise else VoiceMode.DETAILED
    planned_policy = policy
    if not concise and not policy.config.detailed_speech:
        # This product switch is the explicit permission required by DETAILED.
        # Clone only the immutable config; every privacy flag stays unchanged.
        planned_policy = VoiceResponsePolicy(replace(policy.config, detailed_speech=True))
    plan = planned_policy.plan(answer, mode=mode)
    omitted = plan.truncated or bool(plan.removed) or (bool((answer or "").strip()) and not plan.speaks)
    open_chat = bool(open_chat_for_details and omitted)
    spoken = plan.text.strip()
    if open_chat:
        spoken = f"{spoken} {CHAT_NOTICE}".strip()
    return VoiceAnswerDelivery(
        spoken_text=spoken,
        truncated=plan.truncated,
        open_chat=open_chat,
        removed=tuple(plan.removed),
    )
