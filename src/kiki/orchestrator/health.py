"""Honest subsystem health. Degraded is visible, never a smiling lie."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class CharacterState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"
    NOTIFICATION = "notification"


class HealthState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED_WAKE = "degraded_wake"
    DEGRADED_STT = "degraded_stt"
    DEGRADED_TTS = "degraded_tts"
    DEGRADED_LLM = "degraded_llm"
    DEGRADED_AUDIO = "degraded_audio"
    FATAL = "fatal"


@dataclass
class SubsystemHealth:
    audio: dict[str, object] = field(default_factory=dict)
    stt: dict[str, object] = field(default_factory=dict)
    tts: dict[str, object] = field(default_factory=dict)
    llm_ok: bool | None = None
    llm_detail: str = ""

    def issues(self) -> list[str]:
        out: list[str] = []
        for name, payload in (("audio", self.audio), ("stt", self.stt), ("tts", self.tts)):
            if not payload:
                continue
            status = str(payload.get("status") or "")
            if status == "failed" or payload.get("ready") is False:
                out.append(f"{name}_offline")
            elif status == "degraded" or payload.get("degraded"):
                err = str(payload.get("error") or payload.get("issues") or "degraded")
                out.append(f"{name}: {err}")
            extra = payload.get("issues")
            if isinstance(extra, list):
                out.extend(str(item) for item in extra)
        if self.llm_ok is False:
            out.append(self.llm_detail or "llm_offline")
        return out

    def overall(self) -> HealthState:
        if self.audio and self.audio.get("ready") is False:
            return HealthState.DEGRADED_AUDIO
        if self.stt and self.stt.get("ready") is False:
            return HealthState.DEGRADED_STT
        if self.tts and self.tts.get("ready") is False:
            return HealthState.DEGRADED_TTS
        if self.llm_ok is False:
            return HealthState.DEGRADED_LLM
        if self.audio.get("status") == "degraded" or (
            self.audio and self.audio.get("wake_ready") is False
        ):
            return HealthState.DEGRADED_WAKE
        if self.tts.get("degraded"):
            return HealthState.DEGRADED_TTS
        if self.stt.get("degraded"):
            return HealthState.DEGRADED_STT
        return HealthState.HEALTHY

    def spoken_fault(self) -> str:
        """First-person sentence KIKI should say instead of faking competence."""
        state = self.overall()
        mapping = {
            HealthState.DEGRADED_AUDIO: "Mein Mikrofon ist ausgefallen.",
            HealthState.DEGRADED_STT: "Meine Spracherkennung ist ausgefallen.",
            HealthState.DEGRADED_TTS: "Meine Stimme ist ausgefallen.",
            HealthState.DEGRADED_LLM: "Mein Denkmodul antwortet gerade nicht.",
            HealthState.DEGRADED_WAKE: (
                "Ich höre das Weckwort gerade nicht. "
                "Klick auf mich oder Super plus K funktioniert."
            ),
            HealthState.FATAL: "Ich bin nicht startklar.",
        }
        return mapping.get(state, "")
