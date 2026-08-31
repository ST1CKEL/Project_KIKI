"""Per-turn clocks. Law 5: without numbers, speed is a feeling."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


@dataclass
class TurnTelemetry:
    turn_id: str
    t_wake: float = 0.0
    t_ack: float = 0.0
    t_eos: float = 0.0
    t_stt_final: float = 0.0
    t_llm_first_token: float = 0.0
    t_tts_first_pcm: float = 0.0
    t_playback_start: float = 0.0
    t_playback_end: float = 0.0
    source: str = ""
    stt_latency_ms: float = 0.0
    ttft_ms: float = 0.0
    ttfa_ms: float = 0.0
    ack_ms: float = 0.0
    transcript: str = ""
    answer_preview: str = ""
    error: str = ""

    def mark_wake(self, source: str = "") -> None:
        self.t_wake = time.perf_counter()
        self.source = source

    def mark_ack(self) -> None:
        self.t_ack = time.perf_counter()

    def calculate(self) -> None:
        if self.t_wake and self.t_ack:
            self.ack_ms = round((self.t_ack - self.t_wake) * 1000.0, 1)
        if self.t_eos and self.t_stt_final:
            self.stt_latency_ms = round((self.t_stt_final - self.t_eos) * 1000.0, 1)
        if self.t_stt_final and self.t_llm_first_token:
            self.ttft_ms = round((self.t_llm_first_token - self.t_stt_final) * 1000.0, 1)
        if self.t_eos and self.t_playback_start:
            self.ttfa_ms = round((self.t_playback_start - self.t_eos) * 1000.0, 1)

    def as_dict(self) -> dict[str, object]:
        self.calculate()
        return asdict(self)

    def summary(self) -> str:
        self.calculate()
        return (
            f"turn {self.turn_id} ack={self.ack_ms:.0f}ms "
            f"stt={self.stt_latency_ms:.0f}ms ttft={self.ttft_ms:.0f}ms "
            f"ttfa={self.ttfa_ms:.0f}ms source={self.source or '-'}"
        )


@dataclass
class TelemetryLog:
    turns: list[TurnTelemetry] = field(default_factory=list)
    keep: int = 20

    def add(self, turn: TurnTelemetry) -> None:
        turn.calculate()
        self.turns.append(turn)
        if len(self.turns) > self.keep:
            self.turns = self.turns[-self.keep :]

    def last(self) -> TurnTelemetry | None:
        return self.turns[-1] if self.turns else None
