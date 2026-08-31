"""Silero VAD over ONNX Runtime — no PyTorch in the audio process.

Target: end-of-speech in under 300 ms of silence. Energy VAD is not a hidden
fallback — it exists only when the config names it explicitly (`vad.model =
"energy"`), and that is reported as degraded.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from kiki.paths import user_data_dir

log = logging.getLogger("kiki.audio.vad")

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 512  # Silero v5 @ 16 kHz = 32 ms


class SpeechGate(Protocol):
    ready: bool
    error: str
    backend: str

    def is_speech(self, pcm: bytes) -> bool: ...


def silero_onnx_path() -> Path:
    return user_data_dir() / "vad" / "silero_vad.onnx"


class SileroSpeechGate:
    """Streaming Silero VAD: 32 ms / 512-sample frames at 16 kHz via ONNX."""

    backend = "silero_vad"

    def __init__(
        self,
        *,
        threshold: float = 0.55,
        sample_rate: int = SAMPLE_RATE,
        model_path: Path | None = None,
    ) -> None:
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.ready = False
        self.error = ""
        self._session: Any = None
        self._state: Any = None
        self._np: Any = None
        self.model_path = Path(model_path) if model_path else silero_onnx_path()
        self._load()

    def _load(self) -> None:
        if not self.model_path.is_file():
            self.error = (
                f"Silero-VAD-ONNX fehlt ({self.model_path}). "
                "Führe scripts/setup-wakeword.sh aus. Ohne VAD endet keine "
                "Äußerung zuverlässig — das ist kein Fallback, sondern ein Fehler."
            )
            log.error("%s", self.error)
            return
        try:
            import numpy as np
            import onnxruntime as ort
        except Exception as extra:
            self.error = f"onnxruntime/numpy fehlen ({extra}). Installiere das audio-venv."
            log.error("%s", self.error)
            return
        try:
            session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception as extra:
            self.error = f"Silero-ONNX ließ sich nicht laden: {extra}"
            log.error("%s", self.error)
            return
        names = {inp.name for inp in session.get_inputs()}
        if "input" not in names or "state" not in names:
            self.error = (
                f"Unerwartete Silero-Inputs {sorted(names)} — erwartet werden input/state/sr."
            )
            log.error("%s", self.error)
            return
        self._session = session
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._np = np
        self.ready = True
        log.info(
            "Silero VAD ONNX ready (threshold=%.2f path=%s)",
            self.threshold,
            self.model_path,
        )

    def is_speech(self, pcm: bytes) -> bool:
        if not self.ready or self._session is None or not pcm:
            return False
        np = self._np
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size < WINDOW_SAMPLES:
            audio = np.pad(audio, (0, WINDOW_SAMPLES - int(audio.size)))
        elif audio.size > WINDOW_SAMPLES:
            audio = audio[:WINDOW_SAMPLES]
        feeds: dict[str, Any] = {
            "input": audio.reshape(1, WINDOW_SAMPLES),
            "state": self._state,
        }
        if any(inp.name == "sr" for inp in self._session.get_inputs()):
            feeds["sr"] = np.array(self.sample_rate, dtype=np.int64)
        try:
            outputs = self._session.run(None, feeds)
        except Exception:
            log.debug("silero frame failed", exc_info=True)
            return False
        if not outputs:
            return False
        prob = float(np.reshape(outputs[0], (-1))[0])
        if len(outputs) > 1:
            self._state = outputs[1]
        return prob >= self.threshold


class EnergySpeechGate:
    """Explicit test/dev gate. Never selected unless config says `energy`."""

    backend = "energy"

    def __init__(self, *, noise_floor: float = 150.0) -> None:
        self.ready = True
        self.error = (
            "Energie-VAD ist aktiv — das ist der Testmodus, nicht Silero. "
            "Endpunktierung ist grob und darf nicht als Produktionsgehör gelten."
        )
        self._floor = noise_floor

    def is_speech(self, pcm: bytes) -> bool:
        if not pcm or len(pcm) < 4:
            return False
        samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        if rms < self._floor * 1.5:
            self._floor = self._floor * 0.98 + rms * 0.02
        return rms > max(60.0, self._floor * 1.35)


def build_speech_gate(
    model: str,
    *,
    threshold: float = 0.55,
    factory: Callable[..., SpeechGate] | None = None,
) -> SpeechGate:
    if factory is not None:
        return factory()
    name = (model or "silero_vad").strip().lower()
    if name in {"energy", "rms", "energy_vad"}:
        log.warning("VAD backend is explicit energy/RMS — reported as degraded")
        return EnergySpeechGate()
    return SileroSpeechGate(threshold=threshold)
