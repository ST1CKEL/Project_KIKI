"""openWakeWord spotter for the German name "KIKI".

Law 1: there is no silent fall-back to Vosk or to English "hey jarvis".
If the custom KIKI ONNX is missing, this spotter is *not ready*. Hotkey and
pet-click still work — those are explicit user actions, not a worse ear.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from kiki.paths import user_data_dir

log = logging.getLogger("kiki.audio.wake")

SAMPLE_RATE = 16000
# openWakeWord predicts on 80 ms frames (1280 samples @ 16 kHz).
OWW_FRAME_SAMPLES = 1280
OWW_FRAME_BYTES = OWW_FRAME_SAMPLES * 2


class WakeSpotter(Protocol):
    ready: bool
    error: str
    backend: str

    def feed(self, pcm: bytes) -> bool: ...


def default_model_path(model_name: str = "kiki") -> Path:
    data = user_data_dir() / "wake"
    for candidate in (
        data / f"{model_name}.onnx",
        data / f"{model_name}.tflite",
        data / "kiki.onnx",
    ):
        if candidate.is_file():
            return candidate
    return data / f"{model_name}.onnx"


class OpenWakeWordSpotter:
    backend = "openwakeword"

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        model_name: str = "kiki",
        threshold: float = 0.58,
        consecutive_frames: int = 2,
    ) -> None:
        self.threshold = threshold
        self.consecutive_frames = max(1, consecutive_frames)
        self.ready = False
        self.error = ""
        self._model: Any = None
        self._key = ""
        self._hits = 0
        self._carry = bytearray()
        self.model_path = Path(model_path) if model_path else default_model_path(model_name)
        self._load()

    def _load(self) -> None:
        if not self.model_path.is_file():
            self.error = (
                f"Kein KIKI-Wake-Modell unter {self.model_path}. "
                "openWakeWord braucht ein trainiertes ONNX für „KIKI“, "
                "kein englisches Standardwort. Lege die Datei ab oder "
                "führe scripts/setup-wakeword.sh aus. "
                "Hotkey (Super+K) und Klick auf die Figur funktionieren trotzdem."
            )
            log.error("%s", self.error)
            return
        try:
            from openwakeword.model import Model
        except Exception as exc:
            self.error = (
                f"openWakeWord ist nicht installiert ({exc}). "
                "Kein stiller Wechsel auf Vosk."
            )
            log.error("%s", self.error)
            return
        try:
            model = Model(
                wakeword_models=[str(self.model_path)],
                inference_framework="onnx",
            )
        except Exception as exc:
            self.error = f"Wake-Modell ließ sich nicht laden: {exc}"
            log.error("%s", self.error)
            return
        keys = list(getattr(model, "models", {}) or [])
        self._key = keys[0] if keys else self.model_path.stem
        self._model = model
        self.ready = True
        log.info("openWakeWord ready model=%s threshold=%.2f", self.model_path.name, self.threshold)

    def feed(self, pcm: bytes) -> bool:
        if not self.ready or self._model is None or not pcm:
            return False
        self._carry.extend(pcm)
        triggered = False
        while len(self._carry) >= OWW_FRAME_BYTES:
            frame = bytes(self._carry[:OWW_FRAME_BYTES])
            del self._carry[:OWW_FRAME_BYTES]
            if self._predict(frame):
                triggered = True
        return triggered

    def _predict(self, frame: bytes) -> bool:
        import numpy as np

        audio = np.frombuffer(frame, dtype=np.int16)
        try:
            scores = self._model.predict(audio)
        except Exception:
            return False
        if not isinstance(scores, dict):
            return False
        score = 0.0
        if self._key and self._key in scores:
            score = float(scores[self._key])
        elif scores:
            score = float(max(scores.values()))
        if score >= self.threshold:
            self._hits += 1
        else:
            self._hits = 0
        if self._hits >= self.consecutive_frames:
            self._hits = 0
            return True
        return False


class MissingWakeSpotter:
    """Honest stand-in: never matches, always explains why."""

    backend = "missing"
    ready = False

    def __init__(self, error: str) -> None:
        self.error = error

    def feed(self, pcm: bytes) -> bool:  # noqa: ARG002
        return False


def build_wake_spotter(
    *,
    engine: str = "openwakeword",
    model_name: str = "kiki",
    threshold: float = 0.58,
    consecutive_frames: int = 2,
    factory: Callable[..., WakeSpotter] | None = None,
) -> WakeSpotter:
    if factory is not None:
        return factory()
    name = (engine or "openwakeword").strip().lower()
    if name not in {"openwakeword", "oww", "kiki"}:
        return MissingWakeSpotter(
            f"Unbekannte Wake-Engine {engine!r}. "
            "Vosk und englische Standard-Wake-Wörter sind absichtlich nicht verdrahtet."
        )
    return OpenWakeWordSpotter(
        model_name=model_name,
        threshold=threshold,
        consecutive_frames=consecutive_frames,
    )
