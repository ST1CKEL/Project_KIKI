"""Neural TTS engines with an honest German probe.

Law 1: English Kokoro voices, espeak-ng, and empty PCM are never a hidden
fallback. If the primary German engine cannot speak German, we say so.
A configured secondary (Piper) may take over *as degraded*, announced.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger("kiki.tts.engine")

PROBE_TEXT = "Guten Tag, ich bin Kiki."
STANDARD_PHRASES: dict[str, str] = {
    "ack": "Ja?",
    "wait": "Einen Moment.",
    "done": "Fertig.",
    "error": "Das hat leider nicht funktioniert.",
    "listen": "Ich höre zu.",
    "degraded_tts": "Meine Hauptstimme ist ausgefallen. Ich spreche mit der Ersatzstimme.",
    "stt_down": "Meine Spracherkennung ist ausgefallen.",
    "llm_down": "Mein Denkmodul antwortet gerade nicht.",
    "audio_down": "Mein Mikrofon ist ausgefallen.",
    "confirm_yes": "In Ordnung.",
    "confirm_no": "Gut, dann lasse ich das.",
    "vision_wait": "Ich mach das, gib mir einen Moment.",
}


class TtsEngine(Protocol):
    name: str
    voice: str
    ready: bool
    german_verified: bool
    error: str
    sample_rate: int

    def synthesize_pcm(self, text: str) -> bytes: ...


def _float_to_s16(audio: Any) -> bytes:
    import numpy as np

    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    scaled = np.clip(array * 32767.0, -32767, 32767).astype(np.int16)
    return scaled.tobytes()


class KokoroEngine:
    name = "kokoro"
    sample_rate = 24000

    def __init__(self, voice: str = "df_eva", device: str = "auto") -> None:
        self.voice = voice
        self.device = device
        self.ready = False
        self.german_verified = False
        self.error = ""
        self._pipeline: Any = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from kokoro import KPipeline
        except Exception as exc:
            self.error = f"Kokoro ist nicht installiert: {exc}"
            log.error("%s", self.error)
            return
        dev = "cuda:0" if torch.cuda.is_available() and self.device != "cpu" else "cpu"
        try:
            self._pipeline = KPipeline(lang_code="d", device=dev)
        except Exception as exc:
            self.error = (
                f"Kokoro hat keine deutsche Pipeline (lang_code=d): {exc}. "
                "Englische Phonemizer werden nicht still verwendet."
            )
            log.error("%s", self.error)
            return
        self.ready = True
        log.info("Kokoro German pipeline ready device=%s voice=%s", dev, self.voice)
        try:
            probe = self.synthesize_pcm(PROBE_TEXT)
        except Exception as exc:
            self.ready = False
            self.error = f"Kokoro-Stimme {self.voice} spricht kein Deutsch: {exc}"
            log.error("%s", self.error)
            return
        if len(probe) < 24000:  # < 0.5 s of 24 kHz s16
            self.ready = False
            self.german_verified = False
            self.error = (
                f"Kokoro-Stimme {self.voice} lieferte kein brauchbares deutsches Audio. "
                "Community-Gewichte (z. B. df_eva) müssen unter "
                "~/.local/share/kiki/tts/ liegen."
            )
            log.error("%s", self.error)
            return
        self.german_verified = True

    def synthesize_pcm(self, text: str) -> bytes:
        if not text.strip() or self._pipeline is None:
            return b""
        import numpy as np
        import torch

        segments: list[Any] = []
        generator = self._pipeline(text, voice=self.voice, speed=1.05, split_pattern=r"\n+")
        for _gs, _ps, audio in generator:
            if audio is None:
                continue
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            segments.append(audio)
        if not segments:
            raise RuntimeError(f"Kokoro lieferte kein Audio für Stimme {self.voice}")
        full = np.concatenate(segments, axis=0) if len(segments) > 1 else segments[0]
        return _float_to_s16(full)


class PiperEngine:
    """Verified German female path: piper + de_DE-eva_k-medium (or configured)."""

    name = "piper"
    sample_rate = 22050

    def __init__(self, voice: str = "de_DE-eva_k-medium") -> None:
        self.voice = voice
        self.ready = False
        self.german_verified = False
        self.error = ""
        self._bin = _find_piper_bin()
        self._model = _find_piper_model(voice)
        if self._bin is None:
            self.error = "piper Binary fehlt (scripts/setup-piper.sh)."
            log.error("%s", self.error)
            return
        if self._model is None:
            self.error = (
                f"Piper-Modell {voice} nicht gefunden. "
                "Lege onnx+json unter ~/.local/share/kiki/piper/ ab."
            )
            log.error("%s", self.error)
            return
        self.ready = True
        try:
            probe = self.synthesize_pcm(PROBE_TEXT)
        except Exception as exc:
            self.ready = False
            self.error = f"Piper-Probe fehlgeschlagen: {exc}"
            log.error("%s", self.error)
            return
        if len(probe) < 8000:
            self.ready = False
            self.error = "Piper lieferte zu kurzes Audio für den deutschen Probesatz."
            return
        self.german_verified = True
        log.info("Piper ready voice=%s", self._model)

    def synthesize_pcm(self, text: str) -> bytes:
        if not text.strip() or self._bin is None or self._model is None:
            return b""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            proc = subprocess.run(
                [self._bin, "--model", str(self._model), "--output_file", tmp.name],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:300])
            import wave

            with wave.open(tmp.name, "rb") as wav:
                self.sample_rate = wav.getframerate()
                return wav.readframes(wav.getnframes())


def _find_piper_bin() -> str | None:
    found = shutil.which("piper")
    if found:
        return found
    from kiki.paths import user_data_dir

    for candidate in (
        user_data_dir() / "piper-venv" / "bin" / "piper",
        user_data_dir() / "kokoro-venv" / "bin" / "piper",
        user_data_dir() / "audio-venv" / "bin" / "piper",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _find_piper_model(voice: str) -> Path | None:
    from kiki.paths import user_data_dir

    roots = [
        user_data_dir() / "piper",
        Path.home() / ".local/share/piper",
        Path("/usr/share/piper/voices"),
    ]
    name = voice if voice.endswith(".onnx") else f"{voice}.onnx"
    for root in roots:
        candidate = root / name
        if candidate.is_file():
            return candidate
        nested = root / voice / f"{voice}.onnx"
        if nested.is_file():
            return nested
    return None


def build_engine(name: str, voice: str, *, device: str = "auto") -> TtsEngine:
    key = (name or "kokoro").strip().lower()
    if key == "piper":
        return PiperEngine(voice=voice)
    if key == "kokoro":
        return KokoroEngine(voice=voice, device=device)
    raise ValueError(f"Unbekannte TTS-Engine {name!r} — espeak-ng ist kein gültiger Wert.")
