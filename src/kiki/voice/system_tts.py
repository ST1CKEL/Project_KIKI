"""Always-local speech fallback via Fedora's espeak-ng package."""

from __future__ import annotations

import asyncio
import logging
import shutil
import wave
from pathlib import Path

from kiki.voice.tts_client import TtsError

log = logging.getLogger(__name__)

DEFAULT_SYSTEM_VOICE = "de+f3"
DEFAULT_SYSTEM_RATE = 165
DEFAULT_SYSTEM_PITCH = 48
MAX_SYSTEM_TEXT_CHARS = 4000


def system_tts_binary() -> str | None:
    installed = Path("/usr/bin/espeak-ng")
    if installed.is_file():
        return str(installed)
    return shutil.which("espeak-ng")


def system_tts_available() -> bool:
    return system_tts_binary() is not None


async def synthesize_system_wav(
    text: str,
    *,
    dest: Path,
    voice: str = DEFAULT_SYSTEM_VOICE,
    rate: int = DEFAULT_SYSTEM_RATE,
    pitch: int = DEFAULT_SYSTEM_PITCH,
    timeout: float = 30.0,
) -> Path:
    """Render a WAV without network or a model service."""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        raise TtsError("Leerer TTS-Text.")
    if len(cleaned) > MAX_SYSTEM_TEXT_CHARS:
        raise TtsError(f"TTS-Text ist länger als {MAX_SYSTEM_TEXT_CHARS} Zeichen.")
    binary = system_tts_binary()
    if binary is None:
        raise TtsError("Lokale Systemstimme fehlt (`sudo dnf install espeak-ng`).")

    safe_rate = max(80, min(320, int(rate)))
    safe_pitch = max(0, min(99, int(pitch)))
    safe_voice = voice if voice and all(ch.isalnum() or ch in "+-_" for ch in voice) else "de"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # espeak-ng decides the output format from the filename.  Keep ``.wav`` as
    # the final suffix while rendering atomically; ``.wav.part`` can make some
    # builds fall back to the audio device and block indefinitely.
    partial = dest.with_name(f".{dest.stem}.part.wav")
    partial.unlink(missing_ok=True)
    try:
        # espeak-ng can deadlock when initialized in a worker thread on some
        # Linux audio stacks.  Run it as an asyncio child process instead.
        proc = await asyncio.create_subprocess_exec(
            binary,
            "-v",
            safe_voice,
            "-s",
            str(safe_rate),
            "-p",
            str(safe_pitch),
            "-w",
            str(partial),
            "--stdin",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(
            proc.communicate(cleaned.encode("utf-8")), timeout=timeout
        )
    except TimeoutError as exc:
        proc.kill()
        await proc.wait()
        partial.unlink(missing_ok=True)
        raise TtsError(f"Lokale Systemstimme antwortet nach {timeout:g} s nicht.") from exc
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise TtsError(f"Lokale Systemstimme fehlgeschlagen: {exc}") from exc
    if proc.returncode != 0:
        partial.unlink(missing_ok=True)
        detail = stderr.decode("utf-8", "replace").strip()[:240]
        raise TtsError(f"Lokale Systemstimme fehlgeschlagen: {detail or proc.returncode}")
    try:
        with wave.open(str(partial), "rb") as wf:
            valid = wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getnframes() > 0
    except (OSError, wave.Error) as exc:
        partial.unlink(missing_ok=True)
        raise TtsError(f"Lokale Systemstimme lieferte keine gültige WAV-Datei: {exc}") from exc
    if not valid:
        partial.unlink(missing_ok=True)
        raise TtsError("Lokale Systemstimme lieferte keine gültige WAV-Datei.")
    partial.replace(dest)
    log.info("system TTS rendered %s", dest.name)
    return dest
