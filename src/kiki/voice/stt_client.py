"""HTTP client for the local faster-whisper STT service. Loopback by default.

Vosk hears and segments inside the app; this client only asks the service to
write down the exact PCM passage the ear already captured. Every failure maps
to a plain error or a not-ok health value — the app falls back to the Vosk
text instead of blocking the voice turn.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import httpx

DEFAULT_STT_SERVICE_URL = "http://127.0.0.1:18775"
# Push-to-talk caps at the recorder's own limits; 60 s of audio is already a
# dictation, not a command. The server refuses beyond its own cap as well.
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_ERROR_BYTES = 64 * 1024
DEFAULT_TRANSCRIBE_TIMEOUT = 20.0


class SttServiceError(Exception):
    """The STT service refused or could not be reached."""


def pcm16_to_wav(
    pcm: bytes,
    *,
    rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Wrap raw PCM (the wake listener's capture format) into a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


@dataclass(frozen=True)
class SttHealth:
    ok: bool
    ready: bool
    detail: str
    device: str = ""
    model: str = ""
    dummy: bool = False


async def stt_health(base_url: str, *, timeout: float = 2.5) -> SttHealth:
    url = base_url.rstrip("/") + "/health"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
    except httpx.ConnectError:
        return SttHealth(ok=False, ready=False, detail="STT-Dienst nicht erreichbar")
    except httpx.TimeoutException:
        return SttHealth(ok=False, ready=False, detail="STT-Dienst antwortet nicht (Timeout)")
    except httpx.HTTPError as exc:
        return SttHealth(ok=False, ready=False, detail=str(exc))
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    ok = response.status_code == 200 and bool(payload.get("ok", False))
    ready = bool(payload.get("ready", ok))
    bits = ["STT erreichbar"] if ok else [f"STT HTTP {response.status_code}"]
    if payload.get("error"):
        bits = [str(payload["error"])]
    else:
        if payload.get("device"):
            bits.append(str(payload["device"]))
        if payload.get("model"):
            bits.append(str(payload["model"]))
    return SttHealth(
        ok=ok,
        ready=ready,
        detail=" · ".join(bits),
        device=str(payload.get("device") or ""),
        model=str(payload.get("model") or ""),
        dummy=bool(payload.get("dummy", False)),
    )


async def transcribe_wav_remote(
    base_url: str,
    wav_bytes: bytes,
    *,
    timeout: float = DEFAULT_TRANSCRIBE_TIMEOUT,
) -> str:
    """POST one WAV passage, return the transcript. Raises SttServiceError."""
    if not wav_bytes:
        raise SttServiceError("Leere STT-Anfrage.")
    if len(wav_bytes) > MAX_REQUEST_BYTES:
        raise SttServiceError("STT-Anfrage ist zu groß.")
    url = base_url.rstrip("/") + "/v1/transcribe"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                content=wav_bytes,
                headers={"Content-Type": "audio/wav"},
            )
    except httpx.ConnectError as exc:
        raise SttServiceError("STT-Dienst nicht erreichbar") from exc
    except httpx.TimeoutException as exc:
        raise SttServiceError("STT-Dienst Timeout") from exc
    except httpx.HTTPError as exc:
        raise SttServiceError(str(exc)) from exc
    if response.status_code != 200:
        snippet = response.text[:240].strip()
        raise SttServiceError(f"STT-Fehler: {snippet or f'HTTP {response.status_code}'}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise SttServiceError("STT-Dienst lieferte kein JSON.") from exc
    if not isinstance(payload, dict) or not payload.get("ok"):
        raise SttServiceError(str(payload.get("error") or "STT-Fehler."))
    return str(payload.get("text") or "").strip()
