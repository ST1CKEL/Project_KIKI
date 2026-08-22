"""HTTP client for the local Qwen3-TTS service. Loopback by default."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_TTS_URL = "http://127.0.0.1:18765"
DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_SPEAKER = "Serena"
DEFAULT_LANGUAGE = "German"
MAX_WAV_BYTES = 64 * 1024 * 1024
MAX_ERROR_BYTES = 64 * 1024
MAX_WAV_SECONDS = 15 * 60

CUSTOM_VOICE_SPEAKERS: tuple[str, ...] = (
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
)

CUSTOM_VOICE_LANGUAGES: tuple[str, ...] = (
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
)


class TtsError(Exception):
    """The TTS service refused or could not be reached."""


@dataclass(frozen=True)
class TtsHealth:
    ok: bool
    ready: bool
    detail: str
    device: str = ""
    model: str = ""
    speakers: tuple[str, ...] = ()
    dummy: bool = False


async def _read_bounded(
    response: httpx.Response,
    *,
    limit: int,
    description: str,
) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > limit:
            mib = 1024 * 1024
            limit_label = (
                f"{limit // mib} MiB" if limit >= mib and limit % mib == 0 else f"{limit} Bytes"
            )
            raise TtsError(f"{description} ist zu groß (maximal {limit_label}).")
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_wav(payload: bytes) -> None:
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise TtsError("TTS lieferte keine gültigen RIFF/WAVE-Daten.")
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            frame_count = wav.getnframes()
            compression = wav.getcomptype()
            if channels not in {1, 2}:
                raise TtsError(f"Ungültige WAV-Kanalzahl: {channels}.")
            if sample_width not in {1, 2, 3, 4}:
                raise TtsError(f"Ungültige WAV-Samplebreite: {sample_width} Bytes.")
            if not 8_000 <= frame_rate <= 192_000:
                raise TtsError(f"Ungültige WAV-Abtastrate: {frame_rate} Hz.")
            if compression != "NONE":
                raise TtsError(f"Komprimiertes WAV-Audio wird nicht unterstützt: {compression}.")
            if frame_count <= 0:
                raise TtsError("TTS lieferte ein WAV ohne Audiodaten.")
            if frame_count / frame_rate > MAX_WAV_SECONDS:
                raise TtsError("TTS-WAV ist unplausibel lang.")
            frames = wav.readframes(frame_count)
    except TtsError:
        raise
    except (EOFError, OSError, wave.Error) as exc:
        raise TtsError(f"TTS lieferte ein ungültiges WAV: {exc}") from exc
    expected = frame_count * channels * sample_width
    if len(frames) != expected:
        raise TtsError("TTS lieferte unvollständige WAV-Audiodaten.")


def _detail_from_payload(payload: dict[str, Any], *, status: int) -> str:
    if payload.get("detail"):
        return str(payload["detail"])
    if payload.get("error"):
        return str(payload["error"])
    if payload.get("ok"):
        device = str(payload.get("device") or "")
        model = str(payload.get("model") or "")
        bits = ["TTS erreichbar"]
        if device:
            bits.append(device)
        if model:
            bits.append(model)
        return " · ".join(bits)
    return f"TTS HTTP {status}"


async def tts_health(base_url: str, *, timeout: float = 2.5) -> TtsHealth:
    url = base_url.rstrip("/") + "/health"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
    except httpx.ConnectError:
        return TtsHealth(ok=False, ready=False, detail="TTS-Dienst nicht erreichbar")
    except httpx.TimeoutException:
        return TtsHealth(ok=False, ready=False, detail="TTS-Dienst antwortet nicht (Timeout)")
    except httpx.HTTPError as exc:
        return TtsHealth(ok=False, ready=False, detail=str(exc))
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    ok = response.status_code == 200 and bool(payload.get("ok", False))
    ready = bool(payload.get("ready", ok))
    speakers = payload.get("speakers") or []
    names = tuple(str(s) for s in speakers) if isinstance(speakers, list) else ()
    return TtsHealth(
        ok=ok,
        ready=ready,
        detail=_detail_from_payload(payload, status=response.status_code),
        device=str(payload.get("device") or ""),
        model=str(payload.get("model") or ""),
        speakers=names,
        dummy=bool(payload.get("dummy", False)),
    )


async def synthesize_wav(
    base_url: str,
    text: str,
    *,
    dest: Path,
    language: str = DEFAULT_LANGUAGE,
    speaker: str = DEFAULT_SPEAKER,
    timeout: float = 180.0,
) -> Path:
    cleaned = text.strip()
    if not cleaned:
        raise TtsError("Leerer TTS-Text.")
    url = base_url.rstrip("/") + "/v1/synthesize"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                url,
                json={"text": cleaned, "language": language, "speaker": speaker},
                headers={"Accept": "audio/wav"},
            ) as response:
                if response.status_code != 200:
                    body = await _read_bounded(
                        response,
                        limit=MAX_ERROR_BYTES,
                        description="TTS-Fehlerantwort",
                    )
                    snippet = body.decode("utf-8", "replace")[:240].strip()
                    raise TtsError(f"TTS-Fehler: {snippet or f'HTTP {response.status_code}'}")
                content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
                if content_type not in {"audio/wav", "audio/x-wav", "application/octet-stream", ""}:
                    raise TtsError(f"Unerwarteter TTS-Inhalt: {content_type or 'unbekannt'}")
                content = await _read_bounded(
                    response,
                    limit=MAX_WAV_BYTES,
                    description="TTS-WAV",
                )
    except httpx.ConnectError as exc:
        raise TtsError("TTS-Dienst nicht erreichbar") from exc
    except httpx.TimeoutException as exc:
        raise TtsError("TTS-Dienst Timeout") from exc
    except httpx.HTTPError as exc:
        raise TtsError(str(exc)) from exc
    _validate_wav(content)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return dest
