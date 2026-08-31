#!/usr/bin/env python3
"""Local Kokoro-TTS 82M HTTP service for KIKI.

High-speed, natural StyleTTS2-based German and English speech synthesis with
low latency (~20-40 ms per sentence) and live PCM streaming.

Binds loopback only. The GTK app never loads PyTorch or Kokoro directly.

    python kiki_kokoro_server.py --dummy          # no GPU/models, short tone (wiring test)
    python kiki_kokoro_server.py                  # Kokoro-82M on CUDA/CPU
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import select
import socket
import struct
import sys
import threading
import time
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from streaming_http import (  # noqa: E402
    BYTES_PER_SAMPLE,
    STREAM_CHANNELS,
    STREAM_FORMAT,
    STREAM_SAMPLE_RATE,
    CancelToken,
    EngineUnavailable,
    StreamGate,
    StreamSpec,
    StreamValidationError,
    pump_pcm,
    validate_stream_request,
)

log = logging.getLogger("kiki-kokoro-tts")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18765
DEFAULT_MODEL = "hexgrad/Kokoro-82M"
DEFAULT_SPEAKER = "df_sarah"
DEFAULT_LANGUAGE = "German"
MAX_TEXT_CHARS = 4000
MAX_BODY_BYTES = 256 * 1024
STREAM_SELECT_TIMEOUT = 0.25

KOKORO_SPEAKERS = [
    "df_sarah",
    "df_eva",
    "df_nicole",
    "dm_karl",
    "dm_sebastian",
    "af_heart",
    "af_bella",
    "af_nicole",
    "am_adam",
    "am_michael",
    "Serena",  # Alias mapped to df_sarah for backwards compatibility
]

KOKORO_LANGUAGES = [
    "German",
    "English",
    "French",
    "Spanish",
    "Italian",
    "Auto",
]

# Alias mapping for friendly or legacy speaker names
SPEAKER_ALIASES = {
    "serena": "df_sarah",
    "sarah": "df_sarah",
    "eva": "df_eva",
    "nicole": "df_nicole",
    "karl": "dm_karl",
    "sebastian": "dm_sebastian",
    "heart": "af_heart",
}

LANGUAGE_CODE_MAP = {
    "german": "d",
    "de": "d",
    "deutsch": "d",
    "english": "a",
    "en": "a",
    "french": "f",
    "fr": "f",
    "spanish": "e",
    "es": "e",
    "italian": "i",
    "it": "i",
    "auto": "d",
}


class SynthError(Exception):
    def __init__(self, message: str, status: int = 500) -> None:
        super().__init__(message)
        self.status = status


def _tone_wav(*, duration_s: float, freq: float = 220.0, rate: int = STREAM_SAMPLE_RATE) -> bytes:
    n = max(1, int(rate * duration_s))
    frames = bytearray()
    for i in range(n):
        envelope = 0.5 - abs((i / n) - 0.5)
        sample = int(12000 * envelope * math.sin(2 * math.pi * freq * i / rate))
        sample = max(-32767, min(32767, sample))
        frames.extend(struct.pack("<h", sample))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(frames))
    return buf.getvalue()


class DummySynthesizer:
    dummy = True
    ready = True
    model_id = "dummy-kokoro"
    device = "cpu"
    speakers = list(KOKORO_SPEAKERS)
    languages = list(KOKORO_LANGUAGES)

    def synthesize(self, text: str, *, language: str = DEFAULT_LANGUAGE, speaker: str = DEFAULT_SPEAKER) -> bytes:
        del language, speaker
        duration = min(1.6, 0.35 + min(len(text), 80) / 80.0)
        return _tone_wav(duration_s=duration, freq=330.0)

    def stream(self, spec: StreamSpec, token: CancelToken):
        chunk_samples = spec.sample_rate * spec.chunk_ms // 1000
        total_chunks = max(1, min(6, len(spec.text) // 20 + 1))
        for _ in range(total_chunks):
            if token.cancelled:
                return
            token.wait(0.03)
            # generate sine wave chunk
            raw = _tone_wav(duration_s=spec.chunk_ms / 1000.0, freq=330.0)
            # drop 44-byte wav header to get raw pcm
            pcm = raw[44:] if len(raw) > 44 else raw
            yield pcm


class KokoroSynthesizer:
    dummy = False
    ready = False

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "auto") -> None:
        import torch

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.device = device
        self.speakers = list(KOKORO_SPEAKERS)
        self.languages = list(KOKORO_LANGUAGES)
        log.info("initializing Kokoro pipeline on device=%s", device)

        self._pipelines: dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Pre-initialize German pipeline
        self._get_pipeline("d")
        self.ready = True
        log.info("Kokoro synthesizer ready (device=%s)", device)

    def _get_pipeline(self, lang_code: str) -> Any:
        if lang_code in self._pipelines:
            return self._pipelines[lang_code]
        try:
            from kokoro import KPipeline

            pipeline = KPipeline(lang_code=lang_code, device=self.device)
            self._pipelines[lang_code] = pipeline
            return pipeline
        except Exception as exc:
            log.exception("failed to initialize KPipeline for lang=%s", lang_code)
            raise SynthError(f"Kokoro Pipeline Initialisierungsfehler: {exc}") from exc

    def _resolve_speaker(self, wanted: str) -> str:
        text = str(wanted or "").strip().lower()
        if text in SPEAKER_ALIASES:
            return SPEAKER_ALIASES[text]
        for spk in self.speakers:
            if spk.lower() == text:
                return spk
        return DEFAULT_SPEAKER

    def _resolve_lang_code(self, language: str, speaker: str) -> str:
        lang_lower = str(language or "").strip().lower()
        if lang_lower in LANGUAGE_CODE_MAP:
            return LANGUAGE_CODE_MAP[lang_lower]
        # Infer language from speaker prefix if language is auto
        if speaker.startswith("df_") or speaker.startswith("dm_") or speaker.startswith("de_"):
            return "d"
        if speaker.startswith("af_") or speaker.startswith("am_"):
            return "a"
        return "d"

    def synthesize(self, text: str, *, language: str = DEFAULT_LANGUAGE, speaker: str = DEFAULT_SPEAKER) -> bytes:
        import numpy as np
        import soundfile as sf
        import torch

        resolved_speaker = self._resolve_speaker(speaker)
        lang_code = self._resolve_lang_code(language, resolved_speaker)

        with self._lock:
            pipeline = self._get_pipeline(lang_code)
            try:
                audio_segments: list[np.ndarray] = []
                generator = pipeline(text, voice=resolved_speaker, speed=1.0, split_pattern=r"\n+")
                for _, _, audio in generator:
                    if audio is not None:
                        if isinstance(audio, torch.Tensor):
                            audio = audio.detach().cpu().numpy()
                        audio_segments.append(audio)
                if not audio_segments:
                    raise SynthError("Kokoro hat kein Audio erzeugt.")
                full_audio = np.concatenate(audio_segments, axis=0) if len(audio_segments) > 1 else audio_segments[0]
                buf = io.BytesIO()
                sf.write(buf, full_audio, STREAM_SAMPLE_RATE, format="WAV", subtype="PCM_16")
                return buf.getvalue()
            except SynthError:
                raise
            except Exception as exc:
                log.exception("Kokoro synthesis failed")
                raise SynthError(f"Synthesefehler: {exc}") from exc

    def stream(self, spec: StreamSpec, token: CancelToken):
        import numpy as np
        import torch

        resolved_speaker = self._resolve_speaker(spec.speaker)
        lang_code = self._resolve_lang_code(spec.language, resolved_speaker)

        chunk_samples = spec.sample_rate * spec.chunk_ms // 1000
        chunk_bytes_len = chunk_samples * BYTES_PER_SAMPLE

        with self._lock:
            pipeline = self._get_pipeline(lang_code)
            try:
                generator = pipeline(spec.text, voice=resolved_speaker, speed=1.0, split_pattern=r"\n+")
                carry_bytes = bytearray()

                for _, _, audio in generator:
                    if token.cancelled:
                        return
                    if audio is None:
                        continue
                    if isinstance(audio, torch.Tensor):
                        audio = audio.detach().cpu().numpy()

                    # Convert float32 [-1.0, 1.0] to int16 PCM
                    scaled = np.clip(audio * 32767.0, -32767, 32767).astype(np.int16)
                    carry_bytes.extend(scaled.tobytes())

                    while len(carry_bytes) >= chunk_bytes_len:
                        if token.cancelled:
                            return
                        chunk = bytes(carry_bytes[:chunk_bytes_len])
                        del carry_bytes[:chunk_bytes_len]
                        yield chunk

                # Yield remainder if any
                if not token.cancelled and len(carry_bytes) >= BYTES_PER_SAMPLE:
                    usable = len(carry_bytes) - (len(carry_bytes) % BYTES_PER_SAMPLE)
                    yield bytes(carry_bytes[:usable])
            except Exception as exc:
                if token.cancelled:
                    return
                log.exception("Kokoro stream generation failed")
                raise EngineUnavailable(f"Generierungsfehler: {exc}") from exc


class KokoroStreamingEngineAdapter:
    def __init__(self, synth: Any) -> None:
        self._synth = synth

    @property
    def available(self) -> bool:
        return bool(self._synth is not None and getattr(self._synth, "ready", False))

    @property
    def reason(self) -> str | None:
        if self.available:
            return None
        return "engine_not_ready"

    def stream(self, spec: StreamSpec, token: CancelToken):
        if not self.available:
            raise EngineUnavailable("Kokoro Engine nicht verfügbar")
        yield from self._synth.stream(spec, token)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class TtsHandler(BaseHTTPRequestHandler):
    synthesizer: Any = None
    stream_engine: Any = None
    synth_lock = threading.Lock()
    stream_lock = threading.Lock()
    server_version = "kiki-kokoro-tts/0.1"

    def log_message(self, fmt: str, *args: object) -> None:
        log.info("%s - " + fmt, self.address_string(), *args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/health"}:
            synth = self.synthesizer
            if synth is None:
                self._send_json(503, {"ok": False, "ready": False, "error": "starting"})
                return
            self._send_json(
                200,
                {
                    "ok": True,
                    "ready": bool(getattr(synth, "ready", True)),
                    "dummy": bool(getattr(synth, "dummy", False)),
                    "device": str(getattr(synth, "device", "")),
                    "model": str(getattr(synth, "model_id", "Kokoro-82M")),
                    "speakers": list(getattr(synth, "speakers", KOKORO_SPEAKERS)),
                    "languages": list(getattr(synth, "languages", KOKORO_LANGUAGES)),
                    "streaming": self._streaming_available(),
                    "streaming_reason": self._streaming_reason(),
                    "stream_format": STREAM_FORMAT,
                    "stream_sample_rate": STREAM_SAMPLE_RATE,
                },
            )
            return
        if path in {"/v1/speakers", "/speakers"}:
            synth = self.synthesizer
            speakers = list(getattr(synth, "speakers", KOKORO_SPEAKERS)) if synth else KOKORO_SPEAKERS
            self._send_json(200, {"speakers": speakers})
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def _streaming_available(self) -> bool:
        engine = self.stream_engine
        return bool(engine is not None and getattr(engine, "available", False))

    def _streaming_reason(self) -> str | None:
        if self._streaming_available():
            return None
        engine = self.stream_engine
        if engine is None:
            return "no_engine"
        reason = getattr(engine, "reason", None)
        return str(reason) if reason else "runtime_incompatible"

    def _read_json_body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"ok": False, "error": "Content-Length fehlt"})
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(413, {"ok": False, "error": "Anfrage zu groß"})
            return None
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"ok": False, "error": "JSON ungültig"})
            return None
        if not isinstance(payload, dict):
            self._send_json(400, {"ok": False, "error": "JSON-Objekt erwartet"})
            return None
        return payload

    def _send_stream_headers(self, spec: Any) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "audio/pcm")
        self.send_header("X-KIKI-Audio-Format", STREAM_FORMAT)
        self.send_header("X-KIKI-Sample-Rate", str(STREAM_SAMPLE_RATE))
        self.send_header("X-KIKI-Channels", str(STREAM_CHANNELS))
        self.send_header("X-KIKI-Streaming", "true")
        self.send_header("X-KIKI-Chunk-Ms", str(spec.chunk_ms))
        self.send_header("X-KIKI-Transfer", "connection-close")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _peer_gone(self) -> bool:
        conn = getattr(self, "connection", None)
        if conn is None:
            return False
        try:
            ready, _writable, _err = select.select([conn], [], [], 0)
            if not ready:
                return False
            return conn.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def _write_stream(self, data: bytes) -> None:
        conn = self.connection
        view = memoryview(data)
        while view:
            readable, writable, _err = select.select([conn], [conn], [], STREAM_SELECT_TIMEOUT)
            if readable and conn.recv(1, socket.MSG_PEEK) == b"":
                raise BrokenPipeError("client closed the connection")
            if not writable:
                continue
            view = view[conn.send(view):]

    def _do_stream(self) -> None:
        engine = self.stream_engine
        if not self._streaming_available():
            self._send_json(503, {"ok": False, "streaming": False, "error": "Streaming nicht verfügbar"})
            return
        payload = self._read_json_body()
        if payload is None:
            return
        synth = self.synthesizer
        speakers = list(getattr(synth, "speakers", KOKORO_SPEAKERS))
        languages = list(getattr(synth, "languages", KOKORO_LANGUAGES))
        try:
            spec = validate_stream_request(
                payload,
                speakers=speakers,
                languages=languages,
                default_language=DEFAULT_LANGUAGE,
                default_speaker=DEFAULT_SPEAKER,
                max_text_chars=MAX_TEXT_CHARS,
            )
        except StreamValidationError as exc:
            self._send_json(exc.status, {"ok": False, "error": str(exc)})
            return

        gate = StreamGate(self.stream_lock)
        if not gate.acquire():
            self._send_json(503, {"ok": False, "error": "Eine Generation läuft bereits"})
            return

        token = CancelToken()
        try:
            outcome = pump_pcm(
                engine.stream(spec, token),
                on_first=lambda: self._send_stream_headers(spec),
                write=self._write_stream,
                token=token,
                peer_gone=self._peer_gone,
            )
        except EngineUnavailable as exc:
            log.info("stream refused: %s", type(exc).__name__)
            self._send_json(503, {"ok": False, "streaming": False, "error": "Streaming nicht verfügbar"})
            return
        except Exception as exc:
            log.warning("stream failed before first byte: %s", type(exc).__name__)
            self._send_json(500, {"ok": False, "error": "Generierung fehlgeschlagen"})
            return
        finally:
            token.cancel()
            gate.release()

        log.info(
            "stream id=%s chunks=%d bytes=%d audio=%.2fs cancelled=%s disconnected=%s%s",
            spec.request_id or "-",
            outcome.chunks,
            outcome.bytes_sent,
            outcome.audio_seconds,
            outcome.cancelled,
            outcome.disconnected,
            f" error={outcome.error}" if outcome.error else "",
        )

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/v1/synthesize/stream", "/synthesize/stream"}:
            self._do_stream()
            return
        if path not in {"/v1/synthesize", "/synthesize"}:
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        synth = self.synthesizer
        if synth is None or not getattr(synth, "ready", False):
            self._send_json(503, {"ok": False, "ready": False, "error": "TTS-Modell noch nicht geladen"})
            return
        payload = self._read_json_body()
        if payload is None:
            return
        text = str(payload.get("text") or "").strip()
        if not text:
            self._send_json(400, {"ok": False, "error": "text fehlt"})
            return
        if len(text) > MAX_TEXT_CHARS:
            self._send_json(413, {"ok": False, "error": f"text länger als {MAX_TEXT_CHARS} Zeichen"})
            return
        language = str(payload.get("language") or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
        speaker = str(payload.get("speaker") or DEFAULT_SPEAKER).strip() or DEFAULT_SPEAKER
        try:
            with self.synth_lock:
                wav = synth.synthesize(text, language=language, speaker=speaker)
        except SynthError as exc:
            self._send_json(exc.status, {"ok": False, "error": str(exc)})
            return
        except Exception as exc:
            log.exception("synthesize failed")
            self._send_json(500, {"ok": False, "error": str(exc)})
            return
        if not wav:
            self._send_json(500, {"ok": False, "error": "leere WAV"})
            return
        self._send(200, wav, "audio/wav")


def _bind_host(host: str) -> str:
    text = host.strip() or DEFAULT_HOST
    if text not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("kiki-kokoro-tts bindet nur Loopback (127.0.0.1 / ::1).")
    return "127.0.0.1" if text == "localhost" else text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KIKI Kokoro-TTS local service")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="cuda:0, cpu, or auto")
    parser.add_argument("--dummy", action="store_true", help="no GPU/model: return short tones")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s kiki-kokoro-tts: %(message)s",
    )
    host = _bind_host(args.host)
    if args.dummy:
        synth: Any = DummySynthesizer()
        log.info("dummy synthesizer mode active")
    else:
        try:
            synth = KokoroSynthesizer(args.model, args.device)
        except ImportError as exc:
            log.error("Kokoro / PyTorch fehlen (%s). scripts/setup-kokoro-tts.sh oder --dummy.", exc)
            return 1
        except Exception:
            traceback.print_exc()
            return 1

    TtsHandler.synthesizer = synth
    TtsHandler.stream_engine = KokoroStreamingEngineAdapter(synth)
    httpd = ThreadingHTTPServer((host, int(args.port)), TtsHandler)
    log.info("listening on http://%s:%s  model=%s", host, args.port, getattr(synth, "model_id", "unknown"))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("stopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
