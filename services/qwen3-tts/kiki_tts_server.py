#!/usr/bin/env python3
"""Local Qwen3-TTS HTTP service for KIKI.

Binds loopback only. The GTK app never loads PyTorch — it POSTs text here
and plays the WAV through PipeWire.

    python kiki_tts_server.py --dummy          # no GPU, short tone (wiring test)
    python kiki_tts_server.py                  # Qwen3-TTS-12Hz-0.6B-CustomVoice on CUDA
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import struct
import sys
import threading
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger("kiki-tts")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18765
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_SPEAKER = "Serena"
DEFAULT_LANGUAGE = "German"
MAX_TEXT_CHARS = 4000
MAX_BODY_BYTES = 256 * 1024

CUSTOM_VOICE_SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]
CUSTOM_VOICE_LANGUAGES = [
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
]


class SynthError(Exception):
    def __init__(self, message: str, status: int = 500) -> None:
        super().__init__(message)
        self.status = status


class DummySynthesizer:
    """Short audible tone so PipeWire wiring can be tested without CUDA."""

    dummy = True
    ready = True
    model_id = "dummy-tone"
    device = "cpu"
    speakers = list(CUSTOM_VOICE_SPEAKERS)
    languages = list(CUSTOM_VOICE_LANGUAGES)

    def synthesize(self, text: str, *, language: str, speaker: str) -> bytes:
        del language, speaker
        duration = min(1.6, 0.35 + min(len(text), 80) / 80.0)
        return _tone_wav(duration_s=duration, freq=220.0)


class QwenSynthesizer:
    dummy = False
    ready = False

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.device = device
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        log.info("loading %s on %s dtype=%s attn=sdpa", model_id, device, dtype)
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=device,
                dtype=dtype,
                attn_implementation="sdpa",
            )
        except Exception:
            log.warning("sdpa load failed, retrying default attention", exc_info=True)
            self.model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=device,
                dtype=dtype,
            )
        self.ready = True
        try:
            speakers = list(self.model.get_supported_speakers() or [])
        except Exception:
            speakers = list(CUSTOM_VOICE_SPEAKERS)
        self.speakers = [str(s) for s in speakers] or list(CUSTOM_VOICE_SPEAKERS)
        try:
            languages = list(self.model.get_supported_languages() or [])
        except Exception:
            languages = list(CUSTOM_VOICE_LANGUAGES)
        self.languages = [str(s) for s in languages] or list(CUSTOM_VOICE_LANGUAGES)
        log.info("ready speakers=%s languages=%s", self.speakers, self.languages)

    @staticmethod
    def _resolve(wanted: str, allowed: list[str], fallback: str) -> str:
        """Match a requested name against what the model reports, ignoring case.

        The model lists its voices lower-case ("serena"), while KIKI's config and
        UI use the documented capitalised spelling ("Serena"). A plain `in` test
        therefore missed every time and silently fell through to speakers[0] —
        "aiden", a male voice — with the language left at an equally invalid
        value. Matching case-insensitively and returning the model's own
        spelling keeps both sides working.
        """
        text = str(wanted or "").strip()
        by_lower = {str(name).lower(): str(name) for name in allowed}
        if text.lower() in by_lower:
            return by_lower[text.lower()]
        if str(fallback).lower() in by_lower:
            return by_lower[str(fallback).lower()]
        return str(allowed[0]) if allowed else text

    def synthesize(self, text: str, *, language: str, speaker: str) -> bytes:
        import soundfile as sf

        speaker = self._resolve(speaker, self.speakers, DEFAULT_SPEAKER)
        language = self._resolve(language, self.languages, DEFAULT_LANGUAGE)
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
        )
        audio = wavs[0]
        buf = io.BytesIO()
        sf.write(buf, audio, int(sr), format="WAV", subtype="PCM_16")
        return buf.getvalue()


def _tone_wav(*, duration_s: float, freq: float, rate: int = 24000) -> bytes:
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


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class TtsHandler(BaseHTTPRequestHandler):
    synthesizer: Any = None
    synth_lock = threading.Lock()
    server_version = "kiki-tts/0.1"

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

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
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
                    "model": str(getattr(synth, "model_id", getattr(synth, "model", ""))),
                    "speakers": list(getattr(synth, "speakers", CUSTOM_VOICE_SPEAKERS)),
                    "languages": list(getattr(synth, "languages", CUSTOM_VOICE_LANGUAGES)),
                },
            )
            return
        if path in {"/v1/speakers", "/speakers"}:
            synth = self.synthesizer
            speakers = list(getattr(synth, "speakers", CUSTOM_VOICE_SPEAKERS)) if synth else CUSTOM_VOICE_SPEAKERS
            self._send_json(200, {"speakers": speakers})
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/v1/synthesize", "/synthesize"}:
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        synth = self.synthesizer
        if synth is None or not getattr(synth, "ready", False):
            self._send_json(503, {"ok": False, "ready": False, "error": "TTS-Modell noch nicht geladen"})
            return
        length_raw = self.headers.get("Content-Length", "0")
        try:
            length = int(length_raw)
        except ValueError:
            self._send_json(400, {"ok": False, "error": "Content-Length fehlt"})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(413, {"ok": False, "error": "Anfrage zu groß"})
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"ok": False, "error": "JSON ungültig"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"ok": False, "error": "JSON-Objekt erwartet"})
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
        raise SystemExit("kiki-tts bindet nur Loopback (127.0.0.1 / ::1).")
    return "127.0.0.1" if text == "localhost" else text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KIKI Qwen3-TTS local service")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="cuda:0, cpu, or auto")
    parser.add_argument("--dummy", action="store_true", help="no GPU: return a short tone")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s kiki-tts: %(message)s",
    )
    host = _bind_host(args.host)
    if args.dummy:
        synth: Any = DummySynthesizer()
        log.info("dummy synthesizer (no CUDA)")
    else:
        try:
            synth = QwenSynthesizer(args.model, args.device)
        except ImportError as exc:
            log.error("qwen-tts / torch fehlen (%s). scripts/setup-tts.sh oder --dummy.", exc)
            return 1
        except Exception:
            traceback.print_exc()
            return 1
    TtsHandler.synthesizer = synth
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
