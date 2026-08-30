#!/usr/bin/env python3
"""Local faster-whisper STT HTTP service for KIKI.

Binds loopback only. Vosk stays the streaming ear inside the GTK app (wake
word, utterance boundaries); this service only transcribes a finished
utterance, so it needs no streaming and no torch: faster-whisper on
ctranslate2. The app POSTs the exact PCM passage it heard and gets text back.

    python kiki_stt_server.py --dummy   # no model, fixed text (wiring test)
    python kiki_stt_server.py           # Systran/faster-whisper-small, CUDA or CPU
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import threading
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger("kiki-stt")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18775
DEFAULT_MODEL = "Systran/faster-whisper-small"
DEFAULT_LANGUAGE = "de"
# 60 s of 16 kHz mono PCM16 ≈ 1.9 MiB; the cap catches runaway requests, not
# legitimate commands.
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_AUDIO_SECONDS = 60


class SttError(Exception):
    def __init__(self, message: str, status: int = 500) -> None:
        super().__init__(message)
        self.status = status


class DummyTranscriber:
    """Fixed German text so HTTP wiring can be tested without a model."""

    dummy = True
    ready = True
    model_id = "dummy-text"
    device = "cpu"

    def transcribe(self, wav_bytes: bytes) -> str:  # noqa: ARG002 - signature parity
        return "dies ist ein test des spracherkennungsdienstes"


class WhisperTranscriber:
    dummy = False
    ready = False

    def __init__(self, model_id: str, device: str, language: str) -> None:
        import numpy
        from faster_whisper import WhisperModel

        if device == "auto":
            try:
                import ctranslate2

                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        compute_type = "float16" if device.startswith("cuda") else "int8"
        self.model_id = model_id
        self.language = None if language.strip().lower() in {"", "auto"} else language.strip().lower()
        self.device = device
        self._numpy = numpy
        try:
            self._model = WhisperModel(model_id, device=device, compute_type=compute_type)
        except Exception:
            # A GPU without usable cuDNN/cuBLAS must not turn into a systemd
            # restart loop: degrade to CPU and say so.
            if device.startswith("cuda"):
                log.warning("CUDA-Start fehlgeschlagen — wechsle auf CPU.", exc_info=True)
                self.device = "cpu"
                compute_type = "int8"
                self._model = WhisperModel(model_id, device="cpu", compute_type=compute_type)
            else:
                raise
        self._lock = threading.Lock()
        self.ready = True

    def transcribe(self, wav_bytes: bytes) -> str:
        numpy = self._numpy
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            frames = wav.getnframes()
            if width != 2 or wav.getcomptype() != "NONE":
                raise SttError("Erwartet wird PCM16-WAV.", status=400)
            raw = wav.readframes(frames)
        if channels == 2:
            mono = numpy.frombuffer(raw, dtype=numpy.int16).reshape(-1, 2).mean(axis=1)
        else:
            mono = numpy.frombuffer(raw, dtype=numpy.int16)
        audio = mono.astype(numpy.float32) / 32768.0
        with self._lock:
            segments, _info = self._model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                # Every utterance stands alone; conditioning would let one bad
                # segment poison the next command.
                condition_on_previous_text=False,
            )
        return " ".join(segment.text.strip() for segment in segments).strip()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class SttHandler(BaseHTTPRequestHandler):
    transcriber: Any = None
    transcribe_lock = threading.Lock()
    server_version = "kiki-stt/0.1"

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
            engine = self.transcriber
            if engine is None:
                self._send_json(503, {"ok": False, "ready": False, "error": "starting"})
                return
            self._send_json(
                200,
                {
                    "ok": True,
                    "ready": bool(getattr(engine, "ready", True)),
                    "dummy": bool(getattr(engine, "dummy", False)),
                    "device": str(getattr(engine, "device", "")),
                    "model": str(getattr(engine, "model_id", "")),
                    "max_audio_seconds": MAX_AUDIO_SECONDS,
                },
            )
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def _validate_audio(self, wav_bytes: bytes) -> None:
        """Engine-independent audio gate: valid WAV, within the length cap."""
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
        except (EOFError, OSError, wave.Error) as exc:
            raise SttError(f"Ungültiges WAV: {exc}", status=400) from exc
        if rate <= 0:
            raise SttError("Ungültige Abtastrate.", status=400)
        if frames / float(rate) > MAX_AUDIO_SECONDS:
            raise SttError(
                f"Audio länger als {MAX_AUDIO_SECONDS} Sekunden.", status=413
            )

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/v1/transcribe", "/transcribe"}:
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        engine = self.transcriber
        if engine is None or not getattr(engine, "ready", False):
            self._send_json(
                503, {"ok": False, "ready": False, "error": "STT-Modell noch nicht geladen"}
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"ok": False, "error": "Content-Length fehlt"})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(413, {"ok": False, "error": "Anfrage zu groß"})
            return
        wav_bytes = self.rfile.read(length)
        if len(wav_bytes) != length:
            self._send_json(400, {"ok": False, "error": "Unvollständiger Anfragekörper"})
            return
        try:
            self._validate_audio(wav_bytes)
            with self.transcribe_lock:
                text = engine.transcribe(wav_bytes)
        except SttError as exc:
            self._send_json(exc.status, {"ok": False, "error": str(exc)})
            return
        except Exception:
            log.exception("transcribe failed")
            self._send_json(500, {"ok": False, "error": "Transkription fehlgeschlagen"})
            return
        self._send_json(200, {"ok": True, "text": text})


def _bind_host(host: str) -> str:
    text = host.strip() or DEFAULT_HOST
    if text not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("kiki-stt bindet nur Loopback (127.0.0.1 / ::1).")
    return "127.0.0.1" if text == "localhost" else text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KIKI faster-whisper local service")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="de, auto, …")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--dummy", action="store_true", help="no model: fixed text")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s kiki-stt: %(message)s",
    )
    host = _bind_host(args.host)
    if args.dummy:
        engine: Any = DummyTranscriber()
        log.info("dummy transcriber (no model)")
    else:
        try:
            engine = WhisperTranscriber(args.model, args.device, args.language)
        except ImportError as exc:
            log.error("faster-whisper fehlt (%s). scripts/setup-stt.sh oder --dummy.", exc)
            return 1
        except Exception:
            traceback.print_exc()
            return 1
    SttHandler.transcriber = engine
    httpd = ThreadingHTTPServer((host, int(args.port)), SttHandler)
    log.info(
        "listening on http://%s:%s  model=%s device=%s",
        host,
        args.port,
        getattr(engine, "model_id", "unknown"),
        getattr(engine, "device", ""),
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("stopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
