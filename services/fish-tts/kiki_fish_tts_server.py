#!/usr/bin/env python3
"""Local Fish Audio S2 Pro TTS service for KIKI.

Binds loopback only and speaks the same HTTP contract as the Qwen3-TTS
service (/health, /v1/synthesize → WAV), so the app only changes ports —
or nothing at all, since this service takes over port 18765.

The voice is a fixed reference clip: whatever audio/ref-text pair is passed
at startup is cloned for every utterance. KIKI ships a Serena-derived
reference; any clean 10–20 s clip can replace it.

    python kiki_fish_tts_server.py --dummy   # wiring test, no model
    python kiki_fish_tts_server.py           # S2 Pro on CUDA, resident model

Requires the fish-speech checkout with two local fixes: CPU-first weight
loading (a 16 GB card cannot hold the fp32 load peak) and a bounded
max_seq_len (the default 32768-token KV cache alone costs ~4.8 GB).
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
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger("kiki-fish-tts")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18765
DEFAULT_CHECKPOINT = "~/Modelle/test/s2-pro"
MAX_TEXT_CHARS = 2000
MAX_BODY_BYTES = 32 * 1024


class SynthError(Exception):
    def __init__(self, message: str, status: int = 500) -> None:
        super().__init__(message)
        self.status = status


def _silence_wav(seconds: float = 1.0, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


class DummySynthesizer:
    """Silence so HTTP wiring can be tested without the model."""

    dummy = True
    ready = True
    model_id = "dummy-silence"
    device = "cpu"
    voice = "dummy"

    def synthesize(self, text: str) -> bytes:
        del text
        return _silence_wav()


class FishSynthesizer:
    """Resident S2 Pro: model, codec and one fixed voice reference."""

    dummy = False
    ready = False

    def __init__(
        self,
        checkpoint: str,
        ref_tokens: str,
        ref_text: str,
        device: str,
        half: bool,
        quant: str = "int8",
    ) -> None:
        import numpy
        import torch
        from fish_speech.models.text2semantic import inference as t2s

        self._torch = torch
        self._t2s = t2s
        self.model_id = Path(checkpoint).name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        precision = torch.half if half else torch.bfloat16
        self.voice = Path(ref_tokens).stem

        self.model, self.decode_one_token = t2s.init_model(
            checkpoint, self.device, precision, compile=False
        )
        if quant != "none":
            # The S2 Pro composite is ~12 GB in fp16 — more than this GPU can
            # hold next to the STT ear and the LLM brain. Weight-only quants
            # leave activations in fp16; int4 trades a little timbre for ~3 GB.
            from torchao.quantization import int4_weight_only, int8_weight_only, quantize_

            config = int4_weight_only() if quant == "int4" else int8_weight_only()
            quantize_(self.model, config)
            log.info("weights quantized to %s weight-only", quant)
        self.codec = t2s.load_codec_model(
            str(Path(checkpoint) / "codec.pth"), self.device, torch.bfloat16
        )
        self.sample_rate = int(self.codec.sample_rate)
        self._numpy = numpy
        self._ref_text = ref_text
        self._ref_tokens = torch.from_numpy(numpy.load(ref_tokens))
        self._lock = threading.Lock()
        self.ready = True

    def synthesize(self, text: str) -> bytes:
        """Text in, WAV bytes out. Single model, serialized generation."""
        import soundfile as sf

        t2s = self._t2s
        prompt_tokens_list = [self._ref_tokens]
        with self._lock:
            generator = t2s.generate_long(
                model=self.model,
                device=self.device,
                decode_one_token=self.decode_one_token,
                text=text,
                num_samples=1,
                max_new_tokens=1024,
                top_p=0.7,
                top_k=30,
                temperature=0.7,
                compile=False,
                iterative_prompt=True,
                chunk_length=200,
                prompt_text=[self._ref_text],
                prompt_tokens=prompt_tokens_list,
            )
            codes = []
            for response in generator:
                if response.action == "sample":
                    codes.append(response.codes)
                elif response.action == "next":
                    break
        if not codes:
            raise SynthError("S2 Pro hat keine Sprache erzeugt.")
        merged = self._torch.cat(codes, dim=1)
        audio = t2s.decode_to_audio(merged.to(self.device), self.codec)
        buf = io.BytesIO()
        sf.write(buf, audio.cpu().float().numpy(), self.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class TtsHandler(BaseHTTPRequestHandler):
    synthesizer: Any = None
    synth_lock = threading.Lock()
    server_version = "kiki-fish-tts/0.1"

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
            engine = self.synthesizer
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
                    "voice": str(getattr(engine, "voice", "")),
                },
            )
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/v1/synthesize", "/synthesize"}:
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        engine = self.synthesizer
        if engine is None or not getattr(engine, "ready", False):
            self._send_json(
                503, {"ok": False, "ready": False, "error": "TTS-Modell noch nicht geladen"}
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
        try:
            with self.synth_lock:
                wav = engine.synthesize(text)
        except SynthError as exc:
            self._send_json(exc.status, {"ok": False, "error": str(exc)})
            return
        except Exception:
            log.exception("synthesize failed")
            self._send_json(500, {"ok": False, "error": "Synthese fehlgeschlagen"})
            return
        if not wav:
            self._send_json(500, {"ok": False, "error": "leere WAV"})
            return
        self._send(200, wav, "audio/wav")


def _bind_host(host: str) -> str:
    text = host.strip() or DEFAULT_HOST
    if text not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("kiki-fish-tts bindet nur Loopback (127.0.0.1 / ::1).")
    return "127.0.0.1" if text == "localhost" else text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KIKI Fish S2 Pro local service")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ref-tokens", required=True, help="VQ- .npy der Referenzstimme")
    parser.add_argument("--ref-text", required=True, help="Transkript der Referenzstimme")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--half/--no-half", dest="half", default=True)
    parser.add_argument(
        "--quant",
        choices=("int8", "int4", "none"),
        default="int8",
        help="Gewichtsquantisierung; ohne passt das Modell nicht neben STT+LLM",
    )
    parser.add_argument("--dummy", action="store_true", help="no model: silence")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s kiki-fish-tts: %(message)s",
    )
    host = _bind_host(args.host)
    if args.dummy:
        engine: Any = DummySynthesizer()
        log.info("dummy synthesizer (no model)")
    else:
        # The fish-speech checkout must be importable; it carries the two
        # local fixes (CPU-first load, bounded KV cache) the stock repo lacks.
        checkpoint = Path(args.checkpoint).expanduser()
        repo_root = Path(__file__).resolve().parent.parent.parent / "fish-speech"
        for candidate in (repo_root, Path("/home/martin/Modelle/test/fish-speech")):
            if candidate.is_dir():
                sys.path.insert(0, str(candidate))
                break
        ref_text_path = Path(args.ref_text).expanduser()
        try:
            engine = FishSynthesizer(
                str(checkpoint),
                str(Path(args.ref_tokens).expanduser()),
                ref_text_path.read_text(encoding="utf-8").strip(),
                args.device,
                args.half,
                args.quant,
            )
        except ImportError as exc:
            log.error("fish-speech Stack fehlt (%s).", exc)
            return 1
        except Exception:
            traceback.print_exc()
            return 1
    TtsHandler.synthesizer = engine
    httpd = ThreadingHTTPServer((host, int(args.port)), TtsHandler)
    log.info(
        "listening on http://%s:%s model=%s voice=%s",
        host,
        args.port,
        getattr(engine, "model_id", "unknown"),
        getattr(engine, "voice", "?"),
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
