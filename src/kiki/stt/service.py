"""kiki-stt — warm faster-whisper on GPU.

Law 1: if CUDA/FP16 cannot load the configured model, the service is *not
ready*. There is no Vosk path and no silent drop to a smaller model.
CPU is allowed only when runtime.toml sets `stt.device = "cpu"` or
`stt.allow_cpu = true`, and that is advertised as degraded.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiki.config.runtime import RuntimeConfig, load_runtime
from kiki.ipc.protocol import dumps, loads

log = logging.getLogger("kiki-stt")

TARGET_SAMPLE_RATE = 16000


class SttNotReady(RuntimeError):
    """The model is not loaded. Callers must surface this, not invent text."""


@dataclass
class SttResult:
    text: str
    confidence: float
    no_speech_prob: float
    duration_s: float
    latency_ms: float
    turn_id: str
    device: str
    model: str


class WhisperEngine:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg.stt
        self.model_name = self.cfg.model_name
        self.requested_device = self.cfg.device
        self.compute_type = self.cfg.compute_type
        self.language = self.cfg.language
        self._model: Any = None
        self.ready = False
        self.degraded = False
        self.error = ""
        self.device = self.requested_device

    def load(self) -> None:
        try:
            import ctranslate2
            from faster_whisper import WhisperModel
        except Exception as exc:
            self.error = f"faster-whisper/ctranslate2 fehlt: {exc}"
            log.error("%s", self.error)
            return

        cuda_count = 0
        try:
            cuda_count = int(ctranslate2.get_cuda_device_count())
        except Exception:
            cuda_count = 0

        device = self.requested_device
        compute = self.compute_type
        if device in {"auto", "cuda"}:
            if cuda_count <= 0:
                if not self.cfg.allow_cpu and device != "cpu":
                    self.error = (
                        "Keine CUDA-Geräte für faster-whisper. "
                        "Ich wechsle nicht still auf CPU. "
                        "Setze stt.device=cpu nur bewusst, dann ist der Dienst degradiert."
                    )
                    log.error("%s", self.error)
                    return
                device = "cpu"
                self.degraded = True
                log.warning("STT running on CPU because stt.allow_cpu/device=cpu is set")
            else:
                device = "cuda"
        if device == "cpu" and compute == "float16":
            compute = "int8"
            self.degraded = True

        log.info("loading %s device=%s compute=%s", self.model_name, device, compute)
        t0 = time.perf_counter()
        try:
            self._model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute,
            )
        except Exception as exc:
            self.error = (
                f"Modell {self.model_name} ließ sich nicht laden ({exc}). "
                "Kein stiller Wechsel auf ein kleineres Modell."
            )
            log.exception("%s", self.error)
            return
        self.device = device
        self.compute_type = compute
        self.ready = True
        log.info("whisper ready in %.2f s", time.perf_counter() - t0)

    def transcribe(self, turn_id: str, pcm: bytes) -> SttResult:
        if not self.ready or self._model is None:
            raise SttNotReady(self.error or "STT nicht bereit")
        if not pcm:
            return SttResult(
                text="",
                confidence=0.0,
                no_speech_prob=1.0,
                duration_s=0.0,
                latency_ms=0.0,
                turn_id=turn_id,
                device=self.device,
                model=self.model_name,
            )
        import numpy as np

        t0 = time.perf_counter()
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(audio) / TARGET_SAMPLE_RATE
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.cfg.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        texts: list[str] = []
        no_speech: list[float] = []
        for segment in segments:
            piece = str(getattr(segment, "text", "") or "").strip()
            if piece:
                texts.append(piece)
            no_speech.append(float(getattr(segment, "no_speech_prob", 0.0) or 0.0))
        avg = sum(no_speech) / len(no_speech) if no_speech else 0.0
        return SttResult(
            text=" ".join(texts).strip(),
            confidence=float(1.0 - avg),
            no_speech_prob=float(avg),
            duration_s=round(duration, 2),
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            turn_id=turn_id,
            device=self.device,
            model=self.model_name,
        )


def _read_pcm(path: Path) -> bytes:
    data = path.read_bytes()
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        import io
        import wave

        with wave.open(io.BytesIO(data), "rb") as wav:
            return wav.readframes(wav.getnframes())
    return data


class SttService:
    def __init__(self, cfg: RuntimeConfig, *, engine: WhisperEngine | None = None) -> None:
        self.cfg = cfg
        self.socket_path = cfg.socket("stt")
        self.engine = engine or WhisperEngine(cfg)
        self._clients: list[asyncio.StreamWriter] = []
        self._running = True

    def health(self) -> dict[str, Any]:
        status = "healthy"
        if not self.engine.ready:
            status = "failed"
        elif self.engine.degraded:
            status = "degraded"
        return {
            "event": "health",
            "status": status,
            "ready": self.engine.ready,
            "degraded": self.engine.degraded,
            "model": self.engine.model_name,
            "device": self.engine.device,
            "compute_type": self.engine.compute_type,
            "error": self.engine.error,
        }

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._clients.append(writer)
        writer.write(dumps(self.health()))
        await writer.drain()
        try:
            while self._running and not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = loads(line)
                except Exception:
                    continue
                event = str(msg.get("event") or msg.get("command") or "")
                if event in {"healthz", "ping", "health"}:
                    writer.write(dumps(self.health()))
                    await writer.drain()
                    continue
                if event != "transcribe_file":
                    continue
                turn_id = str(msg.get("turn_id") or "")
                path = Path(str(msg.get("path") or ""))
                pcm = b""
                read_error = ""
                if path.is_file():
                    try:
                        pcm = _read_pcm(path)
                        path.unlink(missing_ok=True)
                    except Exception as exc:
                        read_error = str(exc)
                try:
                    result = await asyncio.to_thread(self.engine.transcribe, turn_id, pcm)
                except SttNotReady as exc:
                    writer.write(
                        dumps(
                            {
                                "event": "error",
                                "turn_id": turn_id,
                                "error": str(exc),
                                "code": "stt_not_ready",
                            }
                        )
                    )
                    await writer.drain()
                    continue
                payload = {
                    "event": "final",
                    "turn_id": result.turn_id,
                    "text": result.text,
                    "confidence": result.confidence,
                    "no_speech_prob": result.no_speech_prob,
                    "duration_s": result.duration_s,
                    "latency_ms": result.latency_ms,
                    "device": result.device,
                    "model": result.model,
                    "timestamp": time.time(),
                }
                if read_error:
                    payload["warning"] = read_error
                log.info(
                    "turn %s transcribed in %.1f ms: %r",
                    turn_id,
                    result.latency_ms,
                    result.text,
                )
                writer.write(dumps(payload))
                await writer.drain()
        finally:
            if writer in self._clients:
                self._clients.remove(writer)

    async def run(self) -> None:
        if not self.engine.ready:
            self.engine.load()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        log.info("kiki-stt listening on %s ready=%s", self.socket_path, self.engine.ready)
        try:
            while self._running:
                await asyncio.sleep(1.0)
        finally:
            server.close()
            await server.wait_closed()

    def stop(self) -> None:
        self._running = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KIKI faster-whisper STT")
    parser.add_argument("--socket", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--compute-type", default="")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] kiki-stt: %(message)s",
    )
    cfg = load_runtime()
    if args.socket:
        from dataclasses import replace

        from kiki.ipc.paths import runtime_dir as _rd

        cfg = replace(cfg, socket_dir=_rd(Path(args.socket).parent))
    if args.model or args.device or args.compute_type:
        from dataclasses import replace

        stt = replace(
            cfg.stt,
            model_name=args.model or cfg.stt.model_name,
            device=(args.device or cfg.stt.device),
            compute_type=args.compute_type or cfg.stt.compute_type,
        )
        cfg = replace(cfg, stt=stt)
    service = SttService(cfg)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, service.stop)
    try:
        loop.run_until_complete(service.run())
    except KeyboardInterrupt:
        service.stop()
    finally:
        loop.close()
    return 0 if service.engine.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
