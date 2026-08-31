"""kiki-audio — always-on ear.

PipeWire capture, Silero VAD, openWakeWord for „KIKI“, 400 ms pre-roll,
barge-in, explicit hotkey/click trigger. No Vosk, no invented microphone.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import time
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiki.audio.capture import MicrophoneCapture
from kiki.audio.vad import SpeechGate, build_speech_gate
from kiki.audio.wake import WakeSpotter, build_wake_spotter
from kiki.config.runtime import RuntimeConfig, load_runtime
from kiki.ipc.paths import turns_dir
from kiki.ipc.protocol import dumps, loads
from kiki.paths import bundled_data_dir

log = logging.getLogger("kiki-audio")


def _play_cue(path: Path) -> None:
    """Short earcon on wake. Not TTS — a click is honest if TTS is down."""
    if not path.is_file():
        return
    try:
        import subprocess

        subprocess.Popen(
            ["pw-cat", "-p", str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        log.debug("listen cue failed", exc_info=True)


class AudioDaemon:
    def __init__(
        self,
        cfg: RuntimeConfig,
        *,
        capture: MicrophoneCapture | None = None,
        vad: SpeechGate | None = None,
        wake: WakeSpotter | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.socket_path = cfg.socket("audio")
        self.turns = turns_dir(runtime=cfg.socket_dir)
        audio = cfg.audio
        self.sample_rate = audio.sample_rate
        self.frame_ms = audio.frame_ms
        self.frame_bytes = int(self.sample_rate * self.frame_ms / 1000) * 2
        self.pre_roll_frames = max(1, audio.pre_roll_ms // self.frame_ms)
        self.min_speech_ms = cfg.vad.min_speech_ms
        self.min_silence_ms = cfg.vad.min_silence_ms
        self.max_turn_ms = audio.max_turn_ms
        self.barge_in_ms = cfg.wake.barge_in_ms
        self.cooldown_s = cfg.wake.cooldown_ms / 1000.0
        self.listen_cue = audio.listen_cue

        self.capture = capture or MicrophoneCapture(self.sample_rate)
        self.vad = vad or build_speech_gate(cfg.vad.model, threshold=cfg.vad.threshold)
        self.wake = wake or build_wake_spotter(
            engine=cfg.wake.engine,
            model_name=cfg.wake.model_name,
            threshold=cfg.wake.threshold,
            consecutive_frames=cfg.wake.consecutive_frames,
        )
        self._on_event = on_event

        self.pre_roll: deque[bytes] = deque(maxlen=self.pre_roll_frames)
        self.turn_pcm = bytearray()
        self.listening_for_wake = True
        self.capturing = False
        self.turn_id = ""
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self.last_wake_at = 0.0
        self.tts_playing = False
        self.barge_ms = 0.0
        self.clients: list[asyncio.StreamWriter] = []
        self._running = True
        self._loop: asyncio.AbstractEventLoop | None = None

    def health(self) -> dict[str, Any]:
        issues: list[str] = []
        if not self.capture.ready:
            issues.append(self.capture.error or "mikrofon_offline")
        if not self.vad.ready:
            issues.append(self.vad.error or "vad_offline")
        elif self.vad.backend == "energy":
            issues.append("vad_energy_testmodus")
        if not self.wake.ready:
            issues.append(self.wake.error or "wake_offline")
        ready = self.capture.ready and self.vad.ready
        return {
            "event": "health",
            "status": "healthy" if ready and self.wake.ready and self.vad.backend == "silero_vad" else "degraded",
            "ready": ready,
            "capture_ready": self.capture.ready,
            "vad_ready": self.vad.ready,
            "vad_backend": self.vad.backend,
            "wake_ready": self.wake.ready,
            "wake_backend": self.wake.backend,
            "issues": issues,
        }

    def emit(self, payload: dict[str, Any]) -> None:
        payload.setdefault("timestamp", time.time())
        if self._on_event is not None:
            self._on_event(payload)
        loop = self._loop
        if loop is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop.call_soon_threadsafe(lambda: asyncio.create_task(self._broadcast(payload)))
            return
        asyncio.create_task(self._broadcast(payload))

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        line = dumps(payload)
        for writer in list(self.clients):
            try:
                writer.write(line)
                await writer.drain()
            except Exception:
                if writer in self.clients:
                    self.clients.remove(writer)

    def trigger_turn(self, *, source: str = "manual") -> str:
        now = time.time()
        self.last_wake_at = now
        self.listening_for_wake = False
        self.capturing = True
        self.turn_id = f"turn-{uuid.uuid4().hex[:12]}"
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self.turn_pcm = bytearray()
        for frame in self.pre_roll:
            self.turn_pcm.extend(frame)
        log.info("turn %s started (%s)", self.turn_id, source)
        if self.listen_cue:
            cue = bundled_data_dir() / "sounds" / "listen-start.wav"
            _play_cue(cue)
        self.emit({"event": "wake_detected", "turn_id": self.turn_id, "source": source})
        return self.turn_id

    def process_frame(self, pcm: bytes) -> None:
        if not pcm:
            return
        if len(pcm) < self.frame_bytes:
            pcm = pcm + b"\x00" * (self.frame_bytes - len(pcm))
        self.pre_roll.append(pcm[: self.frame_bytes])
        is_speech = self.vad.is_speech(pcm) if self.vad.ready else False
        now = time.time()

        if self.cfg.wake.barge_in_enabled and self.tts_playing and is_speech:
            self.barge_ms += self.frame_ms
            if self.barge_ms >= self.barge_in_ms:
                log.info("barge-in after %.0f ms of speech during TTS", self.barge_ms)
                self.emit({"event": "barge_in", "fadeout_ms": 35, "turn_id": self.turn_id})
                self.tts_playing = False
                self.barge_ms = 0.0
                self.trigger_turn(source="barge_in")
            return
        if not is_speech:
            self.barge_ms = 0.0

        if (
            self.listening_for_wake
            and self.wake.ready
            and self.cfg.wake.enabled
            and (now - self.last_wake_at) > self.cooldown_s
        ):
            if self.wake.feed(pcm):
                self.trigger_turn(source="wake")
                return

        if not self.capturing:
            return

        self.turn_pcm.extend(pcm)
        duration_ms = (len(self.turn_pcm) / (self.sample_rate * 2)) * 1000.0
        if is_speech:
            self.speech_ms += self.frame_ms
            self.silence_ms = 0.0
        else:
            self.silence_ms += self.frame_ms

        ended = False
        if self.speech_ms >= self.min_speech_ms and self.silence_ms >= self.min_silence_ms:
            ended = True
        elif duration_ms >= self.max_turn_ms:
            ended = True

        if ended:
            self._finish_turn()

    def _finish_turn(self) -> None:
        turn_id = self.turn_id
        pcm = bytes(self.turn_pcm)
        duration_ms = (len(pcm) / (self.sample_rate * 2)) * 1000.0
        self.capturing = False
        self.listening_for_wake = True
        path = self.turns / f"{turn_id}.pcm"
        try:
            path.write_bytes(pcm)
        except Exception as exc:
            log.error("could not write turn audio: %s", exc)
            self.emit(
                {
                    "event": "speech_ended",
                    "turn_id": turn_id,
                    "audio_path": "",
                    "error": str(exc),
                    "duration_ms": round(duration_ms, 1),
                }
            )
            return
        log.info("end of speech turn=%s duration=%.0f ms silence=%.0f ms", turn_id, duration_ms, self.silence_ms)
        self.emit(
            {
                "event": "speech_ended",
                "turn_id": turn_id,
                "audio_path": str(path),
                "duration_ms": round(duration_ms, 1),
                "preroll_ms": self.cfg.audio.pre_roll_ms,
            }
        )

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.clients.append(writer)
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
                cmd = str(msg.get("command") or msg.get("event") or "")
                if cmd in {"trigger_turn", "push_to_talk", "listen", "toggle_turn"}:
                    self.trigger_turn(source=str(msg.get("source") or "ipc"))
                elif cmd == "capture_one":
                    self.trigger_turn(source=str(msg.get("reason") or "capture_one"))
                elif cmd == "set_tts_state":
                    self.tts_playing = bool(msg.get("playing", False))
                    self.barge_ms = 0.0
                elif cmd in {"healthz", "ping", "health"}:
                    writer.write(dumps(self.health()))
                    await writer.drain()
        finally:
            if writer in self.clients:
                self.clients.remove(writer)

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        log.info("kiki-audio listening on %s", self.socket_path)
        log.info("health: %s", self.health())
        carry = bytearray()
        try:
            while self._running:
                raw = await asyncio.to_thread(self.capture.read, self.frame_ms)
                if not self.capture.ready:
                    await asyncio.sleep(0.5)
                    continue
                if raw:
                    carry.extend(raw)
                    while len(carry) >= self.frame_bytes:
                        frame = bytes(carry[: self.frame_bytes])
                        del carry[: self.frame_bytes]
                        self.process_frame(frame)
                else:
                    await asyncio.sleep(0.005)
        finally:
            server.close()
            await server.wait_closed()
            self.capture.close()

    def stop(self) -> None:
        self._running = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KIKI audio daemon")
    parser.add_argument("--socket", default="")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] kiki-audio: %(message)s",
    )
    cfg = load_runtime()
    if args.socket:
        from dataclasses import replace

        from kiki.ipc.paths import runtime_dir as _rd

        cfg = replace(cfg, socket_dir=_rd(Path(args.socket).parent))
    daemon = AudioDaemon(cfg)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, daemon.stop)
    try:
        loop.run_until_complete(daemon.run())
    except KeyboardInterrupt:
        daemon.stop()
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
