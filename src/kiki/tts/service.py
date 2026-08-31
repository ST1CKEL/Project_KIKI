"""kiki-tts — warm engine, phrase cache, persistent PipeWire, barge-in cancel."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

from kiki.config.runtime import RuntimeConfig, load_runtime
from kiki.ipc.protocol import dumps, loads
from kiki.tts.engine import STANDARD_PHRASES, TtsEngine, build_engine
from kiki.tts.normalizer import GermanTextNormalizer
from kiki.tts.playback import PipeWirePcmPlayback

log = logging.getLogger("kiki-tts")


class TtsService:
    def __init__(
        self,
        cfg: RuntimeConfig,
        *,
        engine: TtsEngine | None = None,
        secondary: TtsEngine | None = None,
        playback: PipeWirePcmPlayback | None = None,
    ) -> None:
        self.cfg = cfg
        self.socket_path = cfg.socket("tts")
        self.normalizer = GermanTextNormalizer()
        self.primary: TtsEngine | None = engine
        self.secondary: TtsEngine | None = secondary
        self.active: TtsEngine | None = None
        self.degraded = False
        self.degraded_reason = ""
        self._phrase_cache: dict[str, bytes] = {}
        self._clients: list[asyncio.StreamWriter] = []
        self._running = True
        self._loop: asyncio.AbstractEventLoop | None = None
        self._select_engine()
        rate = self.active.sample_rate if self.active is not None else cfg.tts.sample_rate
        self._stopped: asyncio.Event | None = None
        self.playback = playback or PipeWirePcmPlayback(
            sample_rate=rate,
            on_audio_started=self._on_audio_started,
            on_audio_finished=self._on_audio_finished,
        )
        if cfg.tts.cache_phrases and self.active is not None:
            self._pre_render()

    def _select_engine(self) -> None:
        tts = self.cfg.tts
        if tts.allow_espeak:
            log.error("allow_espeak is set — ignored. espeak-ng is never a KIKI voice.")
        if self.primary is None:
            try:
                self.primary = build_engine(tts.primary_engine, tts.primary_voice)
            except Exception as exc:
                log.error("primary engine failed to construct: %s", exc)
                self.primary = None
        if self.primary is not None and self.primary.ready and self.primary.german_verified:
            self.active = self.primary
            self.degraded = False
            return
        primary_error = ""
        if self.primary is not None:
            primary_error = self.primary.error or "primary not german-verified"
        else:
            primary_error = "primary engine missing"
        if self.secondary is None and tts.secondary_engine:
            try:
                self.secondary = build_engine(tts.secondary_engine, tts.secondary_voice)
            except Exception as exc:
                log.error("secondary engine failed: %s", exc)
                self.secondary = None
        if self.secondary is not None and self.secondary.ready and self.secondary.german_verified:
            self.active = self.secondary
            self.degraded = True
            self.degraded_reason = (
                f"Hauptstimme ({tts.primary_engine}/{tts.primary_voice}) ausgefallen: "
                f"{primary_error}. Ersatzstimme {tts.secondary_engine}/{tts.secondary_voice}."
            )
            log.error("%s", self.degraded_reason)
            return
        self.active = None
        self.degraded = True
        secondary_error = self.secondary.error if self.secondary is not None else "nicht konfiguriert"
        self.degraded_reason = (
            f"Keine deutsche Stimme bereit. Primary: {primary_error}. "
            f"Secondary: {secondary_error}. Ich schweige, statt espeak-ng zu spielen."
        )
        log.error("%s", self.degraded_reason)

    def _pre_render(self) -> None:
        assert self.active is not None
        for key, text in STANDARD_PHRASES.items():
            try:
                pcm = self.active.synthesize_pcm(self.normalizer.normalize(text))
            except Exception as exc:
                log.warning("phrase cache %s failed: %s", key, exc)
                continue
            if pcm:
                self._phrase_cache[key] = pcm
                self._phrase_cache[text] = pcm
        log.info("cached %d instant phrases", len(self._phrase_cache))

    def _on_audio_started(self, turn_id: str) -> None:
        self._broadcast_threadsafe(
            {"event": "audio_started", "turn_id": turn_id, "timestamp": time.time()}
        )

    def _on_audio_finished(self, turn_id: str) -> None:
        self._broadcast_threadsafe(
            {"event": "audio_finished", "turn_id": turn_id, "timestamp": time.time()}
        )

    def _broadcast_threadsafe(self, payload: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(lambda: asyncio.create_task(self._broadcast(payload)))

    def health(self) -> dict[str, Any]:
        active = self.active
        status = "healthy"
        if active is None or not active.ready:
            status = "failed"
        elif self.degraded:
            status = "degraded"
        return {
            "event": "health",
            "status": status,
            "ready": bool(active and active.ready and self.playback.ready),
            "degraded": self.degraded,
            "engine": active.name if active else "",
            "voice": active.voice if active else "",
            "german_verified": bool(active and active.german_verified),
            "playback_ready": self.playback.ready,
            "underruns": self.playback.underrun_count,
            "error": self.degraded_reason or (active.error if active else "no engine"),
        }

    async def _reply(self, writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        try:
            writer.write(dumps(payload))
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError, ConnectionError):
            return

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        line = dumps(payload)
        for writer in list(self._clients):
            try:
                writer.write(line)
                await writer.drain()
            except Exception:
                if writer in self._clients:
                    self._clients.remove(writer)

    def _engine_rate(self) -> int:
        if self.active is not None and getattr(self.active, "sample_rate", 0):
            return int(self.active.sample_rate)
        return int(self.playback.sample_rate)

    def _synthesize(self, text: str) -> tuple[bytes, float]:
        t0 = time.perf_counter()
        if text in self._phrase_cache:
            return self._phrase_cache[text], (time.perf_counter() - t0) * 1000.0
        if self.active is None:
            return b"", 0.0
        pcm = self.active.synthesize_pcm(text)
        return pcm, (time.perf_counter() - t0) * 1000.0

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._clients.append(writer)
        await self._reply(writer, self.health())
        try:
            while self._running and not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = loads(line)
                except Exception:
                    continue
                action = str(msg.get("command") or msg.get("action") or "")
                turn_id = str(msg.get("turn_id") or "turn-anon")
                if action in {"healthz", "ping", "health"}:
                    await self._reply(writer, self.health())
                elif action == "speak_phrase":
                    key = str(msg.get("phrase") or "ack")
                    pcm = self._phrase_cache.get(key, b"")
                    if not pcm and self.active is not None:
                        spoken = STANDARD_PHRASES.get(key, key)
                        pcm, _ = await asyncio.to_thread(
                            self._synthesize, self.normalizer.normalize(spoken)
                        )
                    if pcm:
                        self.playback.enqueue_pcm(turn_id, pcm, sample_rate=self._engine_rate())
                    await self._reply(
                        writer,
                        {"event": "phrase_queued", "turn_id": turn_id, "phrase": key, "bytes": len(pcm)},
                    )
                elif action == "speak_chunk":
                    raw = str(msg.get("text") or "")
                    norm = self.normalizer.normalize(raw)
                    pcm, latency_ms = await asyncio.to_thread(self._synthesize, norm)
                    if pcm:
                        self.playback.enqueue_pcm(turn_id, pcm, sample_rate=self._engine_rate())
                    await self._reply(
                        writer,
                        {
                            "event": "chunk_synthesized",
                            "turn_id": turn_id,
                            "latency_ms": round(latency_ms, 1),
                            "bytes": len(pcm),
                        },
                    )
                elif action == "end_turn":
                    self.playback.mark_turn_complete()
                elif action == "cancel":
                    self.playback.cancel_turn(turn_id)
                    await self._reply(writer, {"event": "cancelled", "turn_id": turn_id})
        except (ConnectionResetError, BrokenPipeError, ConnectionError):
            return
        finally:
            if writer in self._clients:
                self._clients.remove(writer)

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stopped = asyncio.Event()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        log.info("kiki-tts listening on %s", self.socket_path)
        log.info("health: %s", self.health())
        try:
            await self._stopped.wait()
        finally:
            server.close()
            await server.wait_closed()
            self.playback.close()

    def stop(self) -> None:
        self._running = False
        self.playback.close()
        stopped = self._stopped
        if stopped is not None and not stopped.is_set():
            stopped.set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KIKI neural TTS")
    parser.add_argument("--socket", default="")
    parser.add_argument("--voice", default="")
    parser.add_argument("--engine", default="")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] kiki-tts: %(message)s",
    )
    cfg = load_runtime()
    if args.socket:
        from dataclasses import replace

        from kiki.ipc.paths import runtime_dir as _rd

        cfg = replace(cfg, socket_dir=_rd(Path(args.socket).parent))
    if args.voice or args.engine:
        from dataclasses import replace

        tts = replace(
            cfg.tts,
            primary_voice=args.voice or cfg.tts.primary_voice,
            primary_engine=args.engine or cfg.tts.primary_engine,
        )
        cfg = replace(cfg, tts=tts)
    service = TtsService(cfg)
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
    return 0 if service.active is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
