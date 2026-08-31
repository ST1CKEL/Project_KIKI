"""kiki-orchestrator — the nervous system.

Connects audio, STT, TTS, the local LLM, ToolGateway, and the pet. Measures
every turn. Speaks faults instead of degrading in silence.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import time
from typing import Any

import httpx

from kiki.ai.factory import create_provider
from kiki.assistant.adapter import ProviderStepAdapter
from kiki.assistant.runner import AssistantRunner
from kiki.config.runtime import RuntimeConfig, load_runtime
from kiki.config.settings import load_settings
from kiki.harness.models import RunStatus
from kiki.ipc.client import JsonUnixClient
from kiki.ipc.protocol import dumps, loads
from kiki.orchestrator.confirm import confirmation_prompt, parse_spoken_verdict
from kiki.orchestrator.desktop_stack import DesktopStack, build_desktop_stack
from kiki.orchestrator.gpu import GpuResourceManager
from kiki.orchestrator.health import CharacterState, HealthState, SubsystemHealth
from kiki.orchestrator.prompt import voice_system_prompt
from kiki.orchestrator.telemetry import TelemetryLog, TurnTelemetry
from kiki.orchestrator.vision import VisionAgent
from kiki.paths import state_dir
from kiki.storage.secrets import create_secret_store
from kiki.tools.direct_actions import parse_direct_control
from kiki.tts.chunker import StreamingTtsChunker

log = logging.getLogger("kiki-orchestrator")


class Orchestrator:
    def __init__(self, cfg: RuntimeConfig, *, stack: DesktopStack | None = None) -> None:
        self.cfg = cfg
        self.health = SubsystemHealth()
        self.state = CharacterState.IDLE
        self.gpu = GpuResourceManager(
            cfg.hardware.gpu_device_id, cfg.hardware.vram_safety_margin_mb
        )
        self.telemetry = TelemetryLog()
        self.current: TurnTelemetry | None = None
        self._running = True
        self._panic = False
        self._ui_clients: list[asyncio.StreamWriter] = []
        self._pending_verdict: asyncio.Future[str] | None = None
        self._announced_faults: set[str] = set()
        # TODO: arm one follow-up capture when TTS reports playback_end, with a
        # short silence timeout. Doing it immediately after LLM flush would
        # record the assistant's own voice or 8 s of quiet.

        self.audio = JsonUnixClient(cfg.socket("audio"), name="audio")
        self.stt = JsonUnixClient(cfg.socket("stt"), name="stt")
        self.tts = JsonUnixClient(cfg.socket("tts"), name="tts")

        settings = load_settings()
        if cfg.policy.autonomy:
            settings.tools.autonomy = cfg.policy.autonomy
        self.settings = settings
        self.stack = stack or build_desktop_stack(
            settings, vision_handler=self._vision_tool
        )
        self.vision = VisionAgent(
            llm_url=cfg.llm.base_url,
            vision_model=cfg.llm.vision_model,
            can_allocate=lambda: self.gpu.can_allocate_vram(cfg.hardware.vision_vram_mb),
            speak=self.speak_text,
        )

    # -- state / UI ----------------------------------------------------------

    def set_state(self, state: CharacterState) -> None:
        if state is self.state:
            return
        log.info("state %s -> %s", self.state, state)
        self.state = state
        asyncio.create_task(self._broadcast_ui())

    def overall(self) -> HealthState:
        return self.health.overall()

    def ui_payload(self) -> dict[str, Any]:
        last = self.telemetry.last()
        return {
            "event": "state_change",
            "state": self.state.value,
            "health": self.overall().value,
            "issues": self.health.issues(),
            "panic": self._panic,
            "turn": last.as_dict() if last else None,
            "timestamp": time.time(),
        }

    async def _broadcast_ui(self) -> None:
        line = dumps(self.ui_payload())
        for writer in list(self._ui_clients):
            try:
                writer.write(line)
                await writer.drain()
            except Exception:
                if writer in self._ui_clients:
                    self._ui_clients.remove(writer)

    async def speak_phrase(self, phrase: str, turn_id: str = "sys") -> None:
        await self.tts.send({"command": "speak_phrase", "phrase": phrase, "turn_id": turn_id})

    async def speak_text(self, text: str, turn_id: str = "sys") -> None:
        if not text.strip():
            return
        await self.tts.send({"command": "speak_chunk", "turn_id": turn_id, "text": text})
        await self.tts.send({"command": "end_turn", "turn_id": turn_id})

    # -- audio / stt / tts events --------------------------------------------

    async def on_audio(self, msg: dict[str, Any]) -> None:
        event = str(msg.get("event") or "")
        if event == "health":
            self.health.audio = msg
            await self._maybe_announce_fault()
            return
        if event == "wake_detected":
            turn = TurnTelemetry(turn_id=str(msg.get("turn_id") or ""))
            turn.mark_wake(str(msg.get("source") or "wake"))
            self.current = turn
            self.set_state(CharacterState.LISTENING)
            await self.speak_phrase("ack", turn.turn_id)
            turn.mark_ack()
            await self.audio.send({"command": "set_tts_state", "playing": False})
            return
        if event == "speech_ended":
            turn_id = str(msg.get("turn_id") or "")
            path = str(msg.get("path") or msg.get("audio_path") or "")
            if self.current and self.current.turn_id == turn_id:
                self.current.t_eos = time.perf_counter()
            self.set_state(CharacterState.THINKING)
            if not self.stt.connected:
                await self._fail_turn("Meine Spracherkennung ist ausgefallen.")
                return
            ok = await self.stt.send(
                {"event": "transcribe_file", "turn_id": turn_id, "path": path}
            )
            if not ok:
                await self._fail_turn("Meine Spracherkennung ist ausgefallen.")
            return
        if event == "barge_in":
            if self.current:
                await self.tts.send({"command": "cancel", "turn_id": self.current.turn_id})
            self.set_state(CharacterState.LISTENING)

    async def on_stt(self, msg: dict[str, Any]) -> None:
        event = str(msg.get("event") or "")
        if event == "health":
            self.health.stt = msg
            await self._maybe_announce_fault()
            return
        if event == "error":
            await self._fail_turn(str(msg.get("error") or "Spracherkennung fehlgeschlagen."))
            return
        if event != "final":
            return
        text = str(msg.get("text") or "").strip()
        if self.current:
            self.current.t_stt_final = time.perf_counter()
            self.current.transcript = text
        if self._pending_verdict is not None and not self._pending_verdict.done():
            self._pending_verdict.set_result(text)
            return
        if not text:
            log.info("turn %s empty transcript", msg.get("turn_id"))
            await self.speak_text("Das habe ich nicht verstanden.", str(msg.get("turn_id") or "sys"))
            self.set_state(CharacterState.IDLE)
            return
        log.info("heard: %s", text)
        await self.process_turn(text)

    async def on_tts(self, msg: dict[str, Any]) -> None:
        event = str(msg.get("event") or "")
        if event == "health":
            self.health.tts = msg
            await self._maybe_announce_fault()
            return
        if event == "audio_started":
            if self.current:
                self.current.t_playback_start = self.current.t_playback_start or time.perf_counter()
                self.current.t_tts_first_pcm = self.current.t_tts_first_pcm or time.perf_counter()
                log.info("%s", self.current.summary())
            self.set_state(CharacterState.SPEAKING)
            await self.audio.send({"command": "set_tts_state", "playing": True})
            return
        if event == "audio_finished":
            await self.audio.send({"command": "set_tts_state", "playing": False})
            if self.state is CharacterState.SPEAKING:
                self.set_state(CharacterState.IDLE)

    async def _fail_turn(self, spoken: str) -> None:
        if self.current:
            self.current.error = spoken
            self.telemetry.add(self.current)
        self.set_state(CharacterState.ERROR)
        await self.speak_text(spoken, self.current.turn_id if self.current else "sys")
        await asyncio.sleep(0.4)
        self.set_state(CharacterState.IDLE)

    async def _maybe_announce_fault(self) -> None:
        spoken = self.health.spoken_fault()
        key = self.overall().value
        if not spoken:
            self._announced_faults.clear()
            await self._broadcast_ui()
            return
        if key in self._announced_faults:
            await self._broadcast_ui()
            return
        self._announced_faults.add(key)
        log.error("fault %s: %s", key, spoken)
        # Wake-opt-out is expected: do not freeze the pet on the warning pose
        # or talk over the user about a missing ONNX.
        if key == HealthState.DEGRADED_WAKE.value:
            await self._broadcast_ui()
            return
        if key != HealthState.DEGRADED_TTS.value:
            await self.speak_text(spoken, "health")
        self.set_state(CharacterState.NOTIFICATION)
        await self._broadcast_ui()
        await asyncio.sleep(1.6)
        if self.state is CharacterState.NOTIFICATION:
            self.set_state(CharacterState.IDLE)

    # -- turn processing -----------------------------------------------------

    def _vision_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        instruction = str(params.get("instruction") or "").strip()
        if not instruction:
            return {"ok": False, "error": "Keine Aufgabe."}
        job = self.vision.queue(instruction)
        asyncio.create_task(self.vision.run_job(job))
        return {
            "ok": True,
            "job_id": job.job_id,
            "status": "queued",
            "note": "Läuft im Hintergrund. Nutzer bleibt ansprechbar.",
        }

    async def process_turn(self, user_text: str) -> None:
        turn = self.current or TurnTelemetry(turn_id=f"turn-{int(time.time() * 1000)}")
        self.current = turn
        if self._panic:
            await self._fail_turn("Panik-Schalter ist an. Ich führe nichts aus.")
            return
        if not self.health.stt.get("ready", True):
            await self._fail_turn("Meine Spracherkennung ist ausgefallen.")
            return

        # Speed 1: bounded reflex, no model.
        launch = self.stack.direct.parse(user_text)
        if launch is not None:
            result = await self.stack.direct.execute(launch)
            await self._speak_answer(result.answer)
            return
        control = parse_direct_control(user_text)
        if control is not None:
            result = await self.stack.direct.execute_control(control)
            await self._speak_answer(result.answer)
            return

        if not self.health.llm_ok:
            await self._fail_turn("Mein Denkmodul antwortet gerade nicht.")
            return

        self.set_state(CharacterState.THINKING)
        try:
            secrets = create_secret_store()
        except Exception:
            from kiki.storage.secrets import MemorySecretStore

            secrets = MemorySecretStore()
        provider = create_provider(self.settings, secrets)
        adapter = ProviderStepAdapter(
            provider,
            model=self.cfg.llm.model,
            temperature=self.cfg.llm.temperature,
            system_prompt=voice_system_prompt(),
            num_ctx=self.cfg.llm.num_ctx,
        )
        runner = AssistantRunner(
            adapter,
            self.stack.gateway,
            profile="observe",
            trace_dir=state_dir() / "assistant",
            max_steps=self.settings.tools.max_steps,
            max_tool_calls=self.settings.tools.max_tool_calls,
        )
        chunker = StreamingTtsChunker(
            max_chunk_chars=180,
            semantic_min_words=8,
            semantic_max_words=16,
            stream_timeout_ms=300,
        )
        first_token = False
        answer_parts: list[str] = []
        run = runner.begin(user_text)
        try:
            async for event in runner.drive(run):
                if event.kind == "delta" and event.text:
                    if not first_token:
                        turn.t_llm_first_token = time.perf_counter()
                        first_token = True
                    answer_parts.append(event.text)
                    for chunk in chunker.push_token(event.text):
                        await self.tts.send(
                            {"command": "speak_chunk", "turn_id": turn.turn_id, "text": chunk}
                        )
                elif event.kind == "confirmation_requested" and event.request is not None:
                    req = event.request
                    prompt = confirmation_prompt(req.title, req.content, str(req.risk))
                    allowed = await self._ask_spoken_confirm(prompt, turn.turn_id)
                    if allowed:
                        runner.confirm(req.run_id, req.call_id, req.request_id)
                        await self.speak_phrase("confirm_yes", turn.turn_id)
                    else:
                        runner.reject(req.run_id, req.call_id)
                        await self.speak_phrase("confirm_no", turn.turn_id)
                elif event.kind == "finished":
                    for chunk in chunker.flush():
                        await self.tts.send(
                            {"command": "speak_chunk", "turn_id": turn.turn_id, "text": chunk}
                        )
                    await self.tts.send({"command": "end_turn", "turn_id": turn.turn_id})
                    if run.status is not RunStatus.COMPLETED:
                        if not answer_parts:
                            await self._fail_turn("Der Durchlauf ist fehlgeschlagen.")
                    turn.answer_preview = "".join(answer_parts)[:240]
                    self.telemetry.add(turn)
        except Exception as exc:
            log.exception("turn failed")
            await self._fail_turn(f"Das hat nicht geklappt: {exc}")

    async def _speak_answer(self, text: str) -> None:
        turn = self.current
        turn_id = turn.turn_id if turn else "sys"
        if turn:
            turn.t_llm_first_token = time.perf_counter()
            turn.answer_preview = text[:240]
            self.telemetry.add(turn)
        await self.speak_text(text, turn_id)
        self.set_state(CharacterState.SPEAKING)

    async def _ask_spoken_confirm(self, prompt: str, turn_id: str) -> bool:
        await self.speak_text(prompt, turn_id)
        loop = asyncio.get_running_loop()
        self._pending_verdict = loop.create_future()
        await self.audio.send({"command": "capture_one", "reason": "confirm"})
        try:
            heard = await asyncio.wait_for(
                self._pending_verdict, timeout=self.cfg.policy.confirmation_timeout_s
            )
        except TimeoutError:
            log.info("confirmation timed out — treated as no")
            return False
        finally:
            self._pending_verdict = None
        verdict = parse_spoken_verdict(heard)
        if verdict is None:
            log.info("unintelligible confirmation %r — treated as no", heard)
            return False
        return verdict

    # -- supervisors ---------------------------------------------------------

    async def _llm_watch(self) -> None:
        while self._running:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(f"{self.cfg.llm.base_url.rstrip('/')}/api/tags")
                    self.health.llm_ok = response.status_code < 400
                    self.health.llm_detail = "" if self.health.llm_ok else f"HTTP {response.status_code}"
            except Exception as exc:
                self.health.llm_ok = False
                self.health.llm_detail = f"Ollama unerreichbar: {exc}"
            await asyncio.sleep(5.0)

    async def _handle_ui(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._ui_clients.append(writer)
        try:
            writer.write(dumps(self.ui_payload()))
            await writer.drain()
            while self._running and not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = loads(line)
                except Exception:
                    continue
                cmd = str(msg.get("command") or "")
                if cmd == "panic":
                    self._panic = True
                    self.stack.settings.app.privacy_panic = True
                    if self.current:
                        await self.tts.send({"command": "cancel", "turn_id": self.current.turn_id})
                    self.set_state(CharacterState.ERROR)
                elif cmd == "unpanic":
                    self._panic = False
                    self.stack.settings.app.privacy_panic = False
                    self.set_state(CharacterState.IDLE)
                elif cmd == "trigger":
                    await self.audio.send({"command": "trigger_turn", "source": "ui"})
        except (ConnectionResetError, BrokenPipeError, ConnectionError):
            return
        finally:
            if writer in self._ui_clients:
                self._ui_clients.remove(writer)

    async def run(self) -> None:
        ui_path = self.cfg.socket("ui")
        ui_path.parent.mkdir(parents=True, exist_ok=True)
        if ui_path.exists():
            ui_path.unlink()
        server = await asyncio.start_unix_server(self._handle_ui, path=str(ui_path))
        os.chmod(ui_path, 0o600)
        log.info("orchestrator UI socket %s", ui_path)
        tasks = [
            asyncio.create_task(self.audio.run(self.on_audio, on_link=self._on_audio_link)),
            asyncio.create_task(self.stt.run(self.on_stt, on_link=self._on_stt_link)),
            asyncio.create_task(self.tts.run(self.on_tts, on_link=self._on_tts_link)),
            asyncio.create_task(self._llm_watch()),
        ]
        try:
            while self._running:
                await asyncio.sleep(1.0)
        finally:
            for task in tasks:
                task.cancel()
            server.close()
            await server.wait_closed()
            await self.audio.close()
            await self.stt.close()
            await self.tts.close()

    async def _on_audio_link(self, up: bool) -> None:
        if not up:
            self.health.audio = {"ready": False, "status": "failed", "error": "audio socket down"}
            await self._maybe_announce_fault()
        else:
            await self.audio.send({"command": "healthz"})

    async def _on_stt_link(self, up: bool) -> None:
        if not up:
            self.health.stt = {"ready": False, "status": "failed", "error": "stt socket down"}
            await self._maybe_announce_fault()
        else:
            await self.stt.send({"command": "healthz"})

    async def _on_tts_link(self, up: bool) -> None:
        if not up:
            self.health.tts = {"ready": False, "status": "failed", "error": "tts socket down"}
            await self._maybe_announce_fault()
        else:
            await self.tts.send({"command": "healthz"})

    def stop(self) -> None:
        self._running = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KIKI voice orchestrator")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] kiki-orchestrator: %(message)s",
    )
    orch = Orchestrator(load_runtime())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, orch.stop)
    try:
        loop.run_until_complete(orch.run())
    except KeyboardInterrupt:
        orch.stop()
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
