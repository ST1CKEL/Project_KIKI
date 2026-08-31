"""Play TTS PCM through Pulse/PipeWire as closed WAV files.

A long-lived pw-cat stdin never flushed on this PipeWire — audio sat in the
buffer until EOF. paplay (Pulse default sink) of a closed WAV is the path
that actually reaches the headset. pw-play is the fallback. Chunks play
sequentially; barge-in kills the current process.
"""

from __future__ import annotations

import collections
import logging
import shutil
import subprocess
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path

from kiki.ipc.paths import runtime_dir

log = logging.getLogger("kiki.tts.playback")

SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2
MAX_QUEUE_SECONDS = 12.0
PAPLAY_VOLUME = 65536  # 100% stream; sink volume still applies


class PlaybackError(RuntimeError):
    """PipeWire/Pulse sink could not start."""


def _pcm_to_wav(pcm: bytes, sample_rate: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(BYTES_PER_SAMPLE)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm)


def _default_sink() -> str:
    pactl = shutil.which("pactl")
    if not pactl:
        return ""
    try:
        result = subprocess.run(
            [pactl, "get-default-sink"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


class PipeWirePcmPlayback:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        *,
        on_audio_started: Callable[[str], None] | None = None,
        on_audio_finished: Callable[[str], None] | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate) if sample_rate else SAMPLE_RATE
        self.bytes_per_second = self.sample_rate * CHANNELS * BYTES_PER_SAMPLE
        self.on_audio_started = on_audio_started
        self.on_audio_finished = on_audio_finished
        self._queue: collections.deque[tuple[str, bytes, int]] = collections.deque()
        self._lock = threading.Lock()
        self._running = True
        self._speaking = False
        self._current_turn: str | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self.underrun_count = 0
        self.total_pcm_bytes_played = 0
        self.error = ""
        self._wav_dir = runtime_dir() / "tts"
        self._wav_dir.mkdir(parents=True, exist_ok=True)
        self._sink = _default_sink()
        self._paplay = shutil.which("paplay")
        pw_play = shutil.which("pw-play") or "/usr/bin/pw-play"
        self._pw_play = pw_play if Path(pw_play).is_file() else None
        if self._paplay is None and self._pw_play is None:
            self.error = "paplay/pw-play fehlen — PipeWire-Utils sind nicht installiert."
            log.error("%s", self.error)
        else:
            log.info(
                "playback ready paplay=%s pw-play=%s sink=%s rate=%d",
                self._paplay,
                self._pw_play,
                self._sink or "(auto)",
                self.sample_rate,
            )
        self._thread = threading.Thread(target=self._pump, name="kiki-tts-pump", daemon=True)
        self._thread.start()

    @property
    def ready(self) -> bool:
        return self._paplay is not None or self._pw_play is not None

    def enqueue_pcm(self, turn_id: str, pcm: bytes, sample_rate: int | None = None) -> None:
        if not pcm or not self.ready:
            return
        rate = int(sample_rate) if sample_rate else self.sample_rate
        if rate <= 0:
            rate = self.sample_rate
        max_bytes = int(MAX_QUEUE_SECONDS * max(1, rate) * CHANNELS * BYTES_PER_SAMPLE)
        with self._lock:
            queued = sum(len(chunk) for _, chunk, _ in self._queue)
            # Never drop the only chunk of a turn — a 3 s sentence is normal.
            if self._queue and queued + len(pcm) > max_bytes:
                log.warning("TTS queue > %.1fs — dropping extra chunk", MAX_QUEUE_SECONDS)
                self.underrun_count += 1
                return
            self._queue.append((turn_id, pcm, rate))
            self._speaking = True
            self.sample_rate = rate
            self.bytes_per_second = rate * CHANNELS * BYTES_PER_SAMPLE
            log.info(
                "queued %d bytes for turn %s (%d chunks waiting, rate=%d)",
                len(pcm),
                turn_id,
                len(self._queue),
                rate,
            )

    def cancel_turn(self, turn_id: str | None = None) -> None:
        with self._lock:
            if turn_id:
                self._queue = collections.deque(
                    (t, b, r) for t, b, r in self._queue if t != turn_id
                )
            else:
                self._queue.clear()
            self._speaking = False
            self._current_turn = None
        self._kill_play()

    def mark_turn_complete(self) -> None:
        with self._lock:
            if not self._queue:
                self._speaking = False

    def _kill_play(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=0.4)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _player_cmd(self, wav_path: Path) -> list[str] | None:
        if self._paplay:
            cmd = [self._paplay, f"--volume={PAPLAY_VOLUME}"]
            sink = self._sink or _default_sink()
            self._sink = sink
            if sink:
                cmd.append(f"--device={sink}")
            cmd.append(str(wav_path))
            return cmd
        if self._pw_play:
            cmd = [self._pw_play, "--volume=1.0", "--media-role=Speech"]
            sink = self._sink or _default_sink()
            self._sink = sink
            if sink:
                cmd.extend(["--target", sink])
            cmd.append(str(wav_path))
            return cmd
        return None

    def _emit(self, callback: Callable[[str], None] | None, turn_id: str) -> None:
        if callback is None:
            return
        try:
            callback(turn_id)
        except Exception:
            log.debug("playback callback failed", exc_info=True)

    def _pump(self) -> None:
        announced: set[str] = set()
        n = 0
        while self._running:
            item: tuple[str, bytes, int] | None = None
            with self._lock:
                if self._queue:
                    item = self._queue.popleft()
            if item is None:
                time.sleep(0.02)
                continue
            turn_id, pcm, rate = item
            self._current_turn = turn_id
            if turn_id not in announced:
                announced.add(turn_id)
                self._emit(self.on_audio_started, turn_id)
            n += 1
            wav_path = self._wav_dir / f"{turn_id}-{n}.wav"
            try:
                _pcm_to_wav(pcm, rate, wav_path)
            except Exception as extra:
                log.error("could not write wav: %s", extra)
                continue
            cmd = self._player_cmd(wav_path)
            if cmd is None:
                continue
            log.info(
                "playing %s (%d bytes, %d Hz) via %s sink=%s",
                wav_path.name,
                len(pcm),
                rate,
                cmd[0],
                self._sink or "(auto)",
            )
            timeout_s = max(3.0, len(pcm) / max(1, rate * BYTES_PER_SAMPLE) + 3.0)
            proc: subprocess.Popen[bytes] | None = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                self._proc = proc
                try:
                    _out, err = proc.communicate(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    _out, err = proc.communicate()
                    log.error("playback timed out after %.1fs turn=%s", timeout_s, turn_id)
                    self.underrun_count += 1
                    err = err or b"timeout"
                rc = proc.returncode
                if rc not in (0, None) and isinstance(rc, int) and rc < 0:
                    log.info("playback interrupted rc=%s turn=%s", rc, turn_id)
                elif rc not in (0, None):
                    detail = (err or b"").decode("utf-8", "replace").strip()[:400]
                    self.error = detail or f"{cmd[0]} exit {rc}"
                    log.error("playback failed rc=%s: %s", rc, self.error)
                    self.underrun_count += 1
                else:
                    self.total_pcm_bytes_played += len(pcm)
                    self.error = ""
            except Exception as extra:
                log.warning("player failed: %s", extra)
                self.underrun_count += 1
            finally:
                if self._proc is proc:
                    self._proc = None
                try:
                    wav_path.unlink(missing_ok=True)
                except Exception:
                    pass
            with self._lock:
                idle = not self._queue
                if idle:
                    self._speaking = False
                    self._current_turn = None
            if idle:
                announced.discard(turn_id)
                self._emit(self.on_audio_finished, turn_id)

    def close(self) -> None:
        self._running = False
        self._kill_play()
