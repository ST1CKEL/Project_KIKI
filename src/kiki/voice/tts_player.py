"""Play WAV files through PipeWire (GStreamer pulsesink, pw-play fallback)."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)


def _idle(callback: Callable[[], bool]) -> None:
    try:
        from gi.repository import GLib

        GLib.idle_add(callback)
        return
    except Exception:
        callback()


class PipeWirePlayer:
    """Sequential WAV playback. stop() is barge-in."""

    def __init__(self) -> None:
        self._pipeline = None
        self._bus_handler_id: int | None = None
        self._pw_proc: subprocess.Popen[bytes] | None = None
        self._token = 0
        self._on_eos: Callable[[], None] | None = None
        self._on_error: Callable[[str], None] | None = None

    @property
    def playing(self) -> bool:
        return self._pipeline is not None or self._pw_proc is not None

    def play(
        self,
        path: Path,
        *,
        on_eos: Callable[[], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.stop()
        self._token += 1
        token = self._token
        self._on_eos = on_eos
        self._on_error = on_error

        def _start() -> bool:
            if token != self._token:
                return False
            self._launch(path, token)
            return False

        _idle(_start)

    def stop(self) -> None:
        self._token += 1
        self._on_eos = None
        self._on_error = None
        _idle(self._teardown)

    def _teardown(self) -> bool:
        pipeline = self._pipeline
        handler_id = self._bus_handler_id
        self._pipeline = None
        self._bus_handler_id = None
        if pipeline is not None:
            try:
                from gi.repository import Gst

                bus = pipeline.get_bus()
                if handler_id is not None:
                    bus.disconnect(handler_id)
                bus.remove_signal_watch()
                pipeline.set_state(Gst.State.NULL)
            except Exception:
                log.debug("gst stop failed", exc_info=True)
        proc = self._pw_proc
        self._pw_proc = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.5)
            except Exception:
                proc.kill()
        return False

    def _launch(self, path: Path, token: int) -> None:
        if not path.is_file():
            self._fail(token, f"WAV fehlt: {path}")
            return
        if self._try_gstreamer(path, token):
            return
        self._try_pw_play(path, token)

    def _try_gstreamer(self, path: Path, token: int) -> bool:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            Gst.init(None)
            location = str(path).replace("\\", "\\\\").replace('"', '\\"')
            launch = (
                f'filesrc location="{location}" ! wavparse ! audioconvert ! audioresample ! '
                "pulsesink client-name=KIKI"
            )
            pipeline = Gst.parse_launch(launch)
            bus = pipeline.get_bus()
            bus.add_signal_watch()

            def _on_bus(_bus: object, message: object) -> None:
                if token != self._token:
                    return
                kind = message.type
                if kind == Gst.MessageType.EOS:
                    self._succeed(token)
                elif kind == Gst.MessageType.ERROR:
                    err, debug = message.parse_error()
                    self._fail(token, str(err) if err else (debug or "GStreamer-Fehler"))

            handler_id = bus.connect("message", _on_bus)
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                bus.disconnect(handler_id)
                bus.remove_signal_watch()
                pipeline.set_state(Gst.State.NULL)
                return False
            self._pipeline = pipeline
            self._bus_handler_id = handler_id
            log.info("playing %s via GStreamer/PipeWire", path.name)
            return True
        except Exception:
            log.debug("gstreamer playback unavailable", exc_info=True)
            return False

    def _try_pw_play(self, path: Path, token: int) -> None:
        try:
            proc = subprocess.Popen(
                ["pw-play", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self._fail(token, "Weder GStreamer noch pw-play gefunden.")
            return
        except OSError as exc:
            self._fail(token, str(exc))
            return
        self._pw_proc = proc
        log.info("playing %s via pw-play", path.name)

        def _poll() -> bool:
            if token != self._token:
                return False
            code = proc.poll()
            if code is None:
                return True
            self._pw_proc = None
            if code == 0:
                self._succeed(token)
            else:
                self._fail(token, f"pw-play endete mit {code}")
            return False

        try:
            from gi.repository import GLib

            GLib.timeout_add(40, _poll)
        except Exception:
            proc.wait(timeout=30)
            if token == self._token:
                self._succeed(token)

    def _succeed(self, token: int) -> None:
        if token != self._token:
            return
        callback = self._on_eos
        self._on_eos = None
        self._on_error = None
        self._teardown()
        if callback is not None:
            callback()

    def _fail(self, token: int, message: str) -> None:
        if token != self._token:
            return
        callback = self._on_error
        self._on_eos = None
        self._on_error = None
        self._teardown()
        log.warning("playback failed: %s", message)
        if callback is not None:
            callback(message)
