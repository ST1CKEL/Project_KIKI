"""16 kHz mono WAV capture via GStreamer (PipeWire/Pulse)."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class RecorderError(Exception):
    """Microphone pipeline failed."""


class AudioRecorder:
    def __init__(self) -> None:
        self._pipeline = None
        self._path: Path | None = None

    @property
    def recording(self) -> bool:
        return self._pipeline is not None

    @property
    def path(self) -> Path | None:
        return self._path

    def start(self, path: Path) -> None:
        if self._pipeline is not None:
            self.stop()
        pipeline = None
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            Gst.init(None)
            missing = [
                name
                for name in ("pulsesrc", "audioconvert", "audioresample", "wavenc", "filesink")
                if Gst.ElementFactory.find(name) is None
            ]
            if missing:
                raise RecorderError(
                    "Audio-Komponenten fehlen: "
                    + ", ".join(missing)
                    + ". Installiere KIKIs Fedora-Audioabhängigkeiten."
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            location = str(path).replace("\\", "\\\\").replace('"', '\\"')
            launch = (
                "pulsesrc ! audioconvert ! audioresample ! "
                "audio/x-raw,format=S16LE,channels=1,rate=16000 ! "
                f'wavenc ! filesink location="{location}"'
            )
            pipeline = Gst.parse_launch(launch)
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RecorderError("Mikrofon-Pipeline startet nicht (PipeWire/Pulse).")
        except RecorderError:
            if pipeline is not None:
                pipeline.set_state(Gst.State.NULL)
            raise
        except Exception as exc:
            if pipeline is not None:
                try:
                    pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass
            raise RecorderError(f"Mikrofon ist nicht verfügbar: {exc}") from exc
        self._pipeline = pipeline
        self._path = path
        log.info("recording to %s", path)

    def stop(self) -> Path | None:
        pipeline = self._pipeline
        self._pipeline = None
        path = self._path
        if pipeline is None:
            return path
        try:
            from gi.repository import Gst

            # wavenc writes the final RIFF sizes only after EOS reaches the
            # filesink. Immediately forcing NULL can leave an unreadable WAV.
            pipeline.send_event(Gst.Event.new_eos())
            bus = pipeline.get_bus()
            if bus is not None:
                message = bus.timed_pop_filtered(
                    2 * Gst.SECOND,
                    Gst.MessageType.EOS | Gst.MessageType.ERROR,
                )
                if message is None:
                    log.warning("timed out while finalizing microphone WAV")
                elif message.type == Gst.MessageType.ERROR:
                    error, debug = message.parse_error()
                    log.warning("microphone finalization failed: %s (%s)", error, debug)
        except Exception:
            log.warning("could not finalize microphone WAV", exc_info=True)
        finally:
            try:
                from gi.repository import Gst

                pipeline.set_state(Gst.State.NULL)
            except Exception:
                log.debug("could not stop microphone pipeline", exc_info=True)
        return path
