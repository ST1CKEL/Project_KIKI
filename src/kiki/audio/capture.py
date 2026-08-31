"""PipeWire microphone capture via GStreamer pulsesrc.

Law 1: a missing pipeline is a hard failure. We never invent silence and
pretend the microphone works.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("kiki.audio.capture")


class CaptureError(RuntimeError):
    """Microphone pipeline could not start."""


class MicrophoneCapture:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._pipeline: Any = None
        self._sink: Any = None
        self._gst: Any = None
        self._ready = False
        self.error = ""
        self._start()

    @property
    def ready(self) -> bool:
        return self._ready

    def _start(self) -> None:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            gi.require_version("GstApp", "1.0")
            from gi.repository import Gst, GstApp
        except Exception as exc:
            self.error = f"GStreamer nicht ladbar: {exc}"
            log.error("%s", self.error)
            return
        Gst.init(None)
        launch = (
            "pulsesrc ! audioconvert ! audioresample ! "
            f"audio/x-raw,format=S16LE,channels=1,rate={self.sample_rate} ! "
            "appsink name=kikisink emit-signals=false sync=false max-buffers=8 drop=true"
        )
        try:
            pipeline = Gst.parse_launch(launch)
            raw_sink = pipeline.get_by_name("kikisink")
            if raw_sink is None:
                raise CaptureError("appsink fehlt")
            sink = GstApp.AppSink.cast(raw_sink) if hasattr(GstApp.AppSink, "cast") else raw_sink
            result = pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                raise CaptureError("Pipeline startete nicht (PipeWire/pulsesrc)")
            self._gst = Gst
            self._pipeline = pipeline
            self._sink = sink
            self._ready = True
            log.info("microphone pipeline playing at %d Hz", self.sample_rate)
        except Exception as exc:
            self.error = f"Mikrofon fehlgeschlagen: {exc}"
            log.error("%s", self.error)
            self._ready = False

    def read(self, timeout_ms: int = 40) -> bytes:
        """Return the next PCM chunk, or b'' on timeout. Never synthetic noise."""
        if not self._ready or self._sink is None or self._gst is None:
            return b""
        timeout_ns = int(timeout_ms) * int(self._gst.MSECOND)
        if hasattr(self._sink, "try_pull_sample"):
            sample = self._sink.try_pull_sample(timeout_ns)
        else:
            sample = self._sink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            return b""
        buf = sample.get_buffer()
        if buf is None:
            return b""
        ok, mapped = buf.map(self._gst.MapFlags.READ)
        if not ok:
            return b""
        data = bytes(mapped.data)
        buf.unmap(mapped)
        return data

    def close(self) -> None:
        if self._pipeline is None or self._gst is None:
            return
        try:
            self._pipeline.set_state(self._gst.State.NULL)
        except Exception:
            pass
        self._ready = False
        self._pipeline = None
        self._sink = None
