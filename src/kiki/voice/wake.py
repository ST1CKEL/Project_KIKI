"""Local wake-word spotting: "KIKI".

Vosk **grammar mode** was tried first and rejected on measurements. Restricted
to `["kiki", "hey kiki", …, "[unk]"]`, the small German model maps arbitrary
speech onto the wake phrases: "der schlüssel liegt auf dem tisch" decoded to
"hey kiki" and "der key ist abgelaufen" to "kiki" — 6 false alarms across a
12-sentence negative corpus. The same corpus through the **open** recognizer
produced the real sentences and not one stray "kiki" token.

So the listener runs ordinary local recognition. That is a real privacy
tradeoff, and these rules are what make it defensible:

* Audio is never written to disk and never leaves the process.
* While waiting, recognized text never leaves this module. It is matched
  against the wake phrases and dropped — never stored, never logged, never
  sent to a model.
* Text escapes exactly once the wake word was heard: the **next** utterance is
  handed over as the spoken command. Creating that boundary is the entire point
  of a wake word.
* After an answered wake-word turn, the application may explicitly arm one
  follow-up utterance. Silence returns the listener to wake-word matching.
* The feature is opt-in and off by default, the panic switch kills it, and the
  figure shows visibly when the microphone is open.

After the wake word the same recognizer keeps running, so the command utterance
needs no second pipeline and no guessed recording length: Vosk ends an utterance
about 1.0–1.15 s after speech stops.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable
from enum import StrEnum
from pathlib import Path

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Just the name. "hey kiki" and "hallo kiki" contain it, so they trigger too
# without needing their own entries.
DEFAULT_PHRASES: tuple[str, ...] = ("kiki",)
DEFAULT_COOLDOWN_MS = 2000
# How long to wait for a command before falling back to listening for the wake
# word. Room noise can keep Vosk from ending an utterance on its own.
DEFAULT_COMMAND_TIMEOUT_S = 12.0


class WakeError(Exception):
    """The wake word could not be armed."""


class ListenerState(StrEnum):
    WAITING = "waiting"  # matching the wake word; text never escapes
    CAPTURING = "capturing"  # the next utterance is the user's command


def normalize(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def phrase_matches(heard: str, phrases: Iterable[str]) -> bool:
    """True when a wake phrase appears in `heard` on word boundaries.

    Containment is safe here because the open recognizer transcribes what was
    actually said: "ich habe eine kiwi gekauft" stays "kiwi" and never yields a
    "kiki" token. Boundaries still matter, so "kikiriki" does not match.
    """
    tokens = normalize(heard).split()
    if not tokens:
        return False
    for phrase in phrases:
        wanted = normalize(phrase).split()
        if not wanted:
            continue
        for start in range(len(tokens) - len(wanted) + 1):
            if tokens[start : start + len(wanted)] == wanted:
                return True
    return False


def missing_words(model, phrases: Iterable[str]) -> list[str]:
    """Wake-phrase words the model's lexicon does not know.

    A word the acoustic model cannot represent would never be recognised, so
    this is checked up front instead of failing silently at runtime.
    """
    unknown: list[str] = []
    for phrase in phrases:
        for word in normalize(phrase).split():
            try:
                found = model.FindWord(word)
            except Exception:  # pragma: no cover - older runtime without the symbol
                return []
            if found < 0 and word not in unknown:
                unknown.append(word)
    return unknown


class UtteranceStream:
    """Feeds PCM into a Vosk recognizer and yields finished utterances.

    Deliberately dumb: it knows nothing about wake words, so the matching policy
    can be tested without audio hardware and this part can be swapped.
    """

    def __init__(
        self,
        *,
        model_dir: Path,
        sample_rate: int = SAMPLE_RATE,
        model_factory: Callable[[str], object] | None = None,
        recognizer_factory: Callable[..., object] | None = None,
    ) -> None:
        self._model_dir = model_dir
        self._sample_rate = sample_rate
        self._model_factory = model_factory
        self._recognizer_factory = recognizer_factory
        self._model = None
        self._recognizer = None

    def open(self, *, phrases: Iterable[str] = ()) -> None:
        """Load the model. Call off the GTK thread; this takes a moment."""
        if self._recognizer is not None:
            return
        model_factory = self._model_factory
        recognizer_factory = self._recognizer_factory
        if model_factory is None or recognizer_factory is None:
            model_factory, recognizer_factory = _default_factories()
        try:
            model = model_factory(str(self._model_dir))
        except Exception as exc:
            raise WakeError(f"Weckwort-Modell konnte nicht geladen werden: {exc}") from exc
        unknown = missing_words(model, phrases)
        if unknown:
            raise WakeError(
                "Das deutsche Sprachmodell kennt diese Weckwörter nicht: " + ", ".join(unknown)
            )
        try:
            self._recognizer = recognizer_factory(model, self._sample_rate)
        except Exception as exc:
            raise WakeError(f"Weckwort-Erkenner konnte nicht erstellt werden: {exc}") from exc
        self._model = model

    def close(self) -> None:
        self._recognizer = None
        self._model = None

    def reset(self) -> None:
        recognizer = self._recognizer
        if recognizer is None:
            return
        try:
            recognizer.Reset()
        except Exception:
            log.debug("recognizer reset failed", exc_info=True)

    def feed(self, pcm: bytes) -> str | None:
        """Return the finished utterance, or None while one is still forming."""
        recognizer = self._recognizer
        if recognizer is None or not pcm:
            return None
        try:
            if not recognizer.AcceptWaveform(pcm):
                return None
            return _heard_text(recognizer.Result())
        except Exception as exc:
            log.warning("recognizer failed: %s", exc)
            return None


def _heard_text(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("text") or "")


def _default_factories():
    from kiki.voice.stt import _vosk_api  # noqa: PLC0415 - optional native runtime

    KaldiRecognizer, Model, SetLogLevel = _vosk_api()
    SetLogLevel(-1)

    def make_recognizer(model, rate):
        return KaldiRecognizer(model, rate)

    return Model, make_recognizer


def wake_word_supported() -> bool:
    """Whether this machine's Vosk runtime can drive the wake listener."""
    try:
        from kiki.voice.vosk_ffi import wake_support_available
    except ImportError:
        # The pip fallback runtime exposes the same calls through its wrapper.
        try:
            import vosk  # noqa: F401
        except ImportError:
            return False
        return True
    return wake_support_available()


class MicrophoneStream:
    """Continuous 16 kHz mono PCM from PipeWire/Pulse via a GStreamer appsink."""

    def __init__(self, *, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._pipeline = None
        self._sink = None

    @property
    def running(self) -> bool:
        return self._pipeline is not None

    def start(self) -> None:
        if self._pipeline is not None:
            return
        pipeline = None
        try:
            import gi

            gi.require_version("Gst", "1.0")
            # GstApp registers AppSink's methods (try_pull_sample). Without it
            # the element comes back as a plain GstElement wrapper.
            gi.require_version("GstApp", "1.0")
            from gi.repository import Gst, GstApp  # noqa: F401 — registers AppSink

            Gst.init(None)
            missing = [
                name
                for name in ("pulsesrc", "audioconvert", "audioresample", "appsink")
                if Gst.ElementFactory.find(name) is None
            ]
            if missing:
                raise WakeError("Audio-Komponenten fehlen: " + ", ".join(missing))
            launch = (
                "pulsesrc ! audioconvert ! audioresample ! "
                f"audio/x-raw,format=S16LE,channels=1,rate={self._sample_rate} ! "
                # drop=true bounds latency if recognition falls behind, instead
                # of growing an unbounded backlog of stale audio.
                "appsink name=kikiwake emit-signals=false sync=false "
                "max-buffers=16 drop=true"
            )
            pipeline = Gst.parse_launch(launch)
            sink = pipeline.get_by_name("kikiwake")
            if sink is None:
                raise WakeError("Weckwort-Pipeline hat keinen appsink.")
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise WakeError("Mikrofon-Pipeline startet nicht (PipeWire/Pulse).")
        except WakeError:
            _force_null(pipeline)
            raise
        except Exception as exc:
            _force_null(pipeline)
            raise WakeError(f"Mikrofon ist nicht verfügbar: {exc}") from exc
        self._pipeline = pipeline
        self._sink = sink

    def read(self, timeout_ms: int = 200) -> bytes:
        """Pull one chunk. Returns b"" on timeout or end of stream."""
        sink = self._sink
        if sink is None:
            return b""
        from gi.repository import Gst

        sample = sink.try_pull_sample(timeout_ms * Gst.MSECOND)
        if sample is None:
            return b""
        buffer = sample.get_buffer()
        if buffer is None:
            return b""
        ok, info = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return b""
        try:
            return bytes(info.data)
        finally:
            buffer.unmap(info)

    def stop(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        self._sink = None
        _force_null(pipeline)


def _force_null(pipeline: object | None) -> None:
    if pipeline is None:
        return
    try:
        from gi.repository import Gst

        pipeline.set_state(Gst.State.NULL)
    except Exception:
        log.debug("could not stop wake pipeline", exc_info=True)


class WakeWordListener:
    """Wake word, then the command after it, on one background thread.

    `on_detect` and `on_command` are invoked from that thread; callers are
    responsible for hopping to the UI thread.
    """

    def __init__(
        self,
        *,
        stream: UtteranceStream,
        microphone: MicrophoneStream | None = None,
        phrases: Iterable[str] = DEFAULT_PHRASES,
        cooldown_ms: int = DEFAULT_COOLDOWN_MS,
        command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S,
        on_detect: Callable[[], None],
        on_command: Callable[[str], None],
        on_timeout: Callable[[], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        cleaned = tuple(normalize(p) for p in phrases if normalize(p))
        if not cleaned:
            raise WakeError("Kein Weckwort konfiguriert.")
        self._phrases = cleaned
        self._stream = stream
        self._microphone = microphone or MicrophoneStream()
        self._cooldown = max(0.0, cooldown_ms / 1000.0)
        self._command_timeout = max(1.0, command_timeout_s)
        self._on_detect = on_detect
        self._on_command = on_command
        self._on_timeout = on_timeout
        self._on_error = on_error
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        # Vosk's recognizer is not thread-safe. The GTK thread resets it when
        # speech ends, while the listener thread normally feeds it PCM.
        self._stream_lock = threading.Lock()
        self._lock = threading.Lock()
        self._state = ListenerState.WAITING
        self._cooldown_until = 0.0
        self._capture_deadline = 0.0

    @property
    def phrases(self) -> tuple[str, ...]:
        return self._phrases

    @property
    def state(self) -> ListenerState:
        return self._state

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self._stop.clear()
            self._paused.clear()
            self._state = ListenerState.WAITING
            self._stream.open(phrases=self._phrases)
            self._microphone.start()
            thread = threading.Thread(target=self._run, name="kiki-wake", daemon=True)
            self._thread = thread
            thread.start()

    def pause(self) -> None:
        """Stop matching without releasing the microphone (KIKI is speaking)."""
        self._paused.set()

    def resume(self) -> None:
        if self._paused.is_set():
            # Audio heard while paused must not leak into the next utterance.
            with self._stream_lock:
                self._stream.reset()
                self._paused.clear()
            self._state = ListenerState.WAITING

    def capture_next(self) -> bool:
        """Arm one utterance without requiring another wake word.

        The caller owns the visible listening indicator. Returning ``False``
        means KIKI is still paused and no microphone window was opened.
        """
        if self._paused.is_set() or self._stop.is_set():
            return False
        with self._stream_lock:
            self._stream.reset()
            self._state = ListenerState.CAPTURING
            self._capture_deadline = time.monotonic() + self._command_timeout
        return True

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            thread = self._thread
            self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3)
        self._microphone.stop()
        self._stream.close()
        self._state = ListenerState.WAITING

    def handle(self, text: str | None, *, now: float | None = None) -> None:
        """State machine for one finished utterance. Separated out for testing."""
        moment = time.monotonic() if now is None else now
        if self._state is ListenerState.CAPTURING and moment > self._capture_deadline:
            self._state = ListenerState.WAITING
            if self._on_timeout is not None:
                self._on_timeout()
        if text is None:
            return
        if self._state is ListenerState.CAPTURING:
            self._state = ListenerState.WAITING
            self._cooldown_until = moment + self._cooldown
            self._on_command(text.strip())
            return
        if moment < self._cooldown_until:
            return
        if not phrase_matches(text, self._phrases):
            # Discarded here. Non-wake speech never leaves the listener.
            return
        log.info("wake word detected")
        self._state = ListenerState.CAPTURING
        self._capture_deadline = moment + self._command_timeout
        self._on_detect()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                chunk = self._microphone.read()
                if self._stop.is_set():
                    break
                if self._paused.is_set():
                    # Keep draining so the queue cannot back up, but never match.
                    continue
                if not chunk:
                    # Still tick the state machine so a capture can time out.
                    self.handle(None)
                    continue
                with self._stream_lock:
                    heard = self._stream.feed(chunk)
                self.handle(heard)
        except Exception as exc:  # pragma: no cover - hardware failure path
            log.exception("wake listener crashed")
            if self._on_error is not None:
                self._on_error(exc)
