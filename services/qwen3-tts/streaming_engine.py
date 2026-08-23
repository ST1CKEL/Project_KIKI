#!/usr/bin/env python3
"""The real Qwen3-TTS streaming engine behind `POST /v1/synthesize/stream`.

Importing this module pulls in nothing heavy — no torch, no qwen_tts, no model.
Everything it needs is duck-typed off the synthesizer it is handed, so the
service still starts, and the WAV route still works, on a machine where the
streaming runtime is missing or has moved on.

Why this exists at all
----------------------
The installed runtime offers no streaming API: there is no `streamer` support
anywhere in qwen_tts, `Qwen3TTSForConditionalGeneration.generate` builds a
closed `talker_kwargs` dict and drops `**kwargs`, and the library's own
docstring says `non_streaming_mode` merely simulates streaming *text input*.
What it does provide is a causal decoder — `chunked_decode(codes, chunk_size,
left_context_size)` decodes a slice with left context and discards exactly
`context * total_upsample` leading samples, which is the boundary problem
already solved.

So the engine wraps exactly two methods of the talker, for exactly the duration
of one request:

* `forward`, to see each decoding step's codes as they arrive;
* `generate`, to inject a `stopping_criteria` that ends the loop on cancel.

Both are restored in a `finally` on every path. A cancel is a flag the criteria
reads, never an exception used as control flow.

Known limit: at RTF ~1.4 the GPU produces slower than speech is consumed. This
engine streams anyway and does not hide that behind a buffer; underrun is the
playback side's problem and is handled in a later slice.
"""

from __future__ import annotations

import functools
import logging
import queue
import threading
from array import array
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from streaming_http import (
    BYTES_PER_SAMPLE,
    STREAM_SAMPLE_RATE,
    CancelToken,
    EngineUnavailable,
    StreamSpec,
)

log = logging.getLogger("kiki-tts.engine")

# The runtime this engine was written against. Both are pinned conservatively:
# the tap depends on internals, and 0.x libraries move without warning.
MIN_QWEN_TTS = (0, 1, 1)
MAX_QWEN_TTS = (0, 2, 0)          # exclusive
MIN_TRANSFORMERS = (4, 57, 0)
MAX_TRANSFORMERS = (5, 0, 0)      # exclusive
# Only used to build one tensor per chunk, but without it nothing can decode.
MIN_TORCH = (2, 0, 0)

# 12 Hz codec: upsample_rates (8,5,4,3) x upsampling_ratios (2,2) = 1920 samples
# per code, i.e. exactly 80 ms at 24 kHz.
EXPECTED_UPSAMPLE = 1920
EXPECTED_SAMPLE_RATE = STREAM_SAMPLE_RATE
# What chunked_decode() itself uses. Measured to cost no extra decode time.
DEFAULT_LEFT_CONTEXT = 25

# The tap is bounded: a talker that outruns the decoder must wait, not pile up.
CODE_QUEUE_LIMIT = 64
PUT_TIMEOUT_S = 0.1
# A cancelled generate() returns at its next step, which is tens of ms. Well
# past that and something is genuinely wrong.
WORKER_JOIN_TIMEOUT_S = 15.0


class EngineBusy(EngineUnavailable):
    """A generation is already running. Subclasses EngineUnavailable so the
    endpoint answers 503 without having to know this module exists."""


@dataclass(frozen=True)
class GuardReport:
    """Whether the runtime matches what the tap was written against.

    `reason` is a fixed category, never a path, a version string or a message —
    it is handed straight to /health.
    """

    ok: bool
    reason: str = ""
    detail: str = ""          # for the local log only, never for a response

    @property
    def health_reason(self) -> str | None:
        return None if self.ok else (self.reason or "runtime_incompatible")


def _version(package: str) -> tuple[int, ...] | None:
    try:
        from importlib.metadata import version

        raw = version(package)
    except Exception:
        return None
    parts: list[int] = []
    for piece in raw.split(".")[:3]:
        digits = "".join(c for c in piece if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else None


def _callable_attr(owner: Any, name: str) -> bool:
    return callable(getattr(owner, name, None))


def check_runtime(model: Any) -> GuardReport:
    """Verify every internal the tap relies on, before touching anything.

    A failure here installs no hook and leaves the model exactly as it was. The
    service keeps answering WAV requests; only the stream route reports 503.
    """
    qwen = _version("qwen-tts")
    if qwen is None:
        return GuardReport(False, "qwen_tts_missing")
    if not MIN_QWEN_TTS <= qwen < MAX_QWEN_TTS:
        # The version is a local package number, not a secret, and without it
        # the mismatch cannot be diagnosed.
        return GuardReport(False, "qwen_tts_version", f"qwen-tts {qwen}")

    transformers = _version("transformers")
    if transformers is None:
        return GuardReport(False, "transformers_missing")
    if not MIN_TRANSFORMERS <= transformers < MAX_TRANSFORMERS:
        return GuardReport(False, "transformers_version", f"transformers {transformers}")

    torch_version = _version("torch")
    if torch_version is None:
        return GuardReport(False, "torch_missing")
    if torch_version < MIN_TORCH:
        return GuardReport(False, "torch_version", f"torch {torch_version}")

    talker = getattr(model, "talker", None)
    if talker is None:
        return GuardReport(False, "no_talker")
    if not _callable_attr(talker, "forward"):
        return GuardReport(False, "talker_forward")
    if not _callable_attr(talker, "generate"):
        return GuardReport(False, "talker_generate")

    tokenizer = getattr(model, "speech_tokenizer", None)
    if tokenizer is None or getattr(tokenizer, "model", None) is None:
        return GuardReport(False, "no_speech_tokenizer")
    if not _callable_attr(tokenizer, "decode"):
        return GuardReport(False, "tokenizer_decode")

    decoder = getattr(tokenizer.model, "decoder", None)
    if decoder is None:
        return GuardReport(False, "decoder_missing")
    if not _callable_attr(decoder, "chunked_decode"):
        # The causal chunking is the whole reason this is possible; without it
        # the engine would have to invent crossfades, which it must not.
        return GuardReport(False, "chunked_decode_missing")

    upsample = getattr(decoder, "total_upsample", None)
    if upsample is None:
        return GuardReport(False, "upsample_missing")
    if int(upsample) != EXPECTED_UPSAMPLE:
        return GuardReport(False, "upsample_mismatch", f"total_upsample={int(upsample)}")

    if not _callable_attr(tokenizer.model, "get_output_sample_rate"):
        return GuardReport(False, "sample_rate_missing")
    rate = int(tokenizer.model.get_output_sample_rate())
    if rate != EXPECTED_SAMPLE_RATE:
        return GuardReport(False, "sample_rate_mismatch", f"rate={rate}")

    talker_config = getattr(getattr(model, "config", None), "talker_config", None)
    eos = getattr(talker_config, "codec_eos_token_id", None)
    if eos is None:
        return GuardReport(False, "eos_token_missing")

    return GuardReport(True)


def float_to_pcm16(samples: Any) -> bytes:
    """Model floats to little-endian PCM16.

    Stdlib only, and deliberately not shared with the spike: production must not
    import an experiment. Scaling by 32767 keeps a sample sitting exactly at 1.0
    from wrapping to -32768, which would be a click at the loudest moment.
    """
    flat = samples.reshape(-1).tolist() if hasattr(samples, "reshape") else list(samples)
    out = array("h", (max(-32767, min(32767, int(value * 32767.0))) for value in flat))
    import sys

    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def codes_per_chunk(chunk_ms: int, samples_per_code: int, sample_rate: int) -> int:
    """Codec tokens per requested chunk length, rounded up, never below one."""
    per_code_ms = samples_per_code * 1000 / sample_rate
    return max(1, int(-(-chunk_ms // per_code_ms)))


# Stamped on the wrappers so their presence can be told apart from any other
# instance attribute a caller may have set on the talker.
HOOK_MARK = "_kiki_stream_hook"


def hooks_installed(talker: Any) -> bool:
    """True while this engine's wrappers shadow the talker's own methods.

    Restoring means removing the instance attribute, not assigning the captured
    bound method back: `talker.forward` builds a fresh bound method on every
    access, so leaving one in `__dict__` would be residue even if it behaves.
    """
    own = getattr(talker, "__dict__", {})
    return any(getattr(own.get(name), HOOK_MARK, False) for name in ("forward", "generate"))


class _CancelStop:
    """A transformers StoppingCriteria that reads one request's token.

    This is the cooperative half: `generate` sees every sequence as finished and
    returns through its normal path. Nothing raises, so a cancel never travels
    as control flow and cannot be confused with a real failure.
    """

    def __init__(self, token: CancelToken) -> None:
        self._token = token

    def __call__(self, input_ids: Any, scores: Any = None, **_kwargs: Any) -> Any:
        done = self._token.cancelled
        try:
            import torch

            return torch.full(
                (input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device
            )
        except Exception:
            # A fake talker in a test has no torch tensors; a plain bool is the
            # honest answer there.
            return done


class QwenStreamingEngine:
    """Streams PCM from a loaded Qwen3-TTS model. One generation at a time."""

    def __init__(self, synthesizer: Any, *, left_context: int = DEFAULT_LEFT_CONTEXT) -> None:
        self._synth = synthesizer
        self._wrapper = getattr(synthesizer, "model", None)
        self._left_context = max(0, int(left_context))
        self._busy = threading.Lock()
        self._degraded = ""
        model = getattr(self._wrapper, "model", None)
        self._model = model
        self._guard = (
            check_runtime(model) if model is not None else GuardReport(False, "no_model")
        )
        if not self._guard.ok:
            log.warning(
                "streaming disabled: %s%s",
                self._guard.reason,
                f" ({self._guard.detail})" if self._guard.detail else "",
            )

    # --- what the endpoint asks --------------------------------------------

    @property
    def available(self) -> bool:
        return self._guard.ok and not self._degraded

    @property
    def reason(self) -> str | None:
        if self._degraded:
            return self._degraded
        return self._guard.health_reason

    # --- generation ---------------------------------------------------------

    def stream(self, spec: StreamSpec, token: CancelToken) -> Iterator[bytes]:
        if not self.available:
            raise EngineUnavailable(self.reason or "runtime_incompatible")
        # Non-blocking: a second caller is refused, and refusing must never
        # touch the hooks the first one installed.
        if not self._busy.acquire(blocking=False):
            raise EngineBusy("Eine Generation läuft bereits")
        try:
            yield from self._run(spec, token)
        finally:
            self._busy.release()

    def _run(self, spec: StreamSpec, token: CancelToken) -> Iterator[bytes]:
        talker = self._model.talker
        tokenizer = self._model.speech_tokenizer
        samples_per_code = int(tokenizer.model.decoder.total_upsample)
        eos = int(self._model.config.talker_config.codec_eos_token_id)
        per_chunk = codes_per_chunk(spec.chunk_ms, samples_per_code, spec.sample_rate)

        sink: queue.Queue = queue.Queue(maxsize=CODE_QUEUE_LIMIT)
        had_forward = "forward" in talker.__dict__
        had_generate = "generate" in talker.__dict__
        original_forward = talker.forward
        original_generate = talker.generate
        worker: threading.Thread | None = None
        try:
            talker.forward = self._wrap_forward(original_forward, sink, token)
            talker.generate = self._wrap_generate(original_generate, token)
            worker = threading.Thread(
                target=self._produce,
                args=(spec, sink),
                name=f"tts-stream-{spec.request_id or 'anon'}",
                daemon=True,
            )
            worker.start()
            yield from self._consume(sink, token, eos, per_chunk, samples_per_code)
        finally:
            token.cancel()
            _drain(sink)      # a producer parked in put() gets its room back
            if worker is not None:
                worker.join(timeout=WORKER_JOIN_TIMEOUT_S)
                if worker.is_alive():
                    # Restoring under a live wrapper is still better than
                    # leaving one installed, but the engine must not wrap again.
                    self._degraded = "worker_stuck"
                    log.error("streaming worker did not stop; disabling streaming")
            _restore(talker, "forward", original_forward, had_forward)
            _restore(talker, "generate", original_generate, had_generate)

    def _produce(self, spec: StreamSpec, sink: queue.Queue) -> None:
        """Run the blocking generation; the tap does the reporting."""
        try:
            self._wrapper.generate_custom_voice(
                text=spec.text, language=spec.language, speaker=spec.speaker
            )
        except BaseException as exc:      # noqa: BLE001 — carried, not swallowed
            sink.put(("error", type(exc).__name__))
            return
        sink.put(("done", None))

    def _wrap_forward(self, original: Any, sink: queue.Queue, token: CancelToken) -> Any:
        """See each step's codes without changing what the talker computes.

        Signature-preserving on purpose: transformers decides which model_kwargs
        are legal by inspecting forward()'s signature, and a bare
        (*args, **kwargs) wrapper makes generate() reject trailing_text_hidden,
        tts_pad_embed and every subtalker_* option.
        """

        @functools.wraps(original)
        def _tap(*args: Any, **kwargs: Any) -> Any:
            out = original(*args, **kwargs)
            if token.cancelled:
                return out
            states = getattr(out, "hidden_states", None)
            if states:
                step = states[-1]
                if step is not None:
                    _offer(sink, ("codes", step.detach().to("cpu")), token)
            return out

        _tap._kiki_stream_hook = True
        return _tap

    def _wrap_generate(self, original: Any, token: CancelToken) -> Any:
        """Inject the cancel criterion, keeping whatever the caller passed."""

        @functools.wraps(original)
        def _generate(*args: Any, **kwargs: Any) -> Any:
            existing = kwargs.get("stopping_criteria")
            criteria = list(existing) if existing else []
            criteria.append(_CancelStop(token))
            kwargs["stopping_criteria"] = criteria
            return original(*args, **kwargs)

        _generate._kiki_stream_hook = True
        return _generate

    def _consume(
        self,
        sink: queue.Queue,
        token: CancelToken,
        eos: int,
        per_chunk: int,
        samples_per_code: int,
    ) -> Iterator[bytes]:
        """Turn arriving codes into PCM, one chunk at a time.

        Only codes are held, never audio: sixteen small integers per 80 ms, so a
        forty-second answer costs a few kilobytes. Each PCM chunk is yielded the
        moment it exists and is never accumulated.
        """
        collected: list[Any] = []
        emitted = 0
        checked_shape = False
        while True:
            if token.cancelled:
                return
            try:
                kind, payload = sink.get(timeout=0.25)
            except queue.Empty:
                continue
            if kind == "error":
                raise StreamingEngineError(f"generation failed: {payload}")
            if kind == "done":
                break
            if not checked_shape:
                if not _looks_like_codes(payload):
                    self._degraded = "hidden_states_shape"
                    raise EngineUnavailable("hidden_states_shape")
                checked_shape = True
            collected.append(payload)
            if int(payload.reshape(-1)[0]) == eos:
                collected.pop()
                break
            while len(collected) - emitted >= per_chunk:
                if token.cancelled:
                    return
                end = emitted + per_chunk
                pcm = self._decode(collected, emitted, end, samples_per_code)
                emitted = end
                if pcm:
                    yield pcm
        if not token.cancelled and len(collected) > emitted:
            pcm = self._decode(collected, emitted, len(collected), samples_per_code)
            if pcm:
                yield pcm

    def _decode(self, collected: list[Any], start: int, end: int, samples_per_code: int) -> bytes:
        """Decode codes[start:end] with left context, the library's own way.

        `speech_tokenizer.decode` routes through `chunked_decode`, which handles
        the causal context inside the window. The same arithmetic is applied to
        the growing prefix here — decode `[start-ctx:end]`, drop the first
        `ctx * total_upsample` samples — so no crossfade is invented.
        """
        context = min(self._left_context, start)
        window = _stack_codes([step.reshape(-1) for step in collected[start - context : end]])
        wavs, _rate = self._model.speech_tokenizer.decode([{"audio_codes": window}])
        audio = wavs[0].reshape(-1)[context * samples_per_code :]
        pcm = float_to_pcm16(audio)
        # Whole samples only; an odd tail would be half a sample on the wire.
        return pcm[: len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)]


class StreamingEngineError(Exception):
    """The generation failed for a reason that is not a cancel."""


def _stack_codes(steps: list[Any]) -> Any:
    """Build the (T, Q) tensor `decode()` wants.

    The one place torch is needed, and the guard refuses a runtime without it —
    so the list fallback below is only ever reached by the contract tests, which
    hand a fake tokenizer something it can measure.
    """
    try:
        import torch
    except ImportError:
        return steps
    return torch.stack(steps, dim=0)


def _restore(owner: Any, name: str, original: Any, had_own: bool) -> None:
    """Put the method back exactly as it was, leaving no residue."""
    if had_own:
        setattr(owner, name, original)
        return
    owner.__dict__.pop(name, None)


def _offer(sink: queue.Queue, item: Any, token: CancelToken) -> None:
    """Hand one item over, waiting only as long as the request is alive.

    Blocking outright would park the talker forever once the consumer is gone;
    that is exactly the deadlock the bounded queue exists to make visible.
    """
    while not token.cancelled:
        try:
            sink.put(item, timeout=PUT_TIMEOUT_S)
            return
        except queue.Full:
            continue


def _drain(sink: queue.Queue) -> None:
    while True:
        try:
            sink.get_nowait()
        except queue.Empty:
            return


def _looks_like_codes(step: Any) -> bool:
    """One decoding step must be a small tensor of codec indices."""
    reshape = getattr(step, "reshape", None)
    if not callable(reshape):
        return False
    try:
        flat = step.reshape(-1)
        return len(flat) > 0
    except Exception:
        return False


# --- manual smoke / benchmark ------------------------------------------------
#
# Never reached by the service or by any test: this is the separate, hand-run
# proof that the engine works against the real model.
#
#     python streaming_engine.py --smoke
#     python streaming_engine.py --bench


BENCH_TEXTS: dict[str, str] = {
    "kurz": "Guten Abend Martin, die Engine steht.",
    "zwei": (
        "Ich habe die Streaming-Engine eingebaut. Sie liefert PCM, während das "
        "Modell noch rechnet."
    ),
    "umlaute": "Größe, Höhe, Maß — 24.000 Hertz, mono, sechzehn Bit, 3,5 Prozent.",
}


def _vram_mib() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        return 0.0


def _measure(engine: QwenStreamingEngine, text: str, *, chunk_ms: int = 400,
             cancel_after: float | None = None) -> dict:
    """One run, measured. `cancel_after` fires on a timer so it can land either
    side of the first chunk — inside the loop it could only ever land after."""
    import time

    spec = StreamSpec(text=text, language="German", speaker="Serena", chunk_ms=chunk_ms)
    token = CancelToken()
    talker = engine._model.talker
    before = _vram_mib()
    peak = before
    sizes: list[int] = []
    ttfa = None
    cancel_at: list[float] = []
    timer = None
    if cancel_after is not None:
        def _fire() -> None:
            cancel_at.append(time.perf_counter())
            token.cancel()

        timer = threading.Timer(cancel_after, _fire)
        timer.start()
    began = time.perf_counter()
    for block in engine.stream(spec, token):
        if ttfa is None:
            ttfa = time.perf_counter() - began
        sizes.append(len(block))
        peak = max(peak, _vram_mib())
    total = time.perf_counter() - began
    if timer is not None:
        timer.cancel()
    audio = sum(sizes) / BYTES_PER_SAMPLE / STREAM_SAMPLE_RATE
    return {
        "ttfa_s": round(ttfa, 3) if ttfa else None,
        "total_s": round(total, 3),
        "audio_s": round(audio, 3),
        "rtf": round(total / audio, 3) if audio else None,
        "chunks": len(sizes),
        "chunk_bytes": sorted(set(sizes)),
        "cancel_latency_s": (
            round(began + total - cancel_at[0], 3) if cancel_at else None
        ),
        "hooks_left": hooks_installed(talker),
        "vram_mib": {
            "before": round(before, 1),
            "peak": round(peak, 1),
            "after": round(_vram_mib(), 1),
        },
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="Manual smoke test for the real engine")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-ms", type=int, default=400)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--repeat", type=int, default=10, help="VRAM trend over N runs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from kiki_tts_server import QwenSynthesizer

    synth = QwenSynthesizer(args.model, args.device)
    engine = QwenStreamingEngine(synth)
    print(_json.dumps({"available": engine.available, "reason": engine.reason}))
    if not engine.available:
        return 1

    if args.smoke:
        print(_json.dumps(_measure(engine, BENCH_TEXTS["kurz"], chunk_ms=args.chunk_ms),
                          ensure_ascii=False, indent=2))
        return 0

    if args.bench:
        for key, text in BENCH_TEXTS.items():
            result = _measure(engine, text, chunk_ms=args.chunk_ms)
            print(_json.dumps({"text": key, **result}, ensure_ascii=False))
        for label, after in (("cancel_vor_erstem", 0.3), ("cancel_nach_erstem", 2.0)):
            result = _measure(engine, BENCH_TEXTS["zwei"], chunk_ms=args.chunk_ms,
                              cancel_after=after)
            print(_json.dumps({"text": label, **result}, ensure_ascii=False))
        trend = []
        for _ in range(args.repeat):
            _measure(engine, BENCH_TEXTS["kurz"], chunk_ms=args.chunk_ms)
            trend.append(round(_vram_mib(), 1))
        print(_json.dumps({"vram_trend_mib": trend}))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
