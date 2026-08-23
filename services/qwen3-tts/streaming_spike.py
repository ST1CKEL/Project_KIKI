#!/usr/bin/env python3
"""Spike: can Qwen3-TTS 12Hz emit PCM *while* it generates?

Not production code. Nothing here is imported by the service, the app or any
production test — it exists to answer one question with measurements instead of
claims, and to be deleted or promoted afterwards.

Why an adapter is needed at all
-------------------------------
`Qwen3TTSModel.generate_custom_voice()` runs in two stages: the talker emits
codec tokens, then the speech tokenizer decodes *all* of them into a waveform.
Nothing is audible before the second stage, which is where the measured 4-5 s
time-to-first-audio comes from.

The installed library (qwen_tts, transformers 4.57.3) offers no way in:

* no `streamer` support anywhere in the package;
* `Qwen3TTSForConditionalGeneration.generate` builds a closed `talker_kwargs`
  dict and does **not** forward `**kwargs`, so `stopping_criteria` cannot be
  passed through either;
* its own docstring says `non_streaming_mode` only simulates streaming text
  input, "rather than enabling true streaming input or streaming generation".

What it *does* offer is the decoder half. The 12 Hz tokenizer is built from
causal convolutions and ships `chunked_decode(codes, chunk_size, left_context)`,
which decodes a slice with left context and then discards exactly
`context * total_upsample` leading samples. That is the boundary problem already
solved by the library — no overlap-add or crossfade of our own.

So this spike wraps one method — the talker's `forward` — to observe each
decoding step as it happens, and reuses the library's own left-context recipe to
turn the growing code sequence into PCM. Both are reversible and confined to
this file.

Geometry (from the shipped config, verified at runtime by --report):
    upsample_rates (8,5,4,3) x upsampling_ratios (2,2) -> total_upsample 1920
    24000 Hz / 1920 = 12.5 codes per second, i.e. one code is 80 ms of audio.

Usage:
    python streaming_spike.py --report
    python streaming_spike.py --text "Hallo Martin." --chunk-ms 400
    python streaming_spike.py --bench
    python streaming_spike.py --cancel-after 0.5
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import queue
import statistics
import sys
import threading
import time
import traceback
from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

log = logging.getLogger("tts-spike")

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_SPEAKER = "Serena"
DEFAULT_LANGUAGE = "German"

# One code is 1920 samples at 24 kHz. Asserted against the loaded model.
EXPECTED_UPSAMPLE = 1920
EXPECTED_RATE = 24_000

MIN_CHUNK_MS = 160
MAX_CHUNK_MS = 1000
DEFAULT_CHUNK_MS = 400
# What chunked_decode() uses. Smaller values are measured by --context-sweep.
DEFAULT_LEFT_CONTEXT = 25
# Bounded: the tap must never let the talker run ahead without limit.
CODE_QUEUE_LIMIT = 256

BENCH_TEXTS: dict[str, str] = {
    "kurz": "Guten Abend Martin, der Streaming-Pfad steht jetzt bereit.",
    "normal": (
        "Ich habe den neuen Endpunkt eingerichtet. Er liefert die ersten "
        "Audiodaten, während das Modell noch weiterrechnet."
    ),
    "umlaute": (
        "Die Übertragung läuft über 24.000 Hertz, mono, sechzehn Bit. "
        "Größe, Höhe, Maß — und 3,5 Prozent Abweichung."
    ),
    "lang": (
        "Ich fasse den Stand zusammen. Der bisherige Dienst erzeugt zuerst die "
        "vollständige Wellenform und antwortet erst danach, weshalb die Zeit bis "
        "zum ersten Ton bei etwa vier bis fünf Sekunden liegt. Der neue Weg "
        "greift die Codes ab, während der Talker sie erzeugt, und dekodiert sie "
        "abschnittsweise mit linkem Kontext. Dadurch kann die Wiedergabe "
        "beginnen, sobald der erste Abschnitt fertig ist. Die Tonqualität bleibt "
        "unverändert, weil der Dekoder kausal arbeitet und die Bibliothek den "
        "linken Kontext bereits selbst vorsieht. Was bleibt, ist die Frage nach "
        "dem zusätzlichen Rechenaufwand, denn jeder Abschnitt dekodiert seinen "
        "Kontext erneut mit."
    ),
}


class SpikeCancelled(Exception):
    """Raised inside the tap to unwind generate() at a step boundary."""


@dataclass
class ChunkStat:
    index: int
    codes: int
    pcm_bytes: int
    at_s: float
    decode_s: float


@dataclass
class StreamReport:
    text_key: str = ""
    chunk_ms: int = DEFAULT_CHUNK_MS
    left_context: int = DEFAULT_LEFT_CONTEXT
    ttfa_s: float | None = None
    total_s: float = 0.0
    audio_s: float = 0.0
    total_codes: int = 0
    chunks: list[ChunkStat] = field(default_factory=list)
    cancelled: bool = False
    cancel_latency_s: float | None = None
    vram_before_mib: float = 0.0
    vram_peak_mib: float = 0.0
    vram_after_mib: float = 0.0

    @property
    def rtf(self) -> float | None:
        return self.total_s / self.audio_s if self.audio_s > 0 else None

    def as_dict(self) -> dict:
        sizes = [c.pcm_bytes for c in self.chunks]
        decodes = [c.decode_s for c in self.chunks]
        return {
            "text": self.text_key,
            "chunk_ms": self.chunk_ms,
            "left_context_codes": self.left_context,
            "ttfa_s": round(self.ttfa_s, 3) if self.ttfa_s else None,
            "total_s": round(self.total_s, 3),
            "audio_s": round(self.audio_s, 3),
            "rtf": round(self.rtf, 3) if self.rtf else None,
            "codes": self.total_codes,
            "chunks": len(self.chunks),
            "chunk_bytes_min_med_max": (
                [min(sizes), int(statistics.median(sizes)), max(sizes)] if sizes else []
            ),
            "decode_ms_min_med_max": (
                [round(min(decodes) * 1000, 1), round(statistics.median(decodes) * 1000, 1),
                 round(max(decodes) * 1000, 1)] if decodes else []
            ),
            "cancelled": self.cancelled,
            "cancel_latency_s": (
                round(self.cancel_latency_s, 3) if self.cancel_latency_s else None
            ),
            "vram_mib": {
                "before": round(self.vram_before_mib, 1),
                "peak": round(self.vram_peak_mib, 1),
                "after": round(self.vram_after_mib, 1),
            },
        }


def codes_per_chunk(chunk_ms: int, samples_per_code: int, sample_rate: int) -> int:
    """How many codec tokens make up one requested chunk length.

    Rounded up and floored at one: a chunk shorter than a single code cannot
    exist, and rounding down would make chunk_ms=160 produce nothing at all.
    """
    if not MIN_CHUNK_MS <= chunk_ms <= MAX_CHUNK_MS:
        raise ValueError(f"chunk_ms muss zwischen {MIN_CHUNK_MS} und {MAX_CHUNK_MS} liegen")
    per_code_ms = samples_per_code * 1000 / sample_rate
    return max(1, int(-(-chunk_ms // per_code_ms)))


def float_to_pcm16(samples) -> bytes:
    """Clamp and convert model float output to little-endian PCM16.

    Stdlib only, so the conversion can be tested wherever the project runs and
    not just inside the TTS virtualenv. Scaling by 32767 rather than 32768 keeps
    a sample sitting exactly at 1.0 from wrapping to -32768.
    """
    flat = samples.reshape(-1).tolist() if hasattr(samples, "reshape") else list(samples)
    out = array("h", (max(-32767, min(32767, int(value * 32767.0))) for value in flat))
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def _vram_mib() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        return 0.0


class StreamingSpike:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "auto") -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        log.info("loading %s on %s", model_id, device)
        self.device = device
        self.wrapper = Qwen3TTSModel.from_pretrained(
            model_id, device_map=device, dtype=dtype, attn_implementation="sdpa"
        )
        self.model = self.wrapper.model
        self.tokenizer = self.model.speech_tokenizer
        self.eos = self.model.config.talker_config.codec_eos_token_id
        decoder = self.tokenizer.model.decoder
        self.samples_per_code = int(decoder.total_upsample)
        self.sample_rate = int(self.tokenizer.model.get_output_sample_rate())

    def geometry(self) -> dict:
        return {
            "samples_per_code": self.samples_per_code,
            "sample_rate": self.sample_rate,
            "code_ms": round(self.samples_per_code * 1000 / self.sample_rate, 2),
            "matches_expectation": (
                self.samples_per_code == EXPECTED_UPSAMPLE
                and self.sample_rate == EXPECTED_RATE
            ),
        }

    @contextmanager
    def _tap(self, sink: queue.Queue, cancel: threading.Event | None):
        """Observe every talker step without changing what it computes.

        `generate()` collects each step's codes in `hidden_states[-1]` and only
        assembles them once generation is over. Wrapping `forward` is the
        narrowest place to see them arrive. The wrapper adds no tensors and no
        gradients; it reads one entry and puts it on a bounded queue.
        """
        talker = self.model.talker
        original = talker.forward

        # Signature-preserving on purpose: transformers decides which
        # model_kwargs are legal by inspecting forward()'s signature, and a bare
        # (*args, **kwargs) wrapper makes it reject trailing_text_hidden,
        # tts_pad_embed and every subtalker_* option.
        @functools.wraps(original)
        def _wrapped(*args, **kwargs):
            if cancel is not None and cancel.is_set():
                raise SpikeCancelled
            out = original(*args, **kwargs)
            states = getattr(out, "hidden_states", None)
            if states:
                step = states[-1]
                if step is not None:
                    # Bounded on purpose: a full queue means the decoder cannot
                    # keep up, and blocking the talker is the correct answer.
                    sink.put(("codes", step.detach().to("cpu")))
            return out

        talker.forward = _wrapped
        try:
            yield
        finally:
            talker.forward = original

    def stream(
        self,
        text: str,
        *,
        language: str = DEFAULT_LANGUAGE,
        speaker: str = DEFAULT_SPEAKER,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        left_context: int = DEFAULT_LEFT_CONTEXT,
        cancel: threading.Event | None = None,
        report: StreamReport | None = None,
    ) -> Iterator[bytes]:
        """Yield PCM16LE while the talker is still generating."""
        report = report if report is not None else StreamReport()
        report.chunk_ms = chunk_ms
        report.left_context = left_context
        per_chunk = codes_per_chunk(chunk_ms, self.samples_per_code, self.sample_rate)

        sink: queue.Queue = queue.Queue(maxsize=CODE_QUEUE_LIMIT)
        started = time.perf_counter()
        report.vram_before_mib = _vram_mib()

        def _run() -> None:
            try:
                with self._tap(sink, cancel):
                    self.wrapper.generate_custom_voice(
                        text=text, language=language, speaker=speaker
                    )
            except SpikeCancelled:
                sink.put(("cancelled", None))
                return
            except BaseException as exc:  # reported, never re-raised across threads
                # A spike is a diagnostic tool: the traceback is the point.
                sink.put(("error", (exc, traceback.format_exc())))
                return
            sink.put(("done", None))

        worker = threading.Thread(target=_run, name="spike-talker", daemon=True)
        worker.start()

        collected: list = []
        emitted_codes = 0
        peak = report.vram_before_mib
        try:
            while True:
                kind, payload = sink.get()
                if kind in {"done", "cancelled", "error"}:
                    if kind == "cancelled":
                        report.cancelled = True
                    elif kind == "error":
                        exc, tb = payload
                        raise RuntimeError(f"Generation fehlgeschlagen: {exc}\n{tb}")
                    break
                collected.append(payload)
                peak = max(peak, _vram_mib())
                # The talker signals the end of speech in codebook 0.
                if int(payload.reshape(-1)[0]) == self.eos:
                    collected.pop()
                    break
                while len(collected) - emitted_codes >= per_chunk:
                    end = emitted_codes + per_chunk
                    pcm, decode_s = self._decode_slice(
                        collected, emitted_codes, end, left_context
                    )
                    emitted_codes = end
                    if report.ttfa_s is None:
                        report.ttfa_s = time.perf_counter() - started
                    report.chunks.append(
                        ChunkStat(len(report.chunks), per_chunk, len(pcm),
                                  time.perf_counter() - started, decode_s)
                    )
                    yield pcm
                    if cancel is not None and cancel.is_set():
                        report.cancelled = True
                        break
                if report.cancelled:
                    break
            # tail
            if not report.cancelled and len(collected) > emitted_codes:
                pcm, decode_s = self._decode_slice(
                    collected, emitted_codes, len(collected), left_context
                )
                if report.ttfa_s is None:
                    report.ttfa_s = time.perf_counter() - started
                report.chunks.append(
                    ChunkStat(len(report.chunks), len(collected) - emitted_codes,
                              len(pcm), time.perf_counter() - started, decode_s)
                )
                emitted_codes = len(collected)
                yield pcm
        finally:
            if cancel is not None:
                cancel.set()          # unblock the talker if we left early
            worker.join(timeout=30)
            report.total_s = time.perf_counter() - started
            report.total_codes = emitted_codes
            report.audio_s = emitted_codes * self.samples_per_code / self.sample_rate
            report.vram_peak_mib = peak
            try:
                import torch as _t

                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
            except Exception:
                pass
            report.vram_after_mib = _vram_mib()

    def _decode_slice(self, collected, start: int, end: int, left_context: int):
        """Decode codes[start:end] with left context, the way the library does.

        `chunked_decode` decodes `codes[start-ctx:end]` and throws away the first
        `ctx * total_upsample` samples. Doing the same on a growing prefix gives
        the identical samples without waiting for the whole sequence.
        """
        import torch

        context = min(left_context, start)
        # (T, Q): each tapped step is (batch=1, num_quantizers); decode()
        # pads a list of (T, Q) into (1, T, Q) and transposes to (1, Q, T).
        window = torch.stack(
            [step.reshape(-1) for step in collected[start - context : end]], dim=0
        )
        began = time.perf_counter()
        wavs, _sr = self.tokenizer.decode([{"audio_codes": window}])
        audio = wavs[0].reshape(-1)[context * self.samples_per_code :]
        return float_to_pcm16(audio), time.perf_counter() - began

    def decode_all(self, text: str, *, language: str = DEFAULT_LANGUAGE,
                   speaker: str = DEFAULT_SPEAKER) -> tuple[bytes, float]:
        """The current production path, for comparison."""
        began = time.perf_counter()
        wavs, _sr = self.wrapper.generate_custom_voice(
            text=text, language=language, speaker=speaker
        )
        return float_to_pcm16(wavs[0]), time.perf_counter() - began


def _print(report: StreamReport) -> None:
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text", default=BENCH_TEXTS["kurz"])
    parser.add_argument("--chunk-ms", type=int, default=DEFAULT_CHUNK_MS)
    parser.add_argument("--left-context", type=int, default=DEFAULT_LEFT_CONTEXT)
    parser.add_argument("--cancel-after", type=float, default=0.0)
    parser.add_argument("--report", action="store_true", help="geometry only, no synthesis")
    parser.add_argument("--bench", action="store_true", help="all benchmark texts")
    parser.add_argument("--compare", action="store_true", help="streamed vs one-shot audio")
    parser.add_argument("--context-sweep", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    spike = StreamingSpike(args.model, args.device)
    print(json.dumps({"geometry": spike.geometry()}, indent=2))
    if args.report:
        return 0

    if args.compare:
        return _compare(spike, args)
    if args.context_sweep:
        return _context_sweep(spike, args)

    texts = BENCH_TEXTS if args.bench else {"cli": args.text}
    for key, text in texts.items():
        cancel = threading.Event()
        report = StreamReport(text_key=key)
        if args.cancel_after > 0:
            def _fire(delay=args.cancel_after, ev=cancel):
                time.sleep(delay)
                ev.set()

            threading.Thread(target=_fire, daemon=True).start()
        total = 0
        for pcm in spike.stream(
            text, chunk_ms=args.chunk_ms, left_context=args.left_context,
            cancel=cancel if args.cancel_after > 0 else None, report=report,
        ):
            assert len(pcm) % 2 == 0, "halbes Sample im Chunk"
            total += len(pcm)
        if args.cancel_after > 0 and report.cancelled:
            report.cancel_latency_s = report.total_s - args.cancel_after
        _print(report)
    return 0


def _compare(spike: StreamingSpike, args) -> int:
    """Does chunked decoding change the audio? Numbers, not opinions."""
    import numpy as np

    report = StreamReport(text_key="compare")
    streamed = b"".join(
        spike.stream(args.text, chunk_ms=args.chunk_ms,
                     left_context=args.left_context, report=report)
    )
    one_shot, one_shot_s = spike.decode_all(args.text)
    a = np.frombuffer(streamed, dtype="<i2").astype(np.float32)
    b = np.frombuffer(one_shot, dtype="<i2").astype(np.float32)
    n = min(len(a), len(b))
    print(json.dumps({
        "streamed": report.as_dict(),
        "one_shot_s": round(one_shot_s, 3),
        "samples_streamed": len(a),
        "samples_one_shot": len(b),
        # Two runs sample independently, so this compares *shape*, not identity.
        "note": "do_sample=True: different runs differ by design",
        "max_abs_diff_int16": float(np.abs(a[:n] - b[:n]).max()) if n else None,
    }, ensure_ascii=False, indent=2))
    return 0


def _context_sweep(spike: StreamingSpike, args) -> int:
    """How much left context does the causal decoder actually need?"""
    for ctx in (0, 2, 5, 10, 25):
        report = StreamReport(text_key=f"ctx={ctx}")
        for _pcm in spike.stream(args.text, chunk_ms=args.chunk_ms,
                                 left_context=ctx, report=report):
            pass
        _print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
