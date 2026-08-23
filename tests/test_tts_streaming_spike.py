"""Contract tests for the streaming spike. No GPU, no model, no torch.

The spike itself can only be measured on the machine with the model. What can
be pinned everywhere is its arithmetic: how a requested chunk length becomes a
number of codec tokens, and how model floats become the PCM16LE bytes the
future HTTP contract promises. Both are the pieces a production endpoint would
inherit unchanged.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPIKE = ROOT / "services" / "qwen3-tts" / "streaming_spike.py"


def _load():
    spec = importlib.util.spec_from_file_location("streaming_spike", SPIKE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so the module has to
    # be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


spike = _load()


# --- the module must stay light --------------------------------------------


def test_importing_the_spike_loads_no_heavy_runtime() -> None:
    """Torch, CUDA and the model belong behind function calls: the contract
    tests must run on a machine that has none of them."""
    for name in ("torch", "qwen_tts", "numpy", "soundfile"):
        assert name not in sys.modules, name


def test_the_spike_is_not_reachable_from_the_application() -> None:
    """A spike that production code imports is no longer a spike."""
    hits = [
        path
        for path in (ROOT / "src").rglob("*.py")
        if "streaming_spike" in path.read_text(encoding="utf-8")
    ]
    assert hits == []
    server = (ROOT / "services" / "qwen3-tts" / "kiki_tts_server.py").read_text(encoding="utf-8")
    assert "streaming_spike" not in server


# --- chunk arithmetic -------------------------------------------------------


def test_one_code_is_eighty_milliseconds() -> None:
    """1920 samples at 24 kHz. The whole chunk plan rests on this."""
    assert spike.EXPECTED_UPSAMPLE / spike.EXPECTED_RATE == pytest.approx(0.08)


@pytest.mark.parametrize(
    ("chunk_ms", "codes"),
    [
        (160, 2),      # exactly two codes
        (200, 3),      # 2.5 rounds up: a short chunk must not vanish
        (400, 5),      # the default
        (1000, 13),    # 12.5 rounds up
    ],
)
def test_chunk_length_becomes_whole_codes(chunk_ms, codes) -> None:
    assert spike.codes_per_chunk(chunk_ms, 1920, 24_000) == codes


@pytest.mark.parametrize("chunk_ms", [0, 159, 1001, -400])
def test_a_chunk_length_outside_the_contract_is_refused(chunk_ms) -> None:
    with pytest.raises(ValueError):
        spike.codes_per_chunk(chunk_ms, 1920, 24_000)


def test_the_planned_chunk_is_never_shorter_than_one_code() -> None:
    """A hypothetical slower codec must still produce something per chunk."""
    assert spike.codes_per_chunk(160, 24_000, 24_000) == 1


# --- PCM conversion ---------------------------------------------------------


def test_pcm_is_two_bytes_per_sample() -> None:
    pcm = spike.float_to_pcm16([0.0] * 100)
    assert len(pcm) == 200
    assert len(pcm) % 2 == 0


def test_pcm_is_little_endian_signed_16_bit() -> None:
    pcm = spike.float_to_pcm16([0.0, 0.5, -0.5])
    values = struct.unpack("<3h", pcm)
    assert values[0] == 0
    assert values[1] == pytest.approx(16383, abs=1)
    assert values[2] == pytest.approx(-16383, abs=1)


def test_full_scale_never_wraps_to_the_negative_rail() -> None:
    """Scaling by 32768 would turn +1.0 into -32768 — an audible click at the
    loudest moment of the utterance."""
    values = struct.unpack("<2h", spike.float_to_pcm16([1.0, -1.0]))
    assert values == (32767, -32767)


def test_values_beyond_the_rails_are_clamped_not_wrapped() -> None:
    values = struct.unpack("<4h", spike.float_to_pcm16([1.4, -1.4, 12.0, -12.0]))
    assert values == (32767, -32767, 32767, -32767)


def test_a_chunk_always_holds_whole_samples() -> None:
    """The HTTP contract says every chunk carries complete 16-bit samples."""
    for count in (1, 2, 3, 9599, 9600):
        assert len(spike.float_to_pcm16([0.1] * count)) == count * 2


def test_the_default_chunk_is_exactly_the_promised_byte_count() -> None:
    """400 ms mono PCM16 at 24 kHz is 19200 bytes — the number the measurement
    run reported for every chunk."""
    codes = spike.codes_per_chunk(400, 1920, 24_000)
    samples = codes * 1920
    assert len(spike.float_to_pcm16([0.0] * samples)) == 19_200


def test_a_nested_shape_is_flattened_like_the_decoder_output() -> None:
    class _FakeTensor:
        """Stands in for the numpy array the decoder returns."""

        def __init__(self, values):
            self._values = values

        def reshape(self, _shape):
            return self

        def tolist(self):
            return self._values

    pcm = spike.float_to_pcm16(_FakeTensor([0.0, 1.0]))
    assert struct.unpack("<2h", pcm) == (0, 32767)


# --- the documented contract constants --------------------------------------


def test_the_chunk_bounds_match_the_intended_http_contract() -> None:
    assert spike.MIN_CHUNK_MS == 160
    assert spike.DEFAULT_CHUNK_MS == 400
    assert spike.MAX_CHUNK_MS == 1000


def test_the_code_queue_is_bounded() -> None:
    """An unbounded tap would let the talker outrun the decoder and hold the
    whole answer in memory — the thing streaming is meant to avoid."""
    assert 0 < spike.CODE_QUEUE_LIMIT < 10_000


def test_the_left_context_default_follows_the_library() -> None:
    """chunked_decode() uses 25, and the sweep showed it costs no measurable
    time, so there is no reason to deviate."""
    assert spike.DEFAULT_LEFT_CONTEXT == 25
