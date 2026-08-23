"""The real streaming engine, exercised without a model, torch or a GPU.

Everything here runs against fakes for the talker, the tokenizer and the
decoder. What is under test is the part that can silently rot: the runtime
guard, the lifetime of the two wrapped methods, and a cancel that is a flag
rather than an exception.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "services" / "qwen3-tts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SERVICE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sh = _load("streaming_http")
se = _load("streaming_engine")


# --- fakes ------------------------------------------------------------------


class FakeCodes:
    """Stands in for one decoding step's code tensor."""

    def __init__(self, values):
        self._values = list(values)

    def reshape(self, _shape):
        return self

    def detach(self):
        return self

    def to(self, _device):
        return self

    def tolist(self):
        return list(self._values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]


class FakeTalker:
    """Only what the engine touches: forward, generate, and a step counter."""

    def __init__(self, *, steps: int = 10, eos: int = 999, delay_s: float = 0.0):
        self._steps = steps
        self._eos = eos
        self._delay = delay_s
        self.generate_calls: list[dict] = []
        self.forward_calls = 0
        self.stopped_early = False

    def forward(self, step_index: int = 0, **_kwargs):
        self.forward_calls += 1
        codes = [self._eos] * 16 if step_index >= self._steps else [step_index + 1] * 16

        class _Out:
            hidden_states = (None, FakeCodes(codes))

        return _Out()

    def generate(self, **kwargs):
        """A miniature of what transformers does: step, then ask the criteria."""
        self.generate_calls.append(kwargs)
        criteria = kwargs.get("stopping_criteria") or []
        for index in range(self._steps + 1):
            if self._delay:
                time.sleep(self._delay)
            self.forward(step_index=index)
            if any(bool(check(_FakeIds())) for check in criteria):
                self.stopped_early = True
                return "stopped"
        return "done"


class _FakeIds:
    shape = (1,)


class FakeDecoder:
    total_upsample = se.EXPECTED_UPSAMPLE

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):  # noqa: ARG002
        return codes


class FakeTokenizerModel:
    def __init__(self, rate: int = 24_000):
        self.decoder = FakeDecoder()
        self._rate = rate

    def get_output_sample_rate(self):
        return self._rate


class FakeTokenizer:
    """Returns one float sample per code, so byte counts stay predictable."""

    def __init__(self, rate: int = 24_000):
        self.model = FakeTokenizerModel(rate)
        self.windows: list[int] = []

    def decode(self, encoded):
        window = encoded[0]["audio_codes"]
        rows = len(window)
        self.windows.append(rows)
        return [_FakeAudio([0.5] * (rows * se.EXPECTED_UPSAMPLE))], 24_000


class _FakeAudio:
    def __init__(self, values):
        self._values = values

    def reshape(self, _shape):
        return self

    def tolist(self):
        return list(self._values)

    def __getitem__(self, item):
        return _FakeAudio(self._values[item])


class FakeTalkerConfig:
    def __init__(self, eos: int = 999):
        # Per instance, not per class: one test sets this to None, and a shared
        # object would poison every test that ran after it.
        self.codec_eos_token_id = eos


class FakeConfig:
    def __init__(self):
        self.talker_config = FakeTalkerConfig()


class FakeModel:
    def __init__(self, talker=None, tokenizer=None, config=None):
        self.talker = talker if talker is not None else FakeTalker()
        self.speech_tokenizer = tokenizer if tokenizer is not None else FakeTokenizer()
        self.config = config if config is not None else FakeConfig()


_AUTO = object()


class FakeWrapper:
    """The Qwen3TTSModel wrapper: owns the model and the blocking entry point."""

    def __init__(self, model=_AUTO, *, fail: BaseException | None = None):
        self.model = FakeModel() if model is _AUTO else model
        self._fail = fail
        self.calls: list[dict] = []

    def generate_custom_voice(self, *, text, language, speaker):
        self.calls.append({"text": text, "language": language, "speaker": speaker})
        if self._fail is not None:
            raise self._fail
        return self.model.talker.generate()


class FakeSynth:
    def __init__(self, wrapper=None):
        self.model = wrapper if wrapper is not None else FakeWrapper()


@pytest.fixture(autouse=True)
def _real_versions(monkeypatch):
    """The guard reads installed package versions; the fakes are not packages."""
    monkeypatch.setattr(
        se,
        "_version",
        lambda name: {
            "qwen-tts": (0, 1, 1),
            "transformers": (4, 57, 3),
            "torch": (2, 11, 0),
        }.get(name),
    )


def _spec(**kwargs):
    values = {"text": "Hallo Martin.", "language": "German", "speaker": "Serena"}
    values.update(kwargs)
    return sh.StreamSpec(**values)


def _drain(engine, spec=None, token=None, stop_after=None):
    token = token or sh.CancelToken()
    out = []
    for block in engine.stream(spec or _spec(), token):
        out.append(block)
        if stop_after is not None and len(out) >= stop_after:
            token.cancel()
    return out


# --- the runtime guard ------------------------------------------------------


def test_a_matching_runtime_passes() -> None:
    assert se.check_runtime(FakeModel()).ok is True


@pytest.mark.parametrize(
    ("versions", "reason"),
    [
        ({}, "qwen_tts_missing"),
        ({"qwen-tts": (0, 2, 0), "transformers": (4, 57, 3)}, "qwen_tts_version"),
        ({"qwen-tts": (0, 1, 0), "transformers": (4, 57, 3)}, "qwen_tts_version"),
        ({"qwen-tts": (0, 1, 1)}, "transformers_missing"),
        ({"qwen-tts": (0, 1, 1), "transformers": (4, 57, 3)}, "torch_missing"),
        (
            {"qwen-tts": (0, 1, 1), "transformers": (4, 57, 3), "torch": (1, 13, 0)},
            "torch_version",
        ),
        ({"qwen-tts": (0, 1, 1), "transformers": (4, 56, 0)}, "transformers_version"),
        ({"qwen-tts": (0, 1, 1), "transformers": (5, 0, 0)}, "transformers_version"),
    ],
)
def test_a_version_outside_the_tested_range_is_refused(monkeypatch, versions, reason) -> None:
    monkeypatch.setattr(se, "_version", lambda name: versions.get(name))
    report = se.check_runtime(FakeModel())
    assert report.ok is False
    assert report.reason == reason


def test_each_internal_is_checked_on_its_own() -> None:
    """The tap depends on private structure; every piece of it gets a name."""
    cases = {
        "no_talker": lambda m: setattr(m, "talker", None),
        "talker_forward": lambda m: setattr(m.talker, "forward", None),
        "talker_generate": lambda m: setattr(m.talker, "generate", None),
        "no_speech_tokenizer": lambda m: setattr(m, "speech_tokenizer", None),
        "tokenizer_decode": lambda m: setattr(m.speech_tokenizer, "decode", None),
        "decoder_missing": lambda m: setattr(m.speech_tokenizer.model, "decoder", None),
        "chunked_decode_missing": lambda m: setattr(
            m.speech_tokenizer.model.decoder, "chunked_decode", None
        ),
        "upsample_missing": lambda m: setattr(
            m.speech_tokenizer.model.decoder, "total_upsample", None
        ),
        "upsample_mismatch": lambda m: setattr(
            m.speech_tokenizer.model.decoder, "total_upsample", 960
        ),
        "sample_rate_missing": lambda m: setattr(
            m.speech_tokenizer.model, "get_output_sample_rate", None
        ),
        "eos_token_missing": lambda m: setattr(m.config.talker_config, "codec_eos_token_id", None),
    }
    for reason, break_it in cases.items():
        model = FakeModel()
        break_it(model)
        report = se.check_runtime(model)
        assert report.ok is False, reason
        assert report.reason == reason


def test_a_wrong_sample_rate_is_refused() -> None:
    model = FakeModel(tokenizer=FakeTokenizer(rate=16_000))
    report = se.check_runtime(model)
    assert report.ok is False
    assert report.reason == "sample_rate_mismatch"


def test_a_failed_guard_installs_no_hook() -> None:
    """A model the engine cannot read must be left exactly as it was."""
    model = FakeModel()
    model.talker.generate = None
    before = dict(model.talker.__dict__)
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))

    assert engine.available is False
    assert engine.reason == "talker_generate"
    assert se.hooks_installed(model.talker) is False
    assert model.talker.__dict__ == before

    with pytest.raises(sh.EngineUnavailable):
        list(engine.stream(_spec(), sh.CancelToken()))
    assert se.hooks_installed(model.talker) is False
    assert model.talker.__dict__ == before


def test_the_health_reason_is_a_category_not_a_message() -> None:
    report = se.check_runtime(FakeModel(tokenizer=FakeTokenizer(rate=16_000)))
    assert report.health_reason == "sample_rate_mismatch"
    assert " " not in report.health_reason
    assert se.GuardReport(True).health_reason is None


def test_a_missing_model_is_reported_not_crashed() -> None:
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model=None)))
    assert engine.available is False
    assert engine.reason == "no_model"


# --- the hook lifetime ------------------------------------------------------


def _assert_pristine(talker) -> None:
    """No residue at all: not the wrapper, and not an attribute shadowing the
    class method that was never there before."""
    assert "forward" not in talker.__dict__, "forward-Wrapper blieb zurück"
    assert "generate" not in talker.__dict__, "generate-Wrapper blieb zurück"
    assert se.hooks_installed(talker) is False


def test_a_successful_run_leaves_no_wrapper_behind() -> None:
    model = FakeModel(talker=FakeTalker(steps=6))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    blocks = _drain(engine)

    assert blocks
    _assert_pristine(model.talker)


def test_a_generation_failure_leaves_no_wrapper_behind() -> None:
    model = FakeModel()
    wrapper = FakeWrapper(model, fail=RuntimeError("kaputt"))
    engine = se.QwenStreamingEngine(FakeSynth(wrapper))

    with pytest.raises(se.StreamingEngineError):
        _drain(engine)
    _assert_pristine(model.talker)


def test_a_decoder_failure_leaves_no_wrapper_behind() -> None:
    model = FakeModel(talker=FakeTalker(steps=8))

    def _boom(_encoded):
        raise ValueError("Decoder kaputt")

    model.speech_tokenizer.decode = _boom
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))

    with pytest.raises(ValueError):
        _drain(engine)
    _assert_pristine(model.talker)


def test_a_cancel_leaves_no_wrapper_behind() -> None:
    model = FakeModel(talker=FakeTalker(steps=200, delay_s=0.001))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    _drain(engine, stop_after=1)

    _assert_pristine(model.talker)


def test_abandoning_the_generator_leaves_no_wrapper_behind() -> None:
    """The endpoint always closes the source; so must a caller that gives up."""
    model = FakeModel(talker=FakeTalker(steps=200, delay_s=0.001))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    stream = engine.stream(_spec(), sh.CancelToken())
    next(stream)
    stream.close()

    _assert_pristine(model.talker)


def test_an_untouched_engine_never_wraps_anything() -> None:
    model = FakeModel()
    se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    _assert_pristine(model.talker)


def test_a_pre_existing_own_method_is_put_back_not_deleted() -> None:
    """If something else already wrapped forward, restoring must return *that*,
    not strip it and expose the class method."""
    model = FakeModel(talker=FakeTalker(steps=4))
    sentinel = model.talker.forward
    model.talker.forward = sentinel          # an instance attribute of its own
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    _drain(engine)

    assert model.talker.__dict__["forward"] is sentinel


def test_wrappers_are_never_nested() -> None:
    model = FakeModel(talker=FakeTalker(steps=4))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    for _ in range(3):
        _drain(engine)
        _assert_pristine(model.talker)


# --- one generation at a time -----------------------------------------------


def test_a_second_generation_is_refused_without_touching_the_hooks() -> None:
    model = FakeModel(talker=FakeTalker(steps=200, delay_s=0.001))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))

    first = engine.stream(_spec(), sh.CancelToken())
    next(first)                                   # the first one holds the engine
    installed = dict(model.talker.__dict__)

    with pytest.raises(se.EngineBusy):
        list(engine.stream(_spec(), sh.CancelToken()))

    # The refusal changed nothing about the running generation.
    assert model.talker.__dict__ == installed
    first.close()
    _assert_pristine(model.talker)


def test_engine_busy_is_answered_as_unavailable() -> None:
    """The endpoint maps EngineUnavailable to 503; busy must land there too."""
    assert issubclass(se.EngineBusy, sh.EngineUnavailable)


def test_the_engine_is_usable_again_after_a_refusal() -> None:
    model = FakeModel(talker=FakeTalker(steps=200, delay_s=0.001))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    first = engine.stream(_spec(), sh.CancelToken())
    next(first)
    with pytest.raises(se.EngineBusy):
        list(engine.stream(_spec(), sh.CancelToken()))
    first.close()

    model.talker._steps = 4
    assert _drain(engine)


# --- cancellation is a flag, not an exception -------------------------------


def test_the_cancel_criterion_reaches_generate() -> None:
    model = FakeModel(talker=FakeTalker(steps=200, delay_s=0.001))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    _drain(engine, stop_after=1)

    assert model.talker.generate_calls, "generate wurde nie aufgerufen"
    criteria = model.talker.generate_calls[0]["stopping_criteria"]
    assert any(isinstance(c, se._CancelStop) for c in criteria)
    # It returned through its own path rather than being unwound.
    assert model.talker.stopped_early is True


def test_a_criterion_the_caller_passed_is_kept() -> None:
    model = FakeModel(talker=FakeTalker(steps=4))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    marker = object()

    original = model.talker.generate

    def _generate(**kwargs):
        kwargs.setdefault("stopping_criteria", [])
        return original(**kwargs)

    model.talker.generate = _generate
    wrapped = engine._wrap_generate(lambda **kw: kw, sh.CancelToken())
    result = wrapped(stopping_criteria=[marker])

    assert result["stopping_criteria"][0] is marker
    assert isinstance(result["stopping_criteria"][1], se._CancelStop)


def test_the_criterion_reports_only_once_cancelled() -> None:
    token = sh.CancelToken()
    stop = se._CancelStop(token)
    assert bool(stop(_FakeIds())) is False
    token.cancel()
    assert bool(stop(_FakeIds())) is True


def test_a_cancel_before_any_audio_yields_nothing() -> None:
    model = FakeModel(talker=FakeTalker(steps=200, delay_s=0.001))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    token = sh.CancelToken()
    token.cancel()

    assert list(engine.stream(_spec(), token)) == []
    _assert_pristine(model.talker)


def test_a_cancel_after_the_first_chunk_stops_the_stream() -> None:
    model = FakeModel(talker=FakeTalker(steps=200, delay_s=0.001))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    blocks = _drain(engine, stop_after=1)

    assert len(blocks) == 1
    _assert_pristine(model.talker)


def test_a_cancel_is_not_reported_as_a_failure() -> None:
    """A cancel must not look like an error to the caller."""
    model = FakeModel(talker=FakeTalker(steps=200, delay_s=0.001))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    _drain(engine, stop_after=1)          # no exception escapes


def test_no_thread_outlives_the_stream() -> None:
    before = {t.name for t in threading.enumerate()}
    model = FakeModel(talker=FakeTalker(steps=60, delay_s=0.001))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    _drain(engine, stop_after=1)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        leftover = {t.name for t in threading.enumerate()} - before
        if not any(name.startswith("tts-stream-") for name in leftover):
            break
        time.sleep(0.01)
    leftover = {t.name for t in threading.enumerate()} - before
    assert not any(name.startswith("tts-stream-") for name in leftover), leftover


# --- the PCM itself ---------------------------------------------------------


def test_the_pcm_is_whole_ordered_samples() -> None:
    model = FakeModel(talker=FakeTalker(steps=12))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    blocks = _drain(engine)

    assert blocks
    assert all(len(block) % 2 == 0 for block in blocks)
    assert all(block for block in blocks)
    # 400 ms is five codes; each fake code decodes to 1920 samples.
    assert blocks[0] == b"\xff\x3f" * (5 * se.EXPECTED_UPSAMPLE)


def test_the_chunk_length_decides_the_code_count() -> None:
    assert se.codes_per_chunk(400, 1920, 24_000) == 5
    assert se.codes_per_chunk(160, 1920, 24_000) == 2
    assert se.codes_per_chunk(1000, 1920, 24_000) == 13


def test_audio_is_yielded_long_before_generation_ends() -> None:
    """No full answer may be assembled before the first chunk leaves."""
    model = FakeModel(talker=FakeTalker(steps=100, delay_s=0.002))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    stream = engine.stream(_spec(), sh.CancelToken())
    next(stream)
    seen = model.talker.forward_calls
    stream.close()

    assert seen < 30, f"erst nach {seen} Schritten geliefert"


def test_the_left_context_is_dropped_from_every_later_chunk() -> None:
    """chunked_decode's own arithmetic: decode with context, keep only the new
    samples. A second chunk that came back longer would mean it was kept."""
    model = FakeModel(talker=FakeTalker(steps=20))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    blocks = _drain(engine)

    assert len({len(block) for block in blocks[:-1]}) == 1
    tokenizer = model.speech_tokenizer
    # The second window carried context; the emitted chunk did not grow.
    assert tokenizer.windows[1] > tokenizer.windows[0] - 1
    assert len(blocks[1]) == len(blocks[0])


def test_the_eos_code_is_not_spoken() -> None:
    model = FakeModel(talker=FakeTalker(steps=10, eos=999))
    engine = se.QwenStreamingEngine(FakeSynth(FakeWrapper(model)))
    blocks = _drain(engine)
    total_codes = sum(len(b) for b in blocks) // 2 // se.EXPECTED_UPSAMPLE

    assert total_codes == 10


def test_the_request_reaches_the_model_unchanged() -> None:
    model = FakeModel(talker=FakeTalker(steps=6))
    wrapper = FakeWrapper(model)
    engine = se.QwenStreamingEngine(FakeSynth(wrapper))
    _drain(engine, _spec(text="Guten Abend.", language="English", speaker="Vivian"))

    assert wrapper.calls == [
        {"text": "Guten Abend.", "language": "English", "speaker": "Vivian"}
    ]


# --- conversion helpers -----------------------------------------------------


def test_full_scale_never_wraps() -> None:
    import struct

    assert struct.unpack("<2h", se.float_to_pcm16([1.0, -1.0])) == (32767, -32767)


def test_values_beyond_the_rails_are_clamped() -> None:
    import struct

    assert struct.unpack("<2h", se.float_to_pcm16([9.0, -9.0])) == (32767, -32767)


# --- import hygiene ---------------------------------------------------------


def test_importing_the_engine_pulls_in_no_runtime() -> None:
    """The service must start, and WAV must work, without the streaming stack."""
    import subprocess

    probe = f"""
import importlib.util, sys
from pathlib import Path
root = Path({str(SERVICE)!r})
for name in ("streaming_http", "streaming_engine"):
    spec = importlib.util.spec_from_file_location(name, root / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
heavy = [n for n in ("torch", "qwen_tts", "transformers", "numpy") if n in sys.modules]
assert not heavy, heavy
print("sauber")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "sauber" in result.stdout


def test_the_engine_does_not_import_the_spike() -> None:
    source = (SERVICE / "streaming_engine.py").read_text(encoding="utf-8")
    assert "streaming_spike" not in source
