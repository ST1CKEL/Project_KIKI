"""Nothing reaches the speakers that the voice policy forbids.

Speech is an output channel like the audit log: it can be overheard, recorded,
or captured by whatever else is listening in the room. The redaction rules
existed already, but only the PCM streaming route applied them -- and that route
is off by default (`tts.use_controller_route = false`). So on the route KIKI
actually uses, an answer containing a token or a home path was read aloud.

These tests hold both routes to the same rule.
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

import pytest

from kiki.voice.director import SpeechDirector
from kiki.voice.tts.policy import VoiceMode, VoicePolicyConfig, VoiceResponsePolicy

# The fixtures that must never leave the machine, spoken or written.
SECRET = "sk-test-secret"
TOKEN = "ghp_testtoken"
HOME = "/home/martin/secret.txt"
URL = "https://example.invalid/private"

LEAKY = (
    f"Klar, erledigt. Der Schlüssel ist {SECRET} und liegt in {HOME}. "
    f"Details unter {URL}. Das Token {TOKEN} gehört dazu."
)


class _Player:
    def play(self, path, *, on_eos=None, on_error=None) -> None:
        pass

    def stop(self) -> None:
        pass


class _Recording(deque):
    """Remembers everything the sink let through.

    Reading `_synth_queue` after the fact is not enough: arming a job pops the
    entry straight back out, so the queue is empty by the time a test looks and
    every assertion about it would pass whatever the code did.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    def append(self, item) -> None:
        self.seen.append(item)
        super().append(item)


def _director(tmp_path, policy=None) -> SpeechDirector:
    async def _synth(text: str, path: Path) -> Path:
        return path

    director = SpeechDirector(
        synthesize=_synth,
        player=_Player(),
        submit=lambda *a, **k: None,
        wav_dir=tmp_path,
        policy=policy,
    )
    director._synth_queue = _Recording()
    return director


def _queued(director: SpeechDirector) -> list[str]:
    """Everything that ever reached the speech queue, not what is left in it."""
    return list(director._synth_queue.seen)


# -- the shared rule ----------------------------------------------------------


@pytest.mark.parametrize("probe", [SECRET, TOKEN, HOME, URL])
def test_the_policy_removes_every_forbidden_fixture(probe) -> None:
    spoken = VoiceResponsePolicy().redact_chunk(LEAKY)
    assert probe not in spoken


def test_redaction_keeps_the_harmless_words() -> None:
    """Removing a secret must not silence the sentence around it."""
    spoken = VoiceResponsePolicy().redact_chunk(LEAKY)
    assert "Klar, erledigt." in spoken


def test_both_routes_share_one_implementation() -> None:
    """A second copy is a second thing that can drift out of date."""
    source = Path("src/kiki/voice/tts/chunker.py").read_text(encoding="utf-8")
    assert "redact_chunk" in source
    assert "_policy._redact" not in source


# -- the default (WAV) route --------------------------------------------------


def test_say_never_speaks_a_secret(tmp_path) -> None:
    director = _director(tmp_path)
    director.say(LEAKY)
    spoken = " ".join(_queued(director))
    for probe in (SECRET, TOKEN, HOME, URL):
        assert probe not in spoken


def test_a_streamed_answer_never_speaks_a_secret(tmp_path) -> None:
    """The leak arrives a delta at a time, as a real answer does."""
    director = _director(tmp_path)
    for delta in (LEAKY[i : i + 17] for i in range(0, len(LEAKY), 17)):
        director.feed(delta)
    director.flush()
    spoken = " ".join(_queued(director))
    for probe in (SECRET, TOKEN, HOME, URL):
        assert probe not in spoken


def test_the_leftover_at_the_end_is_redacted_too(tmp_path) -> None:
    """A final fragment with no sentence end still goes through the policy."""
    director = _director(tmp_path)
    director.feed(f"Der Schlüssel ist {SECRET} und der Pfad {HOME}")
    director.flush()
    spoken = " ".join(_queued(director))
    assert SECRET not in spoken
    assert HOME not in spoken


def test_an_unclosed_code_fence_is_never_spoken_early(tmp_path) -> None:
    """The closing fence may still be coming; nothing inside may leak meanwhile."""
    director = _director(tmp_path)
    director.feed(f"Hier ist das Ergebnis. ```\nexport KEY={SECRET}\n")
    spoken = " ".join(_queued(director))
    assert SECRET not in spoken


def test_a_configured_permission_is_honoured(tmp_path) -> None:
    """The policy is a policy, not a hard-coded filter."""
    allowed = VoiceResponsePolicy(VoicePolicyConfig(speak_paths=True))
    director = _director(tmp_path, policy=allowed)
    director.say(f"Die Datei liegt in {HOME}")
    spoken = " ".join(_queued(director))
    assert HOME in spoken


# -- the sink -----------------------------------------------------------------


def test_there_is_exactly_one_way_into_the_speech_queue() -> None:
    """Redaction in the sink, so a new caller cannot forget it.

    The one bare append left is inside the sink itself.
    """
    source = Path("src/kiki/voice/director.py").read_text(encoding="utf-8")
    assert source.count("_synth_queue.append(") == 1
    body = source[source.index("def _enqueue_locked(") :]
    assert "_synth_queue.append(" in body[: body.index("\n    def ", 10)]


def test_the_sink_drops_what_redaction_emptied(tmp_path) -> None:
    """A chunk that was nothing but a secret must not queue an empty utterance."""
    director = _director(tmp_path)
    director.say(SECRET)
    assert _queued(director) == []


def test_the_director_holds_a_policy_by_default(tmp_path) -> None:
    """A route without one is a route with no rule at all -- the original bug."""
    assert isinstance(_director(tmp_path)._policy, VoiceResponsePolicy)


def test_a_submitter_that_declines_the_job_leaves_no_phantom_activity(tmp_path) -> None:
    """None means no ownership transfer: the coroutine is closed and rolled back."""
    director = _director(tmp_path)
    director.say("Ein kurzer Satz.")
    assert director.active is False


# -- no speech length cap (decided) -------------------------------------------


def test_the_length_cap_is_not_applied_to_a_chunk() -> None:
    """`redact_chunk` redacts; it must not silently shorten an answer.

    Decided: KIKI speaks the whole answer. Redaction is about *what* may be
    said, never about how much.
    """
    long_answer = " ".join(f"Das ist Satz Nummer {n}." for n in range(1, 21))
    spoken = VoiceResponsePolicy().redact_chunk(long_answer)
    assert "Satz Nummer 20" in spoken
    assert len(spoken) > 300


def test_no_route_caps_what_kiki_says() -> None:
    """The capping machinery still exists; nothing on a speech path calls it.

    `plan()` can still truncate -- it is kept because the policy owns the
    vocabulary -- but neither route uses it, so both speak the full answer.
    If a cap is ever wanted it belongs on a whole answer, and this test should
    be rewritten rather than deleted.
    """
    policy = VoiceResponsePolicy()
    long_answer = " ".join(f"Das ist Satz Nummer {n}." for n in range(1, 21))
    assert policy.plan(long_answer, mode=VoiceMode.CONCISE).truncated is True

    for module in ("director.py", "tts/chunker.py"):
        source = Path("src/kiki/voice") / module
        assert "_policy.plan(" not in source.read_text(encoding="utf-8")


def test_a_long_answer_survives_the_whole_way(tmp_path) -> None:
    """End to end on the default route: twenty sentences in, twenty out."""
    director = _director(tmp_path)
    for n in range(1, 21):
        director.feed(f"Das ist Satz Nummer {n}. ")
    director.flush()
    spoken = " ".join(_queued(director))
    assert "Satz Nummer 1." in spoken
    assert "Satz Nummer 20." in spoken


def test_the_director_stays_thread_safe(tmp_path) -> None:
    """The sink runs under the caller's lock; it must not take it again."""
    director = _director(tmp_path)
    with director._lock:
        director._enqueue_locked("Ein harmloser Satz.")
    assert _queued(director) == ["Ein harmloser Satz."]
    assert isinstance(director._lock, type(threading.Lock()))


# -- the config actually governs speech ---------------------------------------


def test_the_settings_file_reaches_the_policy() -> None:
    """The whole point of wiring it: a switch in the config has an effect.

    Before this, `defaults.toml` advertised these keys and nothing read them --
    the director always used the bare dataclass defaults.
    """
    from kiki.config.settings import settings_from_mapping

    base = settings_from_mapping({}).to_mapping()
    base["voice"]["response_policy"]["speak_paths"] = True
    loosened = settings_from_mapping(base)
    assert loosened.voice.response_policy.speak_paths is True
    assert loosened.voice.response_policy.speak_secrets is False


def test_every_category_defaults_to_not_spoken() -> None:
    """Speech can be overheard; a category is spoken only once switched on."""
    from kiki.config.settings import settings_from_mapping

    chosen = settings_from_mapping({}).voice.response_policy
    assert not any(
        (
            chosen.speak_code,
            chosen.speak_logs,
            chosen.speak_urls,
            chosen.speak_paths,
            chosen.speak_tables,
            chosen.speak_secrets,
        )
    )


@pytest.mark.parametrize("damaged", ["ja", 1, None, "true", [], {}])
def test_a_damaged_value_fails_closed(damaged) -> None:
    """Only a real True opens a category. Anything else stays shut."""
    from kiki.config.settings import settings_from_mapping

    parsed = settings_from_mapping(
        {"voice": {"response_policy": {"speak_secrets": damaged}}}
    )
    assert parsed.voice.response_policy.speak_secrets is False


def test_the_settings_round_trip() -> None:
    """What is written back must be what was read, or a save loses the choice."""
    from kiki.config.settings import settings_from_mapping

    mapping = settings_from_mapping({}).to_mapping()
    mapping["voice"]["response_policy"]["speak_urls"] = True
    again = settings_from_mapping(settings_from_mapping(mapping).to_mapping())
    assert again.voice.response_policy.speak_urls is True


def test_the_application_hands_over_a_live_source() -> None:
    """Uncalled on purpose: a called method would freeze the rule at startup."""
    source = Path("src/kiki/application.py").read_text(encoding="utf-8")
    assert "policy=self._voice_policy," in source
    assert "policy=self._voice_policy()" not in source
    assert "self._settings.voice.response_policy" in source


def test_the_dead_cap_keys_are_gone_from_the_config() -> None:
    """They configured nothing. A config that promises what it cannot do is worse
    than one that stays quiet."""
    toml = Path("src/kiki/config/defaults.toml").read_text(encoding="utf-8")
    # Scoped to the section: "default_mode" is a substring of "default_model",
    # which is a live key in another block.
    block = toml[toml.index("[voice.response_policy]") :]
    block = block[: block.index("\n[")] if "\n[" in block else block
    for dead in (
        "default_mode",
        "concise_max_sentences",
        "concise_max_characters",
        "normal_max_sentences",
        "normal_max_characters",
        "detailed_speech",
    ):
        assert dead not in block, dead


def test_the_surviving_keys_are_all_wired() -> None:
    """Every key the config still offers must reach the policy."""
    from kiki.config.settings import settings_from_mapping

    toml = Path("src/kiki/config/defaults.toml").read_text(encoding="utf-8")
    block = toml[toml.index("[voice.response_policy]") :]
    block = block[: block.index("\n[")] if "\n[" in block else block
    offered = {
        line.split("=")[0].strip()
        for line in block.splitlines()
        if "=" in line and not line.strip().startswith("#")
    }
    chosen = settings_from_mapping({}).voice.response_policy
    assert offered
    for key in offered:
        assert hasattr(chosen, key), key


# -- the rule is live, not a startup snapshot ---------------------------------


def test_tightening_the_rule_reaches_the_next_sentence(tmp_path) -> None:
    """The answer is already being spoken when the setting changes."""
    current = {"policy": VoiceResponsePolicy(VoicePolicyConfig(speak_paths=True))}
    director = _director(tmp_path, policy=lambda: current["policy"])

    director.say(f"Die Datei liegt in {HOME}")
    assert HOME in " ".join(_queued(director))

    current["policy"] = VoiceResponsePolicy()
    director.say(f"Die Datei liegt in {HOME}")
    assert HOME not in _queued(director)[-1]


def test_a_plain_policy_still_works(tmp_path) -> None:
    """Passing a value, not a source, must keep behaving the same."""
    director = _director(tmp_path, policy=VoiceResponsePolicy())
    director.say(f"Der Schlüssel ist {SECRET}")
    assert SECRET not in " ".join(_queued(director))


def test_the_default_is_still_the_strict_policy(tmp_path) -> None:
    assert isinstance(_director(tmp_path)._policy, VoiceResponsePolicy)
    assert _director(tmp_path)._policy.config.speak_secrets is False


def test_the_source_is_read_per_utterance(tmp_path) -> None:
    """Not cached: caching is how a live source quietly becomes a stale one."""
    calls = []

    def _source():
        calls.append(1)
        return VoiceResponsePolicy()

    director = _director(tmp_path, policy=_source)
    director.say("Erster Satz.")
    director.say("Zweiter Satz.")
    assert len(calls) >= 2
