"""What a trace may hold, and what it must never hold."""

from __future__ import annotations

import json

import pytest

from kiki.harness.trace import MAX_VALUE_CHARS, TraceRecorder, TraceWriteError, sanitize


def _recorder(tmp_path, run_id="run-1") -> TraceRecorder:
    return TraceRecorder(tmp_path / "traces", run_id)


def test_every_line_is_json_with_the_required_fields(tmp_path) -> None:
    trace = _recorder(tmp_path)
    trace.write("run_started", user_text_length=12)
    trace.write("run_finished", status="completed")

    lines = trace.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert record["run_id"] == "run-1"
        assert record["event"] in {"run_started", "run_finished"}
        assert isinstance(record["sequence"], int)
        assert record["timestamp"]


def test_sequence_numbers_only_go_up(tmp_path) -> None:
    trace = _recorder(tmp_path)
    for _ in range(5):
        trace.write("model_action_received", kind="final")
    numbers = [record["sequence"] for record in trace.read()]
    assert numbers == [0, 1, 2, 3, 4]


def test_an_unknown_event_is_refused(tmp_path) -> None:
    with pytest.raises(TraceWriteError):
        _recorder(tmp_path).write("erfundenes_ereignis")


def test_the_directory_is_created_where_it_was_told(tmp_path) -> None:
    trace = _recorder(tmp_path)
    trace.write("run_started")
    assert trace.path.parent == tmp_path / "traces"
    assert trace.path.name == "run-1.jsonl"


def test_a_write_failure_is_raised_not_swallowed(tmp_path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("ich bin eine Datei, kein Verzeichnis", encoding="utf-8")
    trace = TraceRecorder(blocked, "run-1")
    with pytest.raises(TraceWriteError):
        trace.write("run_started")


def test_each_event_is_on_disk_before_the_next_one(tmp_path) -> None:
    """A crash mid-run must still leave everything up to that point readable."""
    trace = _recorder(tmp_path)
    trace.write("run_started")
    assert len(trace.read()) == 1
    trace.write("tool_requested", tool="system_status")
    assert len(trace.read()) == 2


# --- what gets sanitised ----------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "https://intern.example.com/v1",
        "/home/martin/.config/kiki/secrets.toml",
        "/Users/someone/notes",
        "/root/.ssh/id_rsa",
        "sk-live-4711abcdef",
        "ghp_abcdefghijklmnop",
        "api_key: hunter2",
        "token=abcdef",
    ],
)
def test_anything_that_looks_like_a_secret_is_removed(value) -> None:
    assert sanitize(value) == "[entfernt]"


def test_a_long_value_is_bounded(tmp_path) -> None:
    trace = _recorder(tmp_path)
    trace.write("tool_requested", tool="system_status", arguments={"x": "a" * 5000})
    stored = trace.read()[0]["arguments"]["x"]
    assert len(stored) <= MAX_VALUE_CHARS + 1


def test_nesting_is_bounded() -> None:
    """A tool cannot hide a payload behind five levels of dictionary."""
    deep: dict = {"a": {"b": {"c": {"d": {"e": "nutzlast"}}}}}
    dumped = json.dumps(sanitize(deep))
    assert "nutzlast" not in dumped
    assert "[tief]" in dumped


def test_an_arbitrary_object_never_reaches_the_trace() -> None:
    class _Thing:
        def __repr__(self) -> str:
            return "/home/martin/geheim"

    assert sanitize(_Thing()) == "[objekt]"


def test_plain_values_pass_through() -> None:
    assert sanitize({"tool": "system_status", "ok": True, "n": 3, "nix": None}) == {
        "tool": "system_status", "ok": True, "n": 3, "nix": None
    }
