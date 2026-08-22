"""Splitting tool calls out of a token stream, including the ugly boundaries."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "services" / "kiki-llm" / "toolcalls.py"


def _load():
    spec = importlib.util.spec_from_file_location("kiki_toolcalls", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tc = _load()


def _drain(chunks: list[str]) -> tuple[str, list]:
    parser = tc.ToolCallStreamParser()
    text: list[str] = []
    calls: list = []
    for chunk in chunks:
        t, c = parser.feed(chunk)
        text.append(t)
        calls.extend(c)
    t, c = parser.finish()
    text.append(t)
    calls.extend(c)
    return "".join(text), calls


def _call_json(name="status_disk", args=None) -> str:
    return json.dumps({"name": name, "arguments": args if args is not None else {}})


# --- whole-chunk cases ------------------------------------------------------


def test_plain_text_passes_through_untouched() -> None:
    text, calls = _drain(["Hallo ", "Martin."])
    assert text == "Hallo Martin."
    assert calls == []


def test_a_single_call_is_extracted_and_removed_from_the_text() -> None:
    text, calls = _drain([f"Ich schaue nach.{tc.OPEN}{_call_json()}{tc.CLOSE}"])
    assert text == "Ich schaue nach."
    assert len(calls) == 1
    assert calls[0].name == "status_disk"
    assert calls[0].parse_error == ""
    assert calls[0].id.startswith("call_")


def test_several_calls_in_one_answer() -> None:
    body = (
        f"{tc.OPEN}{_call_json('status_disk')}{tc.CLOSE}"
        f" und {tc.OPEN}{_call_json('status_upower')}{tc.CLOSE}"
    )
    text, calls = _drain([body])
    assert [c.name for c in calls] == ["status_disk", "status_upower"]
    assert text.strip() == "und"


def test_text_after_a_call_still_arrives() -> None:
    text, calls = _drain([f"{tc.OPEN}{_call_json()}{tc.CLOSE}Fertig."])
    assert text == "Fertig."
    assert len(calls) == 1


# --- the boundaries that actually break parsers -----------------------------

@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 11])
def test_a_call_split_across_arbitrary_chunk_sizes(size) -> None:
    """Token streams cut wherever they like, including inside the tag."""
    body = f"Sehe nach.{tc.OPEN}{_call_json('status_disk', {'path': '/'})}{tc.CLOSE}Fertig."
    chunks = [body[i : i + size] for i in range(0, len(body), size)]
    text, calls = _drain(chunks)
    assert text == "Sehe nach.Fertig."
    assert len(calls) == 1
    assert calls[0].name == "status_disk"
    assert calls[0].arguments == {"path": "/"}


def test_a_partial_opening_tag_is_never_shown_to_the_user() -> None:
    parser = tc.ToolCallStreamParser()
    text, calls = parser.feed("Antwort<tool")
    # "<tool" might still become "<tool_call>", so it must be held back.
    assert text == "Antwort"
    assert calls == []
    text2, calls2 = parser.feed(f"_call>{_call_json()}{tc.CLOSE}")
    assert text2 == ""
    assert len(calls2) == 1


def test_a_lookalike_that_never_completes_is_released_at_the_end() -> None:
    text, calls = _drain(["Vergleiche <tool", "box mit etwas"])
    assert text == "Vergleiche <toolbox mit etwas"
    assert calls == []


def test_a_closing_tag_split_across_chunks() -> None:
    text, calls = _drain([f"{tc.OPEN}{_call_json()}</tool", "_call>Danach."])
    assert text == "Danach."
    assert len(calls) == 1 and calls[0].name == "status_disk"


def test_json_containing_the_closing_marker_as_data() -> None:
    body = tc.OPEN + _call_json("memory_remember", {"content": "sagte </tool_call> laut"}) + tc.CLOSE
    _text, calls = _drain([body])
    # The marker inside the string terminates the block early, so this is
    # reported as broken rather than silently mis-parsed.
    assert calls[0].parse_error != ""


# --- malformed input --------------------------------------------------------


def test_broken_json_is_reported_not_guessed() -> None:
    _text, calls = _drain([f"{tc.OPEN}{{oops{tc.CLOSE}"])
    assert len(calls) == 1
    assert "ungültiges JSON" in calls[0].parse_error
    assert calls[0].arguments == {}


def test_a_call_without_a_name_is_refused() -> None:
    _text, calls = _drain([f'{tc.OPEN}{{"arguments": {{}}}}{tc.CLOSE}'])
    assert calls[0].parse_error == "Aufruf ohne Namen"


def test_arguments_as_a_json_string_are_decoded() -> None:
    body = json.dumps({"name": "status_disk", "arguments": '{"path": "/home"}'})
    _text, calls = _drain([f"{tc.OPEN}{body}{tc.CLOSE}"])
    assert calls[0].arguments == {"path": "/home"}


def test_non_object_arguments_are_refused() -> None:
    body = json.dumps({"name": "x", "arguments": [1, 2]})
    _text, calls = _drain([f"{tc.OPEN}{body}{tc.CLOSE}"])
    assert "kein Objekt" in calls[0].parse_error


def test_an_unfinished_call_is_reported_at_the_end() -> None:
    """Generation stopped mid-call; inventing a call would be worse."""
    text, calls = _drain([f"{tc.OPEN}{{\"name\": \"status_"])
    assert text == ""
    assert len(calls) == 1
    assert "nicht abgeschlossen" in calls[0].parse_error


def test_an_empty_call_is_reported() -> None:
    _text, calls = _drain([f"{tc.OPEN}{tc.CLOSE}"])
    assert calls[0].parse_error == "leerer Aufruf"


def test_missing_arguments_default_to_empty() -> None:
    _text, calls = _drain([f'{tc.OPEN}{{"name": "status_disk"}}{tc.CLOSE}'])
    assert calls[0].arguments == {}
    assert calls[0].parse_error == ""
