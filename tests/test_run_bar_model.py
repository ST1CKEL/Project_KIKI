"""The run bar model: one vocabulary, decided before GTK sees anything.

The bar's logic used to be window code. These tests hold the model's
promises: every run state has its one fixed sentence, success is silent,
cancelling is possible exactly while it means something -- and the word
"harness" appears nowhere, because runs are runs wherever they come from.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kiki.ui.run_bar_model import (
    CANCEL_PENDING_TEXT,
    CANCEL_TEXT,
    RunBarView,
    run_bar_for,
    text_for,
)

SRC = Path(__file__).resolve().parent.parent / "src"

ACTIVE_CODES = ("working", "tool_running", "needs_confirmation")
TERMINAL_CODES = ("cancelled", "failed", "limit_reached")


# --- one sentence per state -----------------------------------------------------


def test_working_spins_and_can_be_cancelled():
    view = run_bar_for("working")
    assert view == RunBarView(
        visible=True, text="KIKI arbeitet …", spinner=True, cancellable=True
    )


def test_tool_running_spins_and_can_be_cancelled():
    view = run_bar_for("tool_running")
    assert view.text == "KIKI führt eine Aufgabe aus …"
    assert view.spinner is True and view.cancellable is True


def test_waiting_for_confirmation_does_not_spin_but_can_be_cancelled():
    # Cancelling is exactly how to say no to the whole run while the card is
    # up -- the button must stay alive there.
    view = run_bar_for("needs_confirmation")
    assert view.text == "KIKI wartet auf deine Bestätigung."
    assert view.spinner is False and view.cancellable is True


@pytest.mark.parametrize("code", TERMINAL_CODES)
def test_terminal_states_are_visible_final_and_uncancellable(code):
    view = run_bar_for(code, terminal=True)
    assert view.visible is True
    assert view.spinner is False
    assert view.cancellable is False


def test_completed_is_the_one_silent_state():
    view = run_bar_for("completed", terminal=True)
    assert view.visible is False
    assert view.text == ""


def test_unknown_codes_are_a_bug_not_a_silence():
    with pytest.raises(ValueError):
        run_bar_for("irgendwas")
    with pytest.raises(ValueError):
        text_for("irgendwas")


def test_a_terminal_active_code_offers_no_cancel():
    # Defensive parity with the old window logic: an active code marked
    # terminal (should not happen -- the events keep the two consistent) is
    # shown, but nothing can be cancelled.
    view = run_bar_for("working", terminal=True)
    assert view.visible is True
    assert view.cancellable is False
    assert view.spinner is False


# --- the vocabulary stays honest ---------------------------------------------------


def test_no_sentence_mentions_internals():
    for code in (*ACTIVE_CODES, *TERMINAL_CODES):
        text = text_for(code)
        lowered = text.lower()
        for forbidden in (
            "harness",
            "/home/",
            "traceback",
            "token",
            "prompt",
            "run_id",
            "tool",
        ):
            assert forbidden not in lowered, (code, forbidden)


def test_the_model_source_names_no_harness():
    source = (SRC / "kiki" / "ui" / "run_bar_model.py").read_text(encoding="utf-8")
    assert "harness" not in source.lower()


def test_the_cancel_vocabulary_is_fixed():
    assert CANCEL_TEXT == "Abbrechen"
    assert CANCEL_PENDING_TEXT == "Abbruch angefordert …"


def test_the_model_imports_without_gtk():
    code = (
        "import sys; import kiki.ui.run_bar_model; "
        "sys.stdout.write(','.join(sorted("
        "m for m in sys.modules if m == 'gi' or m.startswith('gi.'))))"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
