"""The run bar as data: one vocabulary, no internals, provable without GTK.

The bar at the bottom of the chat is the one place a run becomes visible.
Its logic -- which code shows which sentence, when it spins, when it can be
cancelled -- used to live inside the window, where no test could reach it.
It is a model now, and the window only obeys it.

The vocabulary is fixed German on purpose: a run's state is a category, and
the person reading the bar deserves the same sentence every time, never a
tool name, never a path, never an exception. `completed` is the one silent
state: success needs no bar, the answer itself is the report.

Runs are runs, wherever they come from -- the agent path, voice, or whatever
asks next. The internal name this layer once carried belongs to no
user-facing surface, and none of it appears here.
"""

from __future__ import annotations

from dataclasses import dataclass

CANCEL_TEXT = "Abbrechen"
CANCEL_PENDING_TEXT = "Abbruch angefordert …"

# One sentence per run message code. Nothing else may reach the bar.
_TEXTS: dict[str, str] = {
    "working": "KIKI arbeitet …",
    "tool_running": "KIKI führt eine Aufgabe aus …",
    "needs_confirmation": "KIKI wartet auf deine Bestätigung.",
    "cancelled": "KIKI wurde abgebrochen.",
    "failed": "KIKI konnte die Aufgabe nicht ausführen.",
    "limit_reached": "KIKI hat die Aufgabe aus Sicherheitsgründen beendet.",
}

_ACTIVE: frozenset[str] = frozenset({"working", "tool_running", "needs_confirmation"})
_SPINNING: frozenset[str] = frozenset({"working", "tool_running"})


@dataclass(frozen=True)
class RunBarView:
    """What the bar shows for one state. Data only; the window renders it."""

    visible: bool
    text: str
    spinner: bool
    cancellable: bool


def text_for(message_code: str) -> str:
    """The sentence for a code. Unknown codes are a bug, not a silence."""
    try:
        return _TEXTS[message_code]
    except KeyError:
        raise ValueError(f"unbekannter Lauf-Statuscode: {message_code}") from None


def run_bar_for(message_code: str, *, terminal: bool = False) -> RunBarView:
    """The bar for one moment of a run.

    `completed` hides the bar: the answer is the report. Active states can
    still be cancelled -- including waiting for a confirmation, where
    cancelling is exactly how to say no to the whole run. Terminal states
    stay visible long enough to be read and offer nothing to cancel.
    """
    if message_code == "completed":
        return RunBarView(visible=False, text="", spinner=False, cancellable=False)
    text = text_for(message_code)
    active = message_code in _ACTIVE and not terminal
    return RunBarView(
        visible=True,
        text=text,
        spinner=message_code in _SPINNING and not terminal,
        cancellable=active,
    )
