"""The pet menu as data: bounded, complete, honest about state.

The menu used to be GTK code no test could reach. Now it is a model built
from state parameters, and these tests hold its three promises: never more
than seven visible entries in any state, every action the old menu had still
reachable, and labels that tell the truth about the moment.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kiki.ui.menu_model import (
    KNOWN_ACTIONS,
    LEGACY_ACTIONS,
    MAX_TOP_LEVEL,
    MenuItem,
    PetMenu,
    all_states_valid,
    build_pet_menu,
)

SRC = Path(__file__).resolve().parent.parent / "src"


def _menu(**overrides) -> PetMenu:
    state = {
        "listening": False,
        "speaking": False,
        "assistant_paused": False,
        "character_paused": False,
    }
    state.update(overrides)
    return build_pet_menu(**state)


# --- the bound -----------------------------------------------------------------


def test_the_menu_never_exceeds_seven_visible_entries():
    assert all_states_valid() is True


def test_the_worst_state_is_exactly_seven():
    menu = _menu(listening=True, speaking=True, assistant_paused=True, character_paused=True)
    visible = menu.visible_top_level()
    assert len(visible) == MAX_TOP_LEVEL
    assert [item.label for item in visible] == [
        "Chat öffnen",
        "Zuhören beenden",
        "KIKI fortsetzen (Aufgaben)",
        "Sprechen beenden",
        "Mehr",
        "Einstellungen",
        "Beenden",
    ]


def test_an_eight_top_level_entry_is_refused():
    with pytest.raises(ValueError):
        PetMenu(
            items=tuple(
                MenuItem(action="app.chat", label=f"Eintrag {i}") for i in range(8)
            )
        )


# --- completeness -----------------------------------------------------------------


def test_every_legacy_action_is_still_reachable():
    # Pause and resume are one flipping entry, like they always were: each
    # legacy action must be reachable in a state where it makes sense --
    # the union over all states loses nothing.
    union: set[str] = set()
    for listening in (False, True):
        for speaking in (False, True):
            for assistant_paused in (False, True):
                for character_paused in (False, True):
                    union |= _menu(
                        listening=listening,
                        speaking=speaking,
                        assistant_paused=assistant_paused,
                        character_paused=character_paused,
                    ).all_actions()
    assert LEGACY_ACTIONS <= union
    # And nothing invented beyond the fixed vocabulary.
    assert union <= KNOWN_ACTIONS
    assert "app.assistant-pause-toggle" in union


def test_the_submenu_groups_the_rest():
    menu = _menu()
    more = next(item for item in menu.items if item.label == "Mehr")
    assert [child.label for child in more.children] == [
        "PC-Steuerung",
        "Coding-Session",
        "Workspaces",
        "Bildschirm zeigen",
        "Figur pausieren",
        "Figur neu laden",
        "Fenstermenü (Immer im Vordergrund)",
    ]


# --- state honesty ------------------------------------------------------------------


def test_labels_follow_the_moment():
    idle = _menu()
    listening = _menu(listening=True)
    assert _label(idle, "app.voice-toggle") == "Zuhören"
    assert _label(listening, "app.voice-toggle") == "Zuhören beenden"

    paused = _menu(assistant_paused=True)
    assert _label(idle, "app.assistant-pause-toggle") == "KIKI pausieren (Aufgaben)"
    assert _label(paused, "app.assistant-pause-toggle") == "KIKI fortsetzen (Aufgaben)"

    character_idle = _menu()
    character_paused = _menu(character_paused=True)
    assert _child_action(character_idle, "Figur pausieren") == "app.pause"
    assert _child_action(character_paused, "Figur fortsetzen") == "app.resume"


def test_stop_speaking_exists_only_while_speaking():
    silent = _menu(speaking=False)
    talking = _menu(speaking=True)
    assert all(item.action != "app.tts-stop" for item in silent.visible_top_level())
    assert any(item.action == "app.tts-stop" for item in talking.visible_top_level())


# --- construction rules -----------------------------------------------------------


def test_unknown_actions_are_refused():
    with pytest.raises(ValueError):
        PetMenu(items=(MenuItem(action="app.gibtsnicht", label="X"),))


def test_nested_submenus_are_refused():
    inner = MenuItem(action="app.chat", label="Tief")
    outer_child = MenuItem(action="", label="Zu tief", children=(inner,))
    with pytest.raises(ValueError):
        PetMenu(items=(MenuItem(action="", label="Mehr", children=(outer_child,)),))


def test_an_empty_submenu_is_refused():
    hidden_child = MenuItem(action="app.chat", label="Chat", hidden=True)
    with pytest.raises(ValueError):
        PetMenu(items=(MenuItem(action="", label="Mehr", children=(hidden_child,)),))


def test_a_leaf_without_action_is_refused():
    with pytest.raises(ValueError):
        PetMenu(items=(MenuItem(action="", label="Nirgendwohin"),))


# --- GTK-freedom --------------------------------------------------------------------


def test_the_model_imports_without_gtk():
    code = (
        "import sys; import kiki.ui.menu_model; "
        "sys.stdout.write(','.join(sorted("
        "m for m in sys.modules if m == 'gi' or m.startswith('gi.'))))"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def _label(menu: PetMenu, action: str) -> str:
    return next(item.label for item in menu.items if item.action == action)


def _child_action(menu: PetMenu, label: str) -> str:
    more = next(item for item in menu.items if item.label == "Mehr")
    return next(child.action for child in more.children if child.label == label)
