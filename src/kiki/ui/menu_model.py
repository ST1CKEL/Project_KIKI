"""The pet menu as data: what it holds, provable without GTK.

The menu used to be thirteen flat entries assembled inside the popover code,
where no test could reach it. This module is the menu itself -- a pure
structure built from session state, with the rules that matter baked in and
checked at construction:

* at most seven visible top-level entries, in every state combination. The
  pet is small and its menu is a right-click away; a list of thirteen is a
  settings page, not a menu. Everything else lives one level deeper;
* no empty submenus, no nested submenus -- one level of grouping, that is
  what a person can hold in their head next to a small figure;
* every action comes from a fixed vocabulary, so a typo cannot silently
  produce a dead menu item;
* every action the old menu had is still somewhere in the new one. Losing an
  entry in a redesign is a regression like any other.

The GTK side (`pet_window`) turns this into a `Gio.Menu` and owns nothing
but the conversion. State arrives as parameters: the model has no handles
into the application and can be built for any situation in a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

MAX_TOP_LEVEL = 7

# Every action the pet menu may reference. Registered in the application;
# kept here so the model can refuse anything else.
KNOWN_ACTIONS: frozenset[str] = frozenset(
    {
        "app.chat",
        "app.voice-toggle",
        "app.assistant-pause-toggle",
        "app.tts-stop",
        "app.desktop-control",
        "app.coding",
        "app.workspaces",
        "app.screenshot",
        "app.pause",
        "app.resume",
        "app.reload-character",
        "app.window-menu",
        "app.preferences",
        "app.quit",
    }
)

# The actions the old flat menu offered. The redesign must keep every one of
# them reachable -- a menu entry lost in restructuring is a feature lost.
LEGACY_ACTIONS: frozenset[str] = frozenset(KNOWN_ACTIONS - {"app.assistant-pause-toggle"})


@dataclass(frozen=True)
class MenuItem:
    """One entry. `children` non-empty makes it a submenu.

    A submenu parent needs no action of its own -- GTK shows it as a label
    with an arrow -- so its `action` may be empty. A leaf cannot: a dead
    menu item is a bug, and the fixed vocabulary is where that bug dies.
    """

    action: str
    label: str
    children: tuple[MenuItem, ...] = ()
    # Conditional entries are part of the definition but hidden in states
    # where they make no sense (nothing to stop, nothing to resume).
    hidden: bool = False


@dataclass(frozen=True)
class PetMenu:
    items: tuple[MenuItem, ...]

    def __post_init__(self) -> None:
        visible = self.visible_top_level()
        if len(visible) > MAX_TOP_LEVEL:
            raise ValueError(
                f"maximal {MAX_TOP_LEVEL} sichtbare Einträge, nicht {len(visible)}"
            )
        self._check(self.items, depth=0)

    def visible_top_level(self) -> tuple[MenuItem, ...]:
        return tuple(item for item in self.items if not item.hidden)

    def all_actions(self) -> set[str]:
        actions: set[str] = set()
        for item in self.items:
            if item.action:
                actions.add(item.action)
            for child in item.children:
                if child.action:
                    actions.add(child.action)
        return actions

    def _check(self, items: tuple[MenuItem, ...], *, depth: int) -> None:
        for item in items:
            if not item.label:
                raise ValueError("ein Menüeintrag braucht eine Beschriftung")
            if not item.children:
                if item.action not in KNOWN_ACTIONS:
                    raise ValueError(f"unbekannte Menü-Aktion: {item.action}")
            elif item.action and item.action not in KNOWN_ACTIONS:
                raise ValueError(f"unbekannte Menü-Aktion: {item.action}")
            if depth == 1 and item.children:
                raise ValueError("Menüs verschachteln sich genau eine Ebene tief")
            if item.children:
                visible = [child for child in item.children if not child.hidden]
                if not visible:
                    raise ValueError("ein Untermenü ohne sichtbare Einträge")
                self._check(item.children, depth=depth + 1)


def build_pet_menu(
    *,
    listening: bool,
    speaking: bool,
    assistant_paused: bool,
    character_paused: bool,
) -> PetMenu:
    """The menu for one moment. Every state the pet can be asked about comes
    in as a parameter; nothing is read from the application here."""
    return PetMenu(
        items=(
            MenuItem(action="app.chat", label="Chat öffnen"),
            MenuItem(
                action="app.voice-toggle",
                label="Zuhören beenden" if listening else "Zuhören",
            ),
            MenuItem(
                action="app.assistant-pause-toggle",
                label=(
                    "KIKI fortsetzen (Aufgaben)"
                    if assistant_paused
                    else "KIKI pausieren (Aufgaben)"
                ),
            ),
            # Only meaningful while a voice can be stopped.
            MenuItem(
                action="app.tts-stop",
                label="Sprechen beenden",
                hidden=not speaking,
            ),
            MenuItem(
                action="",
                label="Mehr",
                children=(
                    MenuItem(action="app.desktop-control", label="PC-Steuerung"),
                    MenuItem(action="app.coding", label="Coding-Session"),
                    MenuItem(action="app.workspaces", label="Workspaces"),
                    MenuItem(action="app.screenshot", label="Bildschirm zeigen"),
                    MenuItem(
                        action="app.resume" if character_paused else "app.pause",
                        label=(
                            "Figur fortsetzen" if character_paused else "Figur pausieren"
                        ),
                    ),
                    MenuItem(
                        action="app.reload-character", label="Figur neu laden"
                    ),
                    MenuItem(
                        action="app.window-menu",
                        label="Fenstermenü (Immer im Vordergrund)",
                    ),
                ),
            ),
            MenuItem(action="app.preferences", label="Einstellungen"),
            MenuItem(action="app.quit", label="Beenden"),
        )
    )


def all_states_valid() -> bool:
    """The bound holds in every combination of menu-relevant state.

    A property the GTK side never has to think about: whatever the session
    is doing, the built menu is legal or the construction itself complains.
    """
    for listening, speaking, assistant_paused, character_paused in product(
        (False, True), repeat=4
    ):
        menu = build_pet_menu(
            listening=listening,
            speaking=speaking,
            assistant_paused=assistant_paused,
            character_paused=character_paused,
        )
        if len(menu.visible_top_level()) > MAX_TOP_LEVEL:
            return False
    return True
