"""MPRIS media control: status, play/pause, skip, stop.

The session bus is the only surface. KIKI never touches player files or
streams; the metadata she reports is trimmed to title, artist, album and
length so a tool result stays small. Players keep their own rules — when a
player answers CanGoNext=false, that answer is reported, not forced.
"""

from __future__ import annotations

import logging
from typing import Any

from kiki.platform import dbus
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec

log = logging.getLogger(__name__)

_MPRIS_PREFIX = "org.mpris.MediaPlayer2."
_PLAYER_PATH = "/org/mpris/MediaPlayer2"
_PLAYER_IFACE = "org.mpris.MediaPlayer2.Player"
_MAX_PLAYERS_IN_STATUS = 5


class MprisError(RuntimeError):
    """The bus or a player refused the request."""


def short_player_name(bus_name: str) -> str:
    """`org.mpris.MediaPlayer2.firefox.instance_1` → `firefox`."""
    suffix = bus_name.removeprefix(_MPRIS_PREFIX)
    return suffix.split(".", maxsplit=1)[0] or suffix


class MprisClient:
    """Thin MPRIS wrapper. Kept behind a class so tests can replace it."""

    def list_players(self) -> list[str]:
        bus = dbus.session_bus()
        names = dbus.list_names(bus)
        return sorted(n for n in names if n.startswith(_MPRIS_PREFIX))

    def player_properties(self, player: str) -> dict[str, Any]:
        from gi.repository import GLib

        bus = dbus.session_bus()
        reply = dbus.call(
            bus,
            player,
            _PLAYER_PATH,
            "org.freedesktop.DBus.Properties",
            "GetAll",
            GLib.Variant("(s)", (_PLAYER_IFACE,)),
            GLib.VariantType("(a{sv})"),
        )
        return dict(reply.unpack()[0])

    def call_player_method(self, player: str, method: str) -> None:
        dbus.call(
            dbus.session_bus(),
            player,
            _PLAYER_PATH,
            _PLAYER_IFACE,
            method,
        )


def shape_player_properties(props: dict[str, Any]) -> dict[str, Any]:
    """Reduce an MPRIS property dict to the small, stable subset KIKI reports."""
    meta = props.get("Metadata")
    meta = meta if isinstance(meta, dict) else {}
    artists = meta.get("xesam:artist")
    if isinstance(artists, str):
        artists = [artists]
    elif not isinstance(artists, (list, tuple)):
        artists = []
    length_us = meta.get("mpris:length")
    length_s = round(length_us / 1_000_000) if isinstance(length_us, int) else None
    return {
        "playback_status": props.get("PlaybackStatus"),
        "can_play": bool(props.get("CanPlay", False)),
        "can_pause": bool(props.get("CanPause", False)),
        "can_go_next": bool(props.get("CanGoNext", False)),
        "can_go_previous": bool(props.get("CanGoPrevious", False)),
        "title": meta.get("xesam:title"),
        "artist": ", ".join(str(a) for a in artists if a) or None,
        "album": meta.get("xesam:album"),
        "length_s": length_s,
    }


class MediaControlSkill:
    id = "media_control"
    name = "Medien"
    description = "Medienwiedergabe steuern: Status abfragen, Play/Pause, Titel wechseln, Stop."

    def __init__(self, client: MprisClient | None = None) -> None:
        self._client = client or MprisClient()

    def tools(self) -> list[ToolSpec]:
        return [
            self._status_spec(),
            self._control_spec("play_pause", "Play/Pause", "PlayPause", "can_pause"),
            self._control_spec("next", "Nächster Titel", "Next", "can_go_next"),
            self._control_spec("previous", "Vorheriger Titel", "Previous", "can_go_previous"),
            self._control_spec("stop", "Wiedergabe stoppen", "Stop", "can_pause"),
        ]

    def _status_spec(self) -> ToolSpec:
        return ToolSpec(
            name="media.status",
            title="Medienstatus",
            description=(
                "Listet die laufenden Medienplayer mit Wiedergabestatus, Titel, "
                "Interpret und Album. Spielen mehrere Player, entscheidet erst "
                "dieser Status, welcher Name in die anderen Medien-Werkzeuge passt."
            ),
            risk=RiskLevel.READ,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=self._status,
            effect="Liest den MPRIS-Status über den Sitzungsbus. Keine Änderung.",
            target="Medienplayer",
            auto_allow=True,
            model_callable=True,
        )

    def _control_spec(
        self, slug: str, title: str, method: str, can_key: str
    ) -> ToolSpec:
        return ToolSpec(
            name=f"media.{slug}",
            title=title,
            description=(
                f"{title} beim aktiven Medienplayer. Optionaler Parameter `player` "
                " wählt einen Player per Kurznamen (z. B. `firefox`) aus media.status."
            ),
            risk=RiskLevel.CONTROL,
            parameters={
                "type": "object",
                "properties": {
                    "player": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                    }
                },
                "additionalProperties": False,
            },
            handler=lambda params, method=method, can_key=can_key: self._control(
                method, can_key, params
            ),
            effect=f"{title} beim Medienplayer — ändert keine Dateien.",
            target="Medienplayer",
            auto_allow=True,
            model_callable=True,
        )

    def _status(self, _params: dict[str, Any]) -> dict[str, Any]:
        try:
            players = self._client.list_players()
        except dbus.BusError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            return {"ok": False, "error": f"Medienstatus nicht abrufbar: {exc}"}
        if not players:
            return {"ok": True, "count": 0, "players": []}
        reported: list[dict[str, Any]] = []
        for name in players[:_MAX_PLAYERS_IN_STATUS]:
            try:
                props = shape_player_properties(self._client.player_properties(name))
            except Exception as exc:
                props = {"error": f"Status nicht lesbar: {exc}"}
            reported.append({"player": short_player_name(name), **props})
        return {"ok": True, "count": len(players), "players": reported}

    def _select_player(self, params: dict[str, Any]) -> tuple[str | None, str]:
        """Pick the player a control call goes to. None plus a reason on failure."""
        try:
            names = self._client.list_players()
        except Exception as exc:
            return None, f"Medienplayer nicht abrufbar: {exc}"
        if not names:
            return None, "Kein Medienplayer läuft gerade."
        hint = str(params.get("player") or "").strip().lower()
        if hint:
            matches = [n for n in names if hint in short_player_name(n) or hint in n.lower()]
            if not matches:
                return None, f"Kein Medienplayer passt zu „{params['player']}“."
            names = matches
        if len(names) == 1:
            return names[0], ""
        # Several players: prefer the one currently playing, else the first.
        for name in names:
            try:
                props = self._client.player_properties(name)
            except Exception:
                continue
            if props.get("PlaybackStatus") == "Playing":
                return name, ""
        return names[0], ""

    def _control(self, method: str, can_key: str, params: dict[str, Any]) -> dict[str, Any]:
        player, error = self._select_player(params)
        if player is None:
            return {"ok": False, "error": error}
        try:
            props = shape_player_properties(self._client.player_properties(player))
        except Exception:
            props = {}
        if props and not props.get(can_key, True):
            return {
                "ok": False,
                "error": f"{short_player_name(player)} meldet: diese Aktion ist gerade nicht möglich.",
            }
        try:
            self._client.call_player_method(player, method)
        except Exception as exc:
            return {"ok": False, "error": f"Mediensteuerung fehlgeschlagen: {exc}"}
        return {"ok": True, "player": short_player_name(player), "action": method.lower()}
