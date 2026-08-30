"""Deterministic authorization for explicit local launch commands.

This parser is intentionally small. It recognizes one imperative and one local
target, resolves that target against local indexes, and invokes the exact id as
``Origin.USER``. Anything more complex stays in the ordinary assistant path.

`parse_direct_control` extends the same idea to everyday safety controls
(volume, mute, media, brightness, screen lock, time): one bounded phrase maps
to one declared tool call. No model, no 4–10 s run, no tool-selection errors —
spoken commands answer in about half a second.
"""

from __future__ import annotations

import difflib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from kiki.tools.app_launch_tools import DesktopIndex
from kiki.tools.gateway import ToolGateway, ToolInvocation
from kiki.tools.policy import Origin
from kiki.tools.steam_launch_tools import SteamIndex

_PREFIX = re.compile(
    r"^(?:(?:hey\s+)?kiki\s*[,,:]?\s*)?(?:bitte\s+)?"
    r"(?:starte|start|öffne|oeffne|open|launch)\s+(?:bitte\s+)?(.+?)\s*[.!]?$",
    re.IGNORECASE,
)
_STEAM_SUFFIX = re.compile(r"\s+(?:über|ueber|mit)\s+steam\s*$", re.IGNORECASE)
_STEAM_PREFIX = re.compile(r"^(?:das\s+)?(?:steam[- ]spiel|spiel)\s+", re.IGNORECASE)
_ARTICLE = re.compile(r"^(?:die|den|das|der)\s+", re.IGNORECASE)
_MULTI_ACTION = re.compile(
    r"(?:[;?\n\r]|\b(?:und\s+dann|danach|anschließend|anschliessend)\b)", re.IGNORECASE
)
_FUZZY_COMPACT = re.compile(r"[\s.\-_]+")
# Speech recognition mangles foreign app names ("Thunderbird" arrives as
# "sander bord"). The fallback may only close that gap when one candidate is
# close to the heard name AND clearly better than every other candidate.
_FUZZY_MIN_RATIO = 0.6
_FUZZY_MIN_MARGIN = 0.12
_FUZZY_MIN_LENGTH = 4


class LaunchAction(StrEnum):
    OPEN = "open"
    CLOSE = "close"


class LaunchRoute(StrEnum):
    AUTO = "auto"
    STEAM = "steam"


@dataclass(frozen=True)
class DirectLaunchRequest:
    target: str
    route: LaunchRoute = LaunchRoute.AUTO
    action: LaunchAction = LaunchAction.OPEN


@dataclass(frozen=True)
class DirectActionResult:
    answer: str
    ok: bool
    tool: str = ""


_OPEN_VERBS = "starte|start|öffne|oeffne|open|launch"
_CLOSE_VERBS = "beende|beenden|schließt?|schließe|schliessen|schließen|quit|close"
_PREFIX = re.compile(
    rf"^(?:(?:hey\s+)?kiki\s*[,,:]?\s*)?(?:bitte\s+)?"
    rf"(?P<verb>{_OPEN_VERBS}|{_CLOSE_VERBS})\s+"
    rf"(?:bitte\s+)?(?P<target>.+?)\s*[.!]?$",
    re.IGNORECASE,
)
_CLOSE = frozenset(
    {
        "beende",
        "beenden",
        "schließ",
        "schliesst",
        "schließe",
        "schliessen",
        "schließen",
        "quit",
        "close",
    }
)


def parse_direct_launch(text: str) -> DirectLaunchRequest | None:
    """Recognize exactly one bounded launch command, never a general prompt."""
    raw = (text or "").strip()
    if not raw or len(raw) > 180:
        return None
    match = _PREFIX.fullmatch(raw)
    if match is None:
        return None
    # lower(), not casefold(): ß folds to "ss" and would miss the verb.
    action = LaunchAction.CLOSE if match.group("verb").lower() in _CLOSE else LaunchAction.OPEN
    target = match.group("target").strip()
    steam = _STEAM_SUFFIX.search(target)
    route = LaunchRoute.STEAM if steam is not None else LaunchRoute.AUTO
    if steam is not None:
        target = target[: steam.start()].strip()
    steam_prefix = _STEAM_PREFIX.match(target)
    if steam_prefix is not None:
        route = LaunchRoute.STEAM
        target = target[steam_prefix.end() :].strip()
    target = _ARTICLE.sub("", target).strip().strip('"“”')
    if (
        not target
        or len(target) > 128
        or _MULTI_ACTION.search(target)
        or "://" in target
        or "/" in target
        or "\\" in target
        or target.startswith("-")
    ):
        return None
    return DirectLaunchRequest(target=target, route=route, action=action)


def _best_fuzzy(candidates: list[tuple[str, str]], target: str) -> str | None:
    """Return the key of one clearly closest candidate, or None.

    `candidates` are (key, name) pairs. The heard target matches only when its
    compacted form is similar enough to exactly one candidate name; ties stay
    unresolved and fall through to the ordinary "not found" answer.
    """
    needle = _FUZZY_COMPACT.sub("", target.casefold())
    if len(needle) < _FUZZY_MIN_LENGTH:
        return None
    scored = sorted(
        (
            (
                difflib.SequenceMatcher(
                    None, needle, _FUZZY_COMPACT.sub("", name.casefold())
                ).ratio(),
                key,
            )
            for key, name in candidates
        ),
        reverse=True,
    )
    if not scored or scored[0][0] < _FUZZY_MIN_RATIO:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < _FUZZY_MIN_MARGIN:
        return None
    return scored[0][1]


class DirectActionService:
    """Resolve and execute a parsed direct command through the shared gateway."""

    def __init__(
        self,
        gateway: ToolGateway,
        applications: DesktopIndex,
        steam: SteamIndex,
    ) -> None:
        self._gateway = gateway
        self._applications = applications
        self._steam = steam

    @staticmethod
    def parse(text: str) -> DirectLaunchRequest | None:
        return parse_direct_launch(text)

    async def execute(self, request: DirectLaunchRequest) -> DirectActionResult:
        if request.action is LaunchAction.CLOSE:
            return await self._close(request)
        app = None if request.route is LaunchRoute.STEAM else self._applications.find(request.target)
        if app is None and request.route is LaunchRoute.AUTO:
            fuzzy_id = _best_fuzzy(
                [(entry.app_id, entry.name) for entry in self._applications.entries()],
                request.target,
            )
            if fuzzy_id is not None:
                app = self._applications.find(fuzzy_id)
        if app is not None:
            return await self._invoke(
                "app.open",
                {"app_id": app.app_id},
                success=f"Ich starte {app.name}.",
            )
        game = self._steam.find(request.target)
        if game is None:
            fuzzy_id = _best_fuzzy(
                [(game.app_id, game.name) for game in self._steam.entries()],
                request.target,
            )
            if fuzzy_id is not None:
                game = self._steam.find(fuzzy_id)
        if game is not None:
            return await self._invoke(
                "steam.launch",
                {"app_id": game.app_id},
                success=f"Ich starte {game.name} über Steam.",
            )
        category = "Steam-Spiel" if request.route is LaunchRoute.STEAM else "Anwendung oder Steam-Spiel"
        return DirectActionResult(
            answer=f"Ich habe keine eindeutige lokale {category} namens „{request.target}“ gefunden.",
            ok=False,
        )

    async def execute_control(self, request: DirectControlRequest) -> DirectActionResult:
        """Run one everyday control command through the shared gateway.

        Time and date carry no tool: they are answered from the local clock
        and never touch authorization at all.
        """
        if not request.tool:
            return DirectActionResult(answer=request.answer, ok=True)
        token = uuid.uuid4().hex[:12]
        result = await self._gateway.invoke(
            ToolInvocation(
                tool=request.tool,
                arguments=request.arguments,
                actor=Origin.USER,
                run_id=f"direct-{token}",
                call_id=f"call-{token}",
            )
        )
        payload_ok = bool((result.data or {}).get("ok", True))
        if result.ok and payload_ok:
            return DirectActionResult(answer=request.answer, ok=True, tool=request.tool)
        detail = result.error or str(
            (result.data or {}).get("error") or "Der Befehl ist fehlgeschlagen."
        )
        return DirectActionResult(answer=detail, ok=False, tool=request.tool)

    async def _close(self, request: DirectLaunchRequest) -> DirectActionResult:
        """Close a desktop app. Steam games are deliberately out of scope: a
        game that loses its process can lose a save — that choice stays with
        the person inside the game."""
        if request.route is LaunchRoute.STEAM:
            return DirectActionResult(
                answer=(
                    "Steam-Spiele schließe ich nicht selbst — bitte über das "
                    "Spielmenü beenden, damit ein Spielstand sicher ist."
                ),
                ok=False,
            )
        app = self._applications.find(request.target)
        if app is None:
            fuzzy_id = _best_fuzzy(
                [(entry.app_id, entry.name) for entry in self._applications.entries()],
                request.target,
            )
            if fuzzy_id is not None:
                app = self._applications.find(fuzzy_id)
        if app is None:
            return DirectActionResult(
                answer=f"Ich habe keine Anwendung namens „{request.target}“ gefunden.",
                ok=False,
            )
        return await self._invoke(
            "app.close",
            {"app_id": app.app_id},
            success=f"Ich schließe {app.name}.",
        )

    async def _invoke(
        self,
        tool: str,
        arguments: dict[str, str],
        *,
        success: str,
    ) -> DirectActionResult:
        token = uuid.uuid4().hex[:12]
        result = await self._gateway.invoke(
            ToolInvocation(
                tool=tool,
                arguments=arguments,
                actor=Origin.USER,
                run_id=f"direct-{token}",
                call_id=f"call-{token}",
            )
        )
        payload_ok = bool((result.data or {}).get("ok", True))
        if result.ok and payload_ok:
            return DirectActionResult(answer=success, ok=True, tool=tool)
        detail = result.error or str((result.data or {}).get("error") or "Start fehlgeschlagen.")
        return DirectActionResult(answer=detail, ok=False, tool=tool)


# --- everyday control commands ------------------------------------------------


@dataclass(frozen=True)
class DirectControlRequest:
    """One bounded phrase mapped to one declared tool call.

    `tool` empty means the request is answered locally (time and date) and
    never touches the gateway.
    """

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    answer: str = ""


def _percent(raw: str) -> int:
    return max(0, min(100, int(raw)))


def _now_answer(_match: re.Match) -> DirectControlRequest:
    moment = datetime.now()
    return DirectControlRequest(
        tool="",
        answer=f"Es ist {moment.strftime('%H:%M')} Uhr.",
    )


def _date_answer(_match: re.Match) -> DirectControlRequest:
    weekdays = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    months = [
        "Januar", "Februar", "März", "April", "Mai", "Juni",
        "Juli", "August", "September", "Oktober", "November", "Dezember",
    ]
    moment = datetime.now()
    return DirectControlRequest(
        tool="",
        answer=(
            f"Heute ist {weekdays[moment.weekday()]}, der {moment.day}. "
            f"{months[moment.month - 1]} {moment.year}."
        ),
    )


_CONTROL_RULES: tuple[tuple[re.Pattern[str], Any], ...] = (
    # volume
    (
        re.compile(
            r"(?:stell(?:e)?\s+)?(?:die\s+)?lautstärke\s+(?:auf\s+)?"
            r"(?P<value>\d{1,3})(?:\s*(?:prozent|%))?\s*[.!]?",
            re.IGNORECASE,
        ),
        lambda m: DirectControlRequest(
            tool="audio.volume_set",
            arguments={"percent": _percent(m.group("value"))},
            answer=f"Ich stelle die Lautstärke auf {_percent(m.group('value'))} Prozent.",
        ),
    ),
    (
        re.compile(r"lautstärke\s+(?:auf\s+)?(?:voll|maximal|hundert(?:\s+prozent)?)\s*[.!]?", re.IGNORECASE),
        lambda _m: DirectControlRequest(
            tool="audio.volume_set",
            arguments={"percent": 100},
            answer="Ich stelle die Lautstärke auf 100 Prozent.",
        ),
    ),
    # mute
    (
        re.compile(r"(?:schalte\s+(?:den\s+)?ton\s+aus|ton\s+aus|stumm(?:schalten)?|mute)\s*[.!]?", re.IGNORECASE),
        lambda _m: DirectControlRequest(
            tool="audio.mute",
            arguments={"muted": True},
            answer="Ich stelle den Ton stumm.",
        ),
    ),
    (
        re.compile(r"(?:schalte\s+(?:den\s+)?ton\s+(?:wieder\s+)?ein|ton\s+(?:wieder\s+)?an|unmute)\s*[.!]?", re.IGNORECASE),
        lambda _m: DirectControlRequest(
            tool="audio.mute",
            arguments={"muted": False},
            answer="Ich schalte den Ton wieder ein.",
        ),
    ),
    # brightness
    (
        re.compile(
            r"(?:stell(?:e)?\s+)?(?:die\s+)?helligkeit\s+(?:auf\s+)?"
            r"(?P<value>\d{1,3})(?:\s*(?:prozent|%))?\s*[.!]?",
            re.IGNORECASE,
        ),
        lambda m: DirectControlRequest(
            tool="display.brightness_set",
            arguments={"percent": _percent(m.group("value"))},
            answer=f"Ich stelle die Helligkeit auf {_percent(m.group('value'))} Prozent.",
        ),
    ),
    # media
    (
        re.compile(r"(?:pausier(?:e)?|halt(?:e)?)\s+(?:die\s+)?(?:musik|wiedergabe)(?:\s+an)?\s*[.!]?", re.IGNORECASE),
        lambda _m: DirectControlRequest(
            tool="media.play_pause",
            answer="Ich pausiere die Wiedergabe.",
        ),
    ),
    (
        re.compile(r"spiel(?:e)?\s+(?:die\s+)?(?:musik|wiedergabe)\s+(?:wieder\s+)?ab\s*[.!]?", re.IGNORECASE),
        lambda _m: DirectControlRequest(
            tool="media.play_pause",
            answer="Ich spiele die Wiedergabe weiter.",
        ),
    ),
    (
        re.compile(r"(?:nächster\s+titel|nächstes\s+lied|überspring(?:e|en)(?:\s+den\s+titel)?)\s*[.!]?", re.IGNORECASE),
        lambda _m: DirectControlRequest(
            tool="media.next",
            answer="Ich überspringe zum nächsten Titel.",
        ),
    ),
    (
        re.compile(r"(?:vorheriger\s+titel|vorheriges\s+lied|letztes\s+lied)\s*[.!]?", re.IGNORECASE),
        lambda _m: DirectControlRequest(
            tool="media.previous",
            answer="Ich springe zurück zum vorherigen Titel.",
        ),
    ),
    (
        re.compile(r"stop(?:p)?\s+(?:die\s+)?(?:musik|wiedergabe)\s*[.!]?", re.IGNORECASE),
        lambda _m: DirectControlRequest(
            tool="media.stop",
            answer="Ich stoppe die Wiedergabe.",
        ),
    ),
    (
        re.compile(r"stop(?:p)?\s*[.!]?", re.IGNORECASE),
        lambda _m: DirectControlRequest(
            tool="media.play_pause",
            answer="Ich pausiere die Wiedergabe.",
        ),
    ),
    # screen lock
    (
        re.compile(r"sperr(?:e|en)?\s+(?:den\s+)?bildschirm|bildschirm\s+sperren\s*[.!]?", re.IGNORECASE),
        lambda _m: DirectControlRequest(
            tool="session.lock",
            answer="Ich sperre den Bildschirm.",
        ),
    ),
    # time and date — answered locally, never through a tool
    (
        re.compile(r"(?:wie\s+spät(?:\s+ist\s+es)?|wieviel\s+uhr|welche\s+uhrzeit(?:\s+haben\s+wir)?)\s*[.!]?", re.IGNORECASE),
        _now_answer,
    ),
    (
        re.compile(r"welches\s+datum(?:\s+haben\s+wir(?:\s+heute)?)?|welcher\s+tag\s+ist\s+heute\s*[.!]?", re.IGNORECASE),
        _date_answer,
    ),
)


def parse_direct_control(text: str) -> DirectControlRequest | None:
    """Recognize one bounded everyday control command, or None."""
    raw = " ".join((text or "").strip().split())
    # A trailing question mark belongs to the phrase ("Wie spät ist es?"); a
    # question mark *inside* still means a real question and stays rejected.
    raw = re.sub(r"\s*[.!?]+\s*$", "", raw)
    if not raw or len(raw) > 80 or _MULTI_ACTION.search(raw):
        return None
    for pattern, build in _CONTROL_RULES:
        match = pattern.fullmatch(raw)
        if match is not None:
            return build(match)
    return None

