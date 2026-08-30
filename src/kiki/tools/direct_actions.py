"""Deterministic authorization for explicit local launch commands.

This parser is intentionally small. It recognizes one imperative and one local
target, resolves that target against local indexes, and invokes the exact id as
``Origin.USER``. Anything more complex stays in the ordinary assistant path.
"""

from __future__ import annotations

import difflib
import re
import uuid
from dataclasses import dataclass
from enum import StrEnum

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


class LaunchRoute(StrEnum):
    AUTO = "auto"
    STEAM = "steam"


@dataclass(frozen=True)
class DirectLaunchRequest:
    target: str
    route: LaunchRoute = LaunchRoute.AUTO


@dataclass(frozen=True)
class DirectActionResult:
    answer: str
    ok: bool
    tool: str = ""


def parse_direct_launch(text: str) -> DirectLaunchRequest | None:
    """Recognize exactly one bounded launch command, never a general prompt."""
    raw = (text or "").strip()
    if not raw or len(raw) > 180:
        return None
    match = _PREFIX.fullmatch(raw)
    if match is None:
        return None
    target = match.group(1).strip()
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
    return DirectLaunchRequest(target=target, route=route)


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
