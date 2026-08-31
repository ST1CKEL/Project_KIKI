"""Spoken yes/no for ToolGateway confirmations.

Destructive and modifying actions still go through the same broker as the GTK
card. The voice is just the channel. A timeout is a refusal.
"""

from __future__ import annotations

import re
import unicodedata

_YES = frozenset(
    {
        "ja",
        "ja bitte",
        "jawohl",
        "okay",
        "ok",
        "mach",
        "mach das",
        "machs",
        "tu es",
        "tu das",
        "bitte",
        "bestätigt",
        "bestatigt",
        "yes",
        "yep",
        "sure",
        "einverstanden",
        "mach ruhig",
        "leg los",
    }
)
_NO = frozenset(
    {
        "nein",
        "nee",
        "nö",
        "no",
        "nope",
        "abbrechen",
        "stopp",
        "halt",
        "nicht",
        "lass",
        "lass es",
        "lass das",
        "bloß nicht",
        "bloss nicht",
        "auf keinen fall",
        "abbruch",
        "cancel",
    }
)


def _fold(text: str) -> str:
    raw = unicodedata.normalize("NFKC", text or "")
    raw = raw.casefold()
    raw = re.sub(r"[^\w\s+]", " ", raw, flags=re.UNICODE)
    return " ".join(raw.split())


def parse_spoken_verdict(text: str) -> bool | None:
    """True / False / None (unintelligible — ask again, do not guess yes)."""
    folded = _fold(text)
    if not folded:
        return None
    if folded in _YES:
        return True
    if folded in _NO:
        return False
    tokens = folded.split()
    if tokens and tokens[0] in {"ja", "okay", "ok", "yes"}:
        return True
    if tokens and tokens[0] in {"nein", "nee", "no", "stopp", "halt"}:
        return False
    return None


def confirmation_prompt(title: str, effect: str, risk: str) -> str:
    """What KIKI asks before a gated side effect."""
    body = effect.strip() or title.strip()
    if len(body) > 220:
        body = body[:217] + "…"
    if risk in {"write", "external", "destructive"}:
        return f"Das ist folgenschwer. Ich würde Folgendes tun: {body} Ja oder nein?"
    return f"Soll ich das tun: {body} Ja oder nein?"
