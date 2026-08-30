"""What jarvis autonomy actually means, per tool — not per risk level.

The policy's level table is a blanket: at `jarvis`, every WRITE tool whose
author set `auto_allow` could run unattended. That is broader than any review
can follow once more write tools exist. EXTERNAL is stopped by the base policy
at every level. This module is the tool-sharp layer for the remaining write
headroom: the level says how much there is, the spec says which tools may use it.

Two sets carry the whole decision:

* `unattended_writes` — the write tools that may run unattended at
  jarvis, by name, in one reviewable place. Everything else with the same
  risk keeps its approval card: the blanket granted, the spec withheld.
* `never_unattended` — tools that never run unattended, whatever any level,
  list or `auto_allow` flag says. They hold what shapes every later answer
  (memory), act on standing authority (routines), or speak silently into the
  user's desktop (clipboard, notification). The author veto via
  `auto_allow=False` is the first lock; this is the second, and it holds even
  if the first is ever flipped.

What this module deliberately does not do: override a DENY (the hard deny
list, panic, the integration lockout and unknown tools stay exactly as the
policy decided them), touch origins other than the model (a clicked action
and a confirmed routine keep their own rules), or widen anything — `sharpen`
can only turn an unattended run into a card, never a card into a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kiki.tools.policy import AutonomyLevel, DecisionKind, Origin, PolicyDecision, RiskLevel

# The documented jarvis deal, named tool by tool: machine power, because the
# architecture sold jarvis as exactly that trade. Adding a name here is a
# reviewed decision; not adding it is the default.
JARVIS_UNATTENDED_WRITES: frozenset[str] = frozenset({"power.reboot", "power.poweroff"})

# Never unattended, on any level, even if listed capable, even if
# `auto_allow` says otherwise. These tools shape later answers, act on
# standing authority, or reach the desktop silently.
NEVER_UNATTENDED: frozenset[str] = frozenset(
    {
        "memory_remember",
        "memory_forget",
        "routines.create",
        "routines.delete",
        "desktop.copy_text",
        "desktop.show_notification",
        "create_note",
    }
)


@dataclass(frozen=True)
class AutonomySpec:
    """One level's tool-sharp limits. Data, not logic: `sharpen` is the logic."""

    level: AutonomyLevel
    unattended_writes: frozenset[str] = frozenset()
    never_unattended: frozenset[str] = frozenset()


JARVIS_SPEC = AutonomySpec(
    level=AutonomyLevel.JARVIS,
    unattended_writes=JARVIS_UNATTENDED_WRITES,
    never_unattended=NEVER_UNATTENDED,
)

_SPECS: dict[AutonomyLevel, AutonomySpec] = {
    AutonomyLevel.JARVIS: JARVIS_SPEC,
    # The levels below carry no WRITE headroom at all — the policy's
    # table already cards everything this spec would guard. They get the
    # empty spec so `sharpen` has one code path and no level is special.
    AutonomyLevel.STRICT: AutonomySpec(level=AutonomyLevel.STRICT),
    AutonomyLevel.BALANCED: AutonomySpec(level=AutonomyLevel.BALANCED),
    AutonomyLevel.TRUSTED: AutonomySpec(level=AutonomyLevel.TRUSTED),
}


def spec_for(level: AutonomyLevel | str) -> AutonomySpec:
    """The spec for a level. An unknown level names no tools — fail closed."""
    try:
        resolved = level if isinstance(level, AutonomyLevel) else AutonomyLevel(level)
    except ValueError:
        return AutonomySpec(level=AutonomyLevel.STRICT)
    return _SPECS.get(resolved, AutonomySpec(level=resolved))


def sharpen(
    decision: PolicyDecision,
    *,
    name: str,
    spec: Any | None,
    origin: Origin,
    autonomy: AutonomySpec,
) -> PolicyDecision:
    """Apply the tool-sharp limit to an allowed, model-initiated call.

    Returns the input decision unless the level's blanket said "run" and the
    spec says "ask first" — then a CONFIRM decision over the same validated
    parameters, so the card and the grant bind exactly what would have run.
    Never the other way round: this function cannot make anything run.
    """
    if decision.kind is not DecisionKind.ALLOW:
        return decision
    if origin is not Origin.MODEL:
        return decision
    if autonomy.level is not AutonomyLevel.JARVIS:
        return decision
    if name in autonomy.never_unattended:
        return PolicyDecision(
            kind=DecisionKind.CONFIRM,
            reason="Dieses Werkzeug läuft nie unbeaufsichtigt — Freigabe nötig.",
            risk=decision.risk,
            params=decision.params,
        )
    risk = getattr(spec, "risk", None) if spec is not None else None
    if risk in (RiskLevel.WRITE, RiskLevel.EXTERNAL) and name not in autonomy.unattended_writes:
        return PolicyDecision(
            kind=DecisionKind.CONFIRM,
            reason=(
                "Die Jarvis-Stufe deckt dieses Werkzeug nicht — "
                "diese Aktion braucht eine Freigabe."
            ),
            risk=decision.risk,
            params=decision.params,
        )
    return decision
