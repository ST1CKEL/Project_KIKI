"""Default-deny tool policy. Unknown and hard-denied names never run."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from kiki.tools.schemas import ValidationError, validate_params

HARD_DENY: frozenset[str] = frozenset(
    {
        "run_shell",
        "shell",
        "exec",
        "spawn",
        "sudo",
        "su",
        "pkexec",
        "bash",
        "sh",
        "delete_file",
        "rm",
        "send_message",
        "send_email",
        "raw_http",
        "write_file",
        "install_package",
    }
)

VALID_PROFILES: tuple[str, ...] = ("observe", "develop", "operator")


class RiskLevel(StrEnum):
    READ = "read"
    CONTROL = "control"
    # Opens something visible on the user's own desktop. Changes no data and
    # nothing it opens can act on its own — the user still drives whatever
    # appears. Kept apart from WRITE so "open my project folder" does not have
    # to be classified as a data change to be allowed.
    LAUNCH = "launch"
    WRITE = "write"
    EXTERNAL = "external"
    FORBIDDEN = "forbidden"


class DecisionKind(StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


class Origin(StrEnum):
    """Who asked for the call. The user clicking is not the model deciding."""

    USER = "user"
    MODEL = "model"
    # A stored routine firing in the background. The recipe — tool, arguments,
    # trigger — was confirmed by a human when it was created; see kiki/routines.
    ROUTINE = "routine"


class AutonomyLevel(StrEnum):
    """How much a model-initiated call may do before a human sees a dialog."""

    STRICT = "strict"
    BALANCED = "balanced"
    TRUSTED = "trusted"
    # Opt-in "Jarvis" mode: the user explicitly accepted that KIKI may also
    # write unattended. External actions remain confirmation-only at every
    # level. It is never a default. What still stops a call down here: the hard
    # deny list, unknown tools, the panic switch, the integration lockout — and
    # every spec whose author withheld `auto_allow`.
    JARVIS = "jarvis"


VALID_AUTONOMY: tuple[str, ...] = tuple(level.value for level in AutonomyLevel)

# Risk tiers a model-initiated call may trigger unattended, per level. WRITE
# appears in exactly one place: JARVIS, an explicitly chosen mode. EXTERNAL is
# deliberately absent everywhere: leaving the machine always needs a fresh,
# interactive confirmation.
_UNATTENDED: dict[AutonomyLevel, frozenset[RiskLevel]] = {
    AutonomyLevel.STRICT: frozenset({RiskLevel.READ}),
    AutonomyLevel.BALANCED: frozenset({RiskLevel.READ, RiskLevel.CONTROL}),
    AutonomyLevel.TRUSTED: frozenset(
        {RiskLevel.READ, RiskLevel.CONTROL, RiskLevel.LAUNCH}
    ),
    AutonomyLevel.JARVIS: frozenset(
        {
            RiskLevel.READ,
            RiskLevel.CONTROL,
            RiskLevel.LAUNCH,
            RiskLevel.WRITE,
        }
    ),
}


@dataclass(frozen=True)
class PolicyDecision:
    kind: DecisionKind
    reason: str
    risk: RiskLevel = RiskLevel.FORBIDDEN
    params: dict[str, Any] | None = None


class ToolPolicy:
    def __init__(self, autonomy: str = AutonomyLevel.BALANCED.value) -> None:
        self._autonomy = self._coerce(autonomy)

    @property
    def autonomy(self) -> AutonomyLevel:
        return self._autonomy

    def set_autonomy(self, autonomy: str) -> None:
        self._autonomy = self._coerce(autonomy)

    @staticmethod
    def _coerce(autonomy: str) -> AutonomyLevel:
        # An unreadable config must not silently widen what the model may do.
        try:
            return AutonomyLevel((autonomy or "").strip().lower())
        except ValueError:
            return AutonomyLevel.STRICT

    def evaluate(
        self,
        *,
        name: str,
        params: dict[str, Any] | None,
        spec: Any | None,
        panic: bool,
        integrations_enabled: bool,
        profile: str = "observe",
        origin: Origin = Origin.USER,
    ) -> PolicyDecision:
        blocked = self.gate(
            name=name,
            spec=spec,
            panic=panic,
            integrations_enabled=integrations_enabled,
            profile=profile,
            origin=origin,
        )
        if blocked is not None:
            return blocked
        assert spec is not None
        try:
            cleaned = validate_params(spec.parameters, params or {})
        except ValidationError as exc:
            return PolicyDecision(
                kind=DecisionKind.DENY,
                reason=f"Ungültige Parameter: {exc}",
                risk=spec.risk,
            )
        if spec.risk is RiskLevel.EXTERNAL:
            # Product invariant, stronger than profiles and autonomy levels:
            # every action that leaves KIKI's local boundary gets a fresh
            # interactive confirmation. A routine has nobody present to
            # answer that card, so it must fail closed instead of treating an
            # old recipe approval as a reusable external grant.
            if origin is Origin.ROUTINE:
                return PolicyDecision(
                    kind=DecisionKind.DENY,
                    reason="Externe Aktionen dürfen nicht unbeaufsichtigt als Routine laufen.",
                    risk=spec.risk,
                    params=cleaned,
                )
            return PolicyDecision(
                kind=DecisionKind.CONFIRM,
                reason="Externe Aktion — eine aktuelle Nutzerbestätigung ist immer erforderlich.",
                risk=spec.risk,
                params=cleaned,
            )
        if origin is Origin.ROUTINE:
            # The recipe behind this call was confirmed by a human when the
            # routine was created. What still has to hold: the tool's author
            # allowed unattended runs at all. Nobody can answer a card here.
            if not spec.auto_allow:
                return PolicyDecision(
                    kind=DecisionKind.DENY,
                    reason="Dieses Werkzeug ist nicht routinenfähig.",
                    risk=spec.risk,
                    params=cleaned,
                )
            return PolicyDecision(
                kind=DecisionKind.ALLOW,
                reason="Freigegebene Routine — Rezept wurde bei der Erstellung bestätigt.",
                risk=spec.risk,
                params=cleaned,
            )
        if not spec.auto_allow:
            # The spec author withheld unattended execution; no level overrides it.
            return PolicyDecision(
                kind=DecisionKind.CONFIRM,
                reason="Diese Aktion verlangt immer eine Nutzerbestätigung.",
                risk=spec.risk,
                params=cleaned,
            )
        if origin is Origin.MODEL:
            # A level with no table entry grants nothing, so adding a level
            # later fails closed instead of raising at the worst moment.
            if spec.risk not in _UNATTENDED.get(self._autonomy, frozenset()):
                return PolicyDecision(
                    kind=DecisionKind.CONFIRM,
                    reason=(
                        f"Von KIKI selbst angefordert ({spec.risk.value}) — "
                        f"Stufe „{self._autonomy.value}“ verlangt eine Freigabe."
                    ),
                    risk=spec.risk,
                    params=cleaned,
                )
            return PolicyDecision(
                kind=DecisionKind.ALLOW,
                reason=f"Von KIKI angefordert, Risikostufe {spec.risk.value}, Stufe {self._autonomy.value}.",
                risk=spec.risk,
                params=cleaned,
            )
        if spec.risk in {RiskLevel.READ, RiskLevel.CONTROL, RiskLevel.LAUNCH}:
            return PolicyDecision(
                kind=DecisionKind.ALLOW,
                reason=(
                    "Explizite Sicherheitssteuerung, ohne zusätzliche Bestätigung."
                    if spec.risk is RiskLevel.CONTROL
                    else (
                        "Expliziter lokaler Startbefehl — Ziel ist an die Allowlist gebunden."
                        if spec.risk is RiskLevel.LAUNCH
                        else "Lesendes Tool, Allowlist, ohne Bestätigung."
                    )
                ),
                risk=spec.risk,
                params=cleaned,
            )
        return PolicyDecision(
            kind=DecisionKind.CONFIRM,
            reason="Schreibende Aktion — Nutzerbestätigung erforderlich.",
            risk=spec.risk,
            params=cleaned,
        )

    def gate(
        self,
        *,
        name: str,
        spec: Any | None,
        panic: bool,
        integrations_enabled: bool,
        profile: str = "observe",
        origin: Origin = Origin.USER,
    ) -> PolicyDecision | None:
        """Everything decidable without looking at parameters.

        Returns a DENY decision, or None when the call may proceed to parameter
        validation. `exposure` uses it to decide which tools are worth showing
        to the model at all.
        """
        if name in HARD_DENY:
            return PolicyDecision(
                kind=DecisionKind.DENY,
                reason=f"Tool „{name}“ ist grundsätzlich verboten.",
                risk=RiskLevel.FORBIDDEN,
            )
        if spec is None:
            return PolicyDecision(
                kind=DecisionKind.DENY,
                reason=f"Unbekanntes Tool „{name}“ — Default Deny.",
                risk=RiskLevel.FORBIDDEN,
            )
        if spec.risk is RiskLevel.FORBIDDEN:
            return PolicyDecision(
                kind=DecisionKind.DENY,
                reason=f"Tool „{name}“ ist als forbidden registriert.",
                risk=RiskLevel.FORBIDDEN,
            )
        if panic and not bool(getattr(spec, "allowed_in_panic", False)):
            return PolicyDecision(
                kind=DecisionKind.DENY,
                reason="Privacy-/Panic-Schalter ist aktiv. Alle Integrationen sind deaktiviert.",
                risk=spec.risk,
            )
        if spec.requires_integration and not integrations_enabled:
            return PolicyDecision(
                kind=DecisionKind.DENY,
                reason="Integrationen sind deaktiviert.",
                risk=spec.risk,
            )
        current_profile = (profile or "observe").strip() or "observe"
        if current_profile not in VALID_PROFILES:
            return PolicyDecision(
                kind=DecisionKind.DENY,
                reason=f"Unbekanntes Berechtigungsprofil: {current_profile}",
                risk=spec.risk,
            )
        if current_profile == "operator":
            return PolicyDecision(
                kind=DecisionKind.DENY,
                reason="Profil „operator“ ist in dieser Phase deaktiviert.",
                risk=RiskLevel.FORBIDDEN,
            )
        allowed = tuple(getattr(spec, "allowed_profiles", ("observe", "develop")))
        if current_profile not in allowed:
            return PolicyDecision(
                kind=DecisionKind.DENY,
                reason=f"Profil „{current_profile}“ darf Tool „{name}“ nicht ausführen.",
                risk=spec.risk,
            )
        if origin is Origin.MODEL and not bool(getattr(spec, "model_callable", False)):
            return PolicyDecision(
                kind=DecisionKind.DENY,
                reason=f"Tool „{name}“ ist nicht für Modellaufrufe freigegeben — Default Deny.",
                risk=spec.risk,
            )
        return None
