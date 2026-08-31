"""Voice-facing security classes. Not a second executor.

Every call still goes through `ToolGateway`. These four names are how KIKI
talks about risk in speech and logs — they map onto the existing `RiskLevel`.
"""

from __future__ import annotations

from enum import StrEnum

from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec


class SecurityClass(StrEnum):
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    MODIFYING = "modifying"
    DESTRUCTIVE = "destructive"


def security_class(spec: ToolSpec) -> SecurityClass:
    if spec.risk is RiskLevel.READ:
        return SecurityClass.READ_ONLY
    if spec.risk is RiskLevel.CONTROL:
        return SecurityClass.REVERSIBLE
    if spec.risk is RiskLevel.LAUNCH:
        return SecurityClass.REVERSIBLE if spec.auto_allow else SecurityClass.MODIFYING
    if spec.risk is RiskLevel.WRITE and spec.auto_allow:
        return SecurityClass.MODIFYING
    # WRITE without auto_allow, EXTERNAL, FORBIDDEN — spoken confirmation,
    # never a hidden shell, never an unlisted delete.
    return SecurityClass.DESTRUCTIVE
