"""Allowlist of declared tools. Nothing not in this registry can run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from kiki.tools.policy import RiskLevel

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    title: str
    description: str
    risk: RiskLevel
    parameters: dict[str, Any]
    handler: ToolHandler
    effect: str
    target: str = "local"
    auto_allow: bool = False
    requires_integration: bool = True
    allowed_profiles: tuple[str, ...] = ("observe", "develop")
    allowed_in_panic: bool = False
    sensitive_parameters: tuple[str, ...] = ()
    # Whether the model may request this tool itself. Default deny: a tool is
    # reachable by the agent loop only when its author opted in explicitly.
    model_callable: bool = False


@dataclass
class ActionPreview:
    tool: str
    title: str
    params: dict[str, Any]
    target: str
    effect: str
    risk: RiskLevel
    reason: str


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> Iterable[ToolSpec]:
        return self._tools.values()

    def preview(self, spec: ToolSpec, params: dict[str, Any], reason: str) -> ActionPreview:
        return ActionPreview(
            tool=spec.name,
            title=spec.title,
            params=params,
            target=spec.target,
            effect=spec.effect,
            risk=spec.risk,
            reason=reason,
        )
