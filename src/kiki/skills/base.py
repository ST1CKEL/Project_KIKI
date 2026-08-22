"""Skill contract for Phase 2+. A skill owns zero or more declared tools."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kiki.tools.registry import ToolSpec


@runtime_checkable
class Skill(Protocol):
    id: str
    name: str
    description: str

    def tools(self) -> list[ToolSpec]: ...
