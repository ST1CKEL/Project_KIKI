from __future__ import annotations

from kiki.skills.base import Skill
from kiki.tools.registry import ToolRegistry, ToolSpec


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if skill.id in self._skills:
            raise ValueError(f"duplicate skill {skill.id}")
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def install_into(self, tools: ToolRegistry) -> None:
        for skill in self._skills.values():
            spec: ToolSpec
            for spec in skill.tools():
                tools.register(spec)
