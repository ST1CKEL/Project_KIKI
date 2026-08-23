"""The tools a run may reach, and the gate in front of them.

Deliberately not `kiki.tools.ToolRegistry`: that one carries risk tiers, an
approval card, an audit log and a SQLite handle, because it guards actions a
user can trigger from the UI. This harness needs the opposite — a registry small
enough to reason about in one sitting, with a single read-only tool in it.

Nothing is reachable that was not registered, and nothing runs before its name
and its arguments have been checked.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from kiki.harness.models import ToolCall, ToolResult


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]
    # Read-only tools cannot change anything; this slice registers no others.
    read_only: bool

    async def execute(self, arguments: dict[str, Any]) -> ToolResult: ...


class ToolRegistry:
    """Explicit registration only. An unregistered name does not exist."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool bereits registriert: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def schemas(self) -> list[dict[str, Any]]:
        """What the adapter is told exists. Registered tools and nothing else."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.input_schema),
                "read_only": bool(tool.read_only),
            }
            for tool in (self._tools[name] for name in self.names)
        ]

    def validate(self, call: ToolCall) -> str:
        """Empty when the call may run; otherwise the category that stops it."""
        tool = self._tools.get(call.name)
        if tool is None:
            return "unknown_tool"
        return _check_arguments(call.arguments, tool.input_schema)

    async def execute(self, call: ToolCall) -> ToolResult:
        """Validate, then run. Never the other way round."""
        problem = self.validate(call)
        if problem:
            return ToolResult(call_id=call.id, name=call.name, ok=False, error_code=problem)
        tool = self._tools[call.name]
        try:
            result = await tool.execute(dict(call.arguments))
        except Exception:
            # The exception text could name a path or a value; the category
            # cannot, and the category is all a model needs to change course.
            return ToolResult(call_id=call.id, name=call.name, ok=False, error_code="tool_failed")
        if not isinstance(result, ToolResult):
            return ToolResult(call_id=call.id, name=call.name, ok=False, error_code="tool_failed")
        # The tool signature has no call id, so the registry stamps it. That is
        # what makes "a result always points at a call that happened" a property
        # of the harness rather than a promise each tool has to keep.
        return ToolResult(
            call_id=call.id,
            name=call.name,
            ok=result.ok,
            data=result.data,
            error_code=result.error_code,
        )


def _check_arguments(arguments: Any, schema: dict[str, Any]) -> str:
    """A deliberately small subset of JSON Schema: objects, required, no extras."""
    if not isinstance(arguments, dict):
        return "invalid_arguments"
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    if schema.get("additionalProperties", False) is False:
        if set(arguments) - set(properties):
            return "invalid_arguments"
    required = schema.get("required")
    required = required if isinstance(required, list | tuple) else ()
    if set(required) - set(arguments):
        return "invalid_arguments"
    return ""
