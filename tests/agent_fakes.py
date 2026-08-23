"""Model adapters with no model behind them. GPU-free, network-free, exact.

Each one exists to drive the runner into one specific corner: answer at once,
use the tool once, keep asking, hand back nonsense, or block until cancelled.
"""

from __future__ import annotations

import asyncio
from typing import Any

from kiki.harness.models import ActionKind, CancelToken, ModelAction, ToolResult


class _Base:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_observations: list[list[ToolResult]] = []

    def _record(self, observations: list[ToolResult]) -> None:
        self.calls += 1
        self.seen_observations.append(list(observations))


class FinalOnlyModel(_Base):
    """Answers immediately, never asks for a tool."""

    def __init__(self, text: str = "Alles in Ordnung.") -> None:
        super().__init__()
        self._text = text

    async def next_action(self, *, user_text, tool_schemas, observations, cancel_token):
        del user_text, tool_schemas, cancel_token
        self._record(observations)
        return ModelAction.answer(self._text)


class OneToolThenFinalModel(_Base):
    """One tool call, then an answer built from what it saw."""

    def __init__(self, tool: str = "system_status", arguments: dict[str, Any] | None = None):
        super().__init__()
        self._tool = tool
        self._arguments = arguments or {}

    async def next_action(self, *, user_text, tool_schemas, observations, cancel_token):
        del user_text, tool_schemas, cancel_token
        self._record(observations)
        if not observations:
            return ModelAction.call(self._tool, self._arguments)
        return ModelAction.answer("Der Harness ist erreichbar.")


class RepeatedToolModel(_Base):
    """Never stops asking for the tool. Drives the limits."""

    def __init__(self, tool: str = "system_status") -> None:
        super().__init__()
        self._tool = tool

    async def next_action(self, *, user_text, tool_schemas, observations, cancel_token):
        del user_text, tool_schemas, cancel_token
        self._record(observations)
        return ModelAction.call(self._tool)


class InvalidActionModel(_Base):
    """Returns something the runner must refuse."""

    def __init__(self, action: Any = None) -> None:
        super().__init__()
        self._action = action if action is not None else ModelAction(ActionKind.FINAL)

    async def next_action(self, *, user_text, tool_schemas, observations, cancel_token):
        del user_text, tool_schemas, cancel_token
        self._record(observations)
        return self._action


class WaitingModel(_Base):
    """Blocks until the run is cancelled, then returns an answer nobody uses."""

    def __init__(self, *, poll_s: float = 0.001) -> None:
        super().__init__()
        self._poll = poll_s
        self.released = False

    async def next_action(self, *, user_text, tool_schemas, observations, cancel_token: CancelToken):
        del user_text, tool_schemas
        self._record(observations)
        for _ in range(5000):
            if cancel_token.cancelled:
                self.released = True
                return ModelAction.answer("zu spät")
            await asyncio.sleep(self._poll)
        raise AssertionError("WaitingModel wurde nie freigegeben")


class CancelAfterObservationModel(_Base):
    """Asks for the tool, then cancels its own run before the next step."""

    def __init__(self, runner_ref: list[Any]) -> None:
        super().__init__()
        self._runner_ref = runner_ref

    async def next_action(self, *, user_text, tool_schemas, observations, cancel_token):
        del user_text, tool_schemas, cancel_token
        self._record(observations)
        if not observations:
            return ModelAction.call("system_status")
        raise AssertionError("nach dem Cancel darf kein zweiter Schritt kommen")


class FailingTool:
    """A registered tool that raises. The registry must turn that into a code."""

    name = "system_status"
    description = "Meldet den lokalen KIKI-Harness-Status. Nimmt keine Argumente."
    read_only = True
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    async def execute(self, arguments):
        del arguments
        raise RuntimeError("/home/martin/geheim.txt konnte nicht gelesen werden")
