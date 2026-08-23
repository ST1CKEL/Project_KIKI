"""The registry gate: only registered names, only valid arguments, no leaks."""

from __future__ import annotations

import asyncio

import pytest

from kiki.harness.models import ToolCall, ToolResult
from kiki.harness.system_status import SystemStatusTool, uptime_category
from kiki.harness.tools import Tool, ToolRegistry
from tests.agent_fakes import FailingTool


def _registry(tool=None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool or SystemStatusTool(uptime=lambda: 5.0))
    return registry


def test_the_status_tool_satisfies_the_protocol() -> None:
    assert isinstance(SystemStatusTool(), Tool)


def test_only_registered_tools_exist() -> None:
    registry = _registry()
    assert registry.names == ("system_status",)
    assert registry.get("shell") is None
    assert registry.validate(ToolCall("shell")) == "unknown_tool"


def test_registering_the_same_name_twice_is_refused() -> None:
    registry = _registry()
    with pytest.raises(ValueError):
        registry.register(SystemStatusTool())


def test_the_schema_states_the_name_description_input_and_class() -> None:
    schema = _registry().schemas()[0]
    assert schema["name"] == "system_status"
    assert schema["description"]
    assert schema["read_only"] is True
    assert schema["input_schema"]["additionalProperties"] is False


def test_the_status_tool_takes_no_arguments() -> None:
    registry = _registry()
    assert registry.validate(ToolCall("system_status", {})) == ""
    assert registry.validate(ToolCall("system_status", {"verbose": True})) == "invalid_arguments"
    assert registry.validate(ToolCall("system_status", {"": ""})) == "invalid_arguments"


def test_non_dict_arguments_are_refused() -> None:
    registry = _registry()
    call = ToolCall("system_status")
    object.__setattr__(call, "arguments", ["nicht", "dict"])
    assert registry.validate(call) == "invalid_arguments"


def test_an_unknown_tool_never_runs() -> None:
    result = asyncio.run(_registry().execute(ToolCall("rm")))
    assert result.ok is False
    assert result.error_code == "unknown_tool"


def test_bad_arguments_stop_the_tool_before_it_starts() -> None:
    ran: list[int] = []

    class _Watching(SystemStatusTool):
        async def execute(self, arguments):
            ran.append(1)
            return await super().execute(arguments)

    registry = ToolRegistry()
    registry.register(_Watching())
    result = asyncio.run(registry.execute(ToolCall("system_status", {"x": 1})))

    assert result.error_code == "invalid_arguments"
    assert ran == []


def test_a_raising_tool_becomes_a_category_not_a_message() -> None:
    result = asyncio.run(_registry(FailingTool()).execute(ToolCall("system_status")))
    assert result.ok is False
    assert result.error_code == "tool_failed"
    assert result.data is None
    assert "/home/" not in repr(result)
    assert "geheim" not in repr(result)


def test_every_result_points_at_the_call_that_produced_it() -> None:
    call = ToolCall("system_status")
    result = asyncio.run(_registry().execute(call))
    assert result.call_id == call.id
    assert result.name == call.name


# --- what the status tool may say -------------------------------------------


def test_the_status_is_small_and_harmless() -> None:
    result = asyncio.run(_registry().execute(ToolCall("system_status")))
    assert result.ok is True
    assert result.data == {
        "ok": True,
        "service": "kiki",
        "agent_harness": "available",
        "harness_version": "1",
        "uptime": "fresh",
    }


def test_the_status_names_nothing_about_the_machine() -> None:
    """No hostname, no user, no home, no environment, no hardware."""
    import getpass
    import os
    import socket

    text = repr(asyncio.run(_registry().execute(ToolCall("system_status"))).data)
    for secret in (socket.gethostname(), getpass.getuser(), os.path.expanduser("~")):
        if secret:
            assert secret not in text, secret
    assert "/" not in text.replace("kiki", "")


@pytest.mark.parametrize(
    ("seconds", "category"), [(0.0, "fresh"), (59.9, "fresh"), (60.0, "recent"),
                              (3599.0, "recent"), (3600.0, "long"), (99999.0, "long")]
)
def test_uptime_is_a_bucket_not_a_number(seconds, category) -> None:
    """A precise uptime is a fingerprint; a bucket answers the same question."""
    assert uptime_category(seconds) == category


def test_the_status_tool_is_deterministic() -> None:
    tool = SystemStatusTool(uptime=lambda: 5.0)
    first = asyncio.run(tool.execute({}))
    second = asyncio.run(tool.execute({}))
    assert first.data == second.data


def test_a_tool_result_cannot_claim_success_and_failure() -> None:
    with pytest.raises(ValueError):
        ToolResult(call_id="c", name="t", ok=True, error_code="tool_failed")
    with pytest.raises(ValueError):
        ToolResult(call_id="c", name="t", ok=False, error_code="ausgedacht")
