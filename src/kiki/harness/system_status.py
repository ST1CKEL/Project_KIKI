"""The one tool this harness knows: a small, local, read-only status.

What it may report is the narrow question "is the harness there and how long has
this process been up, roughly". What it may not report is anything that
identifies the machine or the person: no hostname, no user, no home path, no
environment, no process list, no hardware. It runs no shell and reads no file.

Uptime is a category rather than a number, because a number is a fingerprint and
a category answers the only question a model would ask.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from kiki.harness.models import ToolResult

HARNESS_VERSION = "1"
SERVICE = "kiki"

# Bucket edges in seconds. Coarse on purpose.
_FRESH_S = 60
_RECENT_S = 3600

_PROCESS_START = time.monotonic()


def uptime_category(seconds: float) -> str:
    if seconds < _FRESH_S:
        return "fresh"
    if seconds < _RECENT_S:
        return "recent"
    return "long"


class SystemStatusTool:
    """`system_status`. Takes no arguments and changes nothing."""

    name = "system_status"
    description = "Meldet den lokalen KIKI-Harness-Status. Nimmt keine Argumente."
    read_only = True
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, *, uptime: Callable[[], float] | None = None) -> None:
        # Injected so the tool is fully deterministic in tests without patching
        # the clock globally.
        self._uptime = uptime or (lambda: time.monotonic() - _PROCESS_START)

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        del arguments  # the schema already refused anything but {}
        return ToolResult(
            # The registry stamps the real call id; a tool never sees one.
            call_id="",
            name=self.name,
            ok=True,
            data={
                "ok": True,
                "service": SERVICE,
                "agent_harness": "available",
                "harness_version": HARNESS_VERSION,
                "uptime": uptime_category(self._uptime()),
            },
        )
