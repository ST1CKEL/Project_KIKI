"""Adapter lookup. Does not spawn processes itself."""

from __future__ import annotations

from kiki.agents.models import AgentError
from kiki.agents.opencode import OpenCodeAdapter


class AgentBroker:
    def __init__(
        self,
        *,
        opencode_binary: str = "opencode",
        stop_grace_seconds: float = 2.0,
    ) -> None:
        self._adapters = {
            "opencode": OpenCodeAdapter(
                opencode_binary,
                stop_grace_seconds=stop_grace_seconds,
            )
        }

    def get(self, name: str) -> OpenCodeAdapter:
        adapter = self._adapters.get(name)
        if adapter is None:
            raise AgentError("unknown_agent", f"Unbekannter Agent: {name}")
        return adapter

    def names(self) -> tuple[str, ...]:
        return tuple(self._adapters)
