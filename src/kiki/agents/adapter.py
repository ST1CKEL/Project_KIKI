"""Coding-agent adapter protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from kiki.agents.models import AgentAvailability, AgentEvent, AgentSession, AgentStartRequest


class CodingAgentAdapter(Protocol):
    name: str

    async def check_availability(self) -> AgentAvailability: ...

    async def start_session(self, request: AgentStartRequest) -> AgentSession: ...

    def stream_events(self, session_id: str) -> AsyncIterator[AgentEvent]: ...

    async def stop_session(self, session_id: str) -> None: ...
