"""Read-only integration contract. Write actions never live here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class IntegrationError(Exception):
    """The integration is unavailable or the snapshot failed."""


@dataclass(frozen=True)
class IntegrationSnapshot:
    id: str
    title: str
    available: bool
    data: dict[str, Any]
    error: str | None = None


@runtime_checkable
class Integration(Protocol):
    id: str
    title: str

    def snapshot(self) -> IntegrationSnapshot: ...
