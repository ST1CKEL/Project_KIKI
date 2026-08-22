"""The contract every TTS backend fulfils.

Kept as a Protocol so a provider needs no base class and tests can supply a fake
without importing anything heavy.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from kiki.voice.tts.models import (
    AudioChunk,
    TTSHealth,
    TTSProviderCapabilities,
    TTSProviderStatus,
    TTSRequest,
)


@runtime_checkable
class TTSProvider(Protocol):
    """A speech backend.

    Implementations must be safe to call from the asyncio thread and must never
    block the GTK main loop. `load` is explicit so the core decides when a model
    occupies VRAM, and `unload` must actually release it.
    """

    provider_id: str

    @property
    def status(self) -> TTSProviderStatus:
        ...

    def capabilities(self) -> TTSProviderCapabilities:
        ...

    async def health_check(self) -> TTSHealth:
        ...

    async def load(self) -> None:
        """Bring the backend to READY. Calling it twice is not an error."""
        ...

    async def unload(self) -> None:
        """Release model and device memory. Safe when already unloaded."""
        ...

    def synthesize(self, request: TTSRequest) -> AsyncIterator[AudioChunk]:
        """Yield audio for one request.

        Must raise `TTSError` rather than a backend-specific exception, and must
        stop promptly once `cancel` was called for this request id.
        """
        ...

    async def cancel(self, request_id: str) -> None:
        """Abort one request by id. Unknown ids are ignored, never an error."""
        ...
