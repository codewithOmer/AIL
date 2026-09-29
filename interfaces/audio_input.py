"""AIL-owned contracts for capturing audio."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AudioInput(ABC):
    """Asynchronously capture one bounded audio recording."""

    sample_rate: int
    channels: int

    @abstractmethod
    async def record(self) -> bytes:
        """Return one recording as audio bytes."""
        raise NotImplementedError