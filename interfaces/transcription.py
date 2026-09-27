"""AIL speech-to-text interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Transcription:
    """Recognized text and optional detected language."""

    text: str
    language: str | None = None


class Transcriber(ABC):
    """Interface for asynchronous audio transcription."""

    name: str

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        *,
        mime_type: str,
        filename: str,
    ) -> Transcription:
        """Transcribe audio bytes into text."""
        raise NotImplementedError
