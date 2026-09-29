"""AIL-owned contracts for speech synthesis and playback."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SynthesizedSpeech:
    """Audio bytes and enough metadata for a playback adapter."""

    audio: bytes
    mime_type: str
    filename: str


class SpeechSynthesizer(ABC):
    """Asynchronously convert text into playable audio."""

    @abstractmethod
    async def synthesize(self, text: str) -> SynthesizedSpeech:
        raise NotImplementedError


class AudioPlayer(ABC):
    """Asynchronously play synthesized audio."""

    @abstractmethod
    async def play(self, speech: SynthesizedSpeech) -> None:
        raise NotImplementedError