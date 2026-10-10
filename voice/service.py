"""Voice orchestration around AIL's existing text application path."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from interfaces.audio_input import AudioInput
from interfaces.speech import AudioPlayer, SpeechSynthesizer, SynthesizedSpeech
from interfaces.transcription import Transcriber
from core.reporting import report_text


class TextApplication(Protocol):
    def handle(self, message: str): ...


@dataclass(frozen=True)
class VoiceResult:
    transcription: str
    response: str
    speech: SynthesizedSpeech


def response_text(result: object) -> str:
    """Convert the existing application result into concise speech text."""
    if isinstance(result, tuple):
        return f"Remembered {len(result)} item" + ("." if len(result) == 1 else "s.")

    return report_text(result)


class VoiceService:
    """Compose capture, transcription, text execution, synthesis, and playback."""

    def __init__(
        self,
        *,
        audio_input: AudioInput,
        transcriber: Transcriber,
        application: TextApplication,
        synthesizer: SpeechSynthesizer,
        player: AudioPlayer,
    ) -> None:
        self.audio_input = audio_input
        self.transcriber = transcriber
        self.application = application
        self.synthesizer = synthesizer
        self.player = player

    async def run_once(self) -> VoiceResult:
        audio = await self.audio_input.record()
        transcription = await self.transcriber.transcribe(
            audio,
            mime_type="audio/wav",
            filename="microphone.wav",
        )
        if not transcription.text.strip():
            raise ValueError("microphone recording did not contain speech")
        response = response_text(self.application.handle(transcription.text))
        speech = await self.synthesizer.synthesize(response)
        await self.player.play(speech)
        return VoiceResult(
            transcription=transcription.text,
            response=response,
            speech=speech,
        )

    def run_once_sync(self) -> VoiceResult:
        return asyncio.run(self.run_once())