"""Network-backed Edge TTS and a small Windows playback adapter."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from interfaces.speech import AudioPlayer, SpeechSynthesizer, SynthesizedSpeech


class EdgeTTS(SpeechSynthesizer):
    """Synthesize with Microsoft's online Edge TTS service."""

    def __init__(
        self,
        *,
        voice: str | None = None,
        rate: str | None = None,
        volume: str | None = None,
    ) -> None:
        self.voice = voice or os.getenv("AIL_TTS_VOICE", "en-US-AriaNeural")
        self.rate = rate or os.getenv("AIL_TTS_RATE", "+0%")
        self.volume = volume or os.getenv("AIL_TTS_VOLUME", "+0%")

    async def synthesize(self, text: str) -> SynthesizedSpeech:
        if not text.strip():
            raise ValueError("cannot synthesize empty text")
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("edge-tts is required for online speech synthesis") from exc

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as output:
            path = Path(output.name)
        try:
            communicate = edge_tts.Communicate(
                text,
                self.voice,
                rate=self.rate,
                volume=self.volume,
            )
            await communicate.save(str(path))
            audio = await asyncio.to_thread(path.read_bytes)
        finally:
            path.unlink(missing_ok=True)
        return SynthesizedSpeech(audio=audio, mime_type="audio/mpeg", filename="speech.mp3")


class WindowsAudioPlayer(AudioPlayer):
    """Play MP3/WAV bytes synchronously in a worker via playsound3."""

    async def play(self, speech: SynthesizedSpeech) -> None:
        try:
            from playsound3 import playsound
        except ImportError as exc:
            raise RuntimeError("playsound3 is required for Windows audio playback") from exc

        suffix = Path(speech.filename).suffix or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as output:
            output.write(speech.audio)
            path = Path(output.name)
        try:
            await asyncio.to_thread(playsound, str(path), block=True)
        finally:
            path.unlink(missing_ok=True)