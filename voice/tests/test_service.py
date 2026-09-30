"""Offline tests for the voice service orchestration."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock

from voice.service import VoiceService, response_text
from interfaces.audio_input import AudioInput
from interfaces.speech import AudioPlayer, SpeechSynthesizer, SynthesizedSpeech
from interfaces.transcription import Transcriber, Transcription
from interfaces.planning import ExecutionReport, Goal


class _FakeInput(AudioInput):
    sample_rate = 16000
    channels = 1

    async def record(self) -> bytes:
        return b"audio"


class _FakeTranscriber(Transcriber):
    name = "fake"

    def __init__(self, text: str = "Create a file") -> None:
        self.text = text
        self.calls: list[tuple[bytes, str, str]] = []

    async def transcribe(self, audio: bytes, *, mime_type: str, filename: str) -> Transcription:
        self.calls.append((audio, mime_type, filename))
        return Transcription(self.text)


class _FakeSynthesizer(SpeechSynthesizer):
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def synthesize(self, text: str) -> SynthesizedSpeech:
        self.texts.append(text)
        return SynthesizedSpeech(b"mp3", "audio/mpeg", "speech.mp3")


class _FakePlayer(AudioPlayer):
    def __init__(self) -> None:
        self.speech: list[SynthesizedSpeech] = []

    async def play(self, speech: SynthesizedSpeech) -> None:
        self.speech.append(speech)


class _FakeApplication:
    def __init__(self, result: object) -> None:
        self.result = result
        self.messages: list[str] = []

    def handle(self, message: str) -> object:
        self.messages.append(message)
        return self.result


class TestVoiceService(unittest.TestCase):
    def setUp(self) -> None:
        self.report = ExecutionReport(goal=Goal("goal"), passed=True, steps=())
        self.application = _FakeApplication(self.report)
        self.transcriber = _FakeTranscriber("  Create a file  ")
        self.synthesizer = _FakeSynthesizer()
        self.player = _FakePlayer()
        self.service = VoiceService(
            audio_input=_FakeInput(),
            transcriber=self.transcriber,
            application=self.application,
            synthesizer=self.synthesizer,
            player=self.player,
        )

    def test_composes_audio_to_text_to_application_to_audio(self) -> None:
        result = asyncio.run(self.service.run_once())

        self.assertEqual(result.transcription, "  Create a file  ")
        self.assertEqual(result.response, "verified task completed")
        self.assertEqual(self.transcriber.calls, [(b"audio", "audio/wav", "microphone.wav")])
        self.assertEqual(self.application.messages, ["  Create a file  "])
        self.assertEqual(self.synthesizer.texts, ["verified task completed"])
        self.assertEqual(self.player.speech, [result.speech])

    def test_empty_transcription_stops_before_application(self) -> None:
        self.transcriber.text = "  "

        with self.assertRaisesRegex(ValueError, "did not contain speech"):
            asyncio.run(self.service.run_once())

        self.assertEqual(self.application.messages, [])

    def test_errors_propagate_without_speaking(self) -> None:
        self.synthesizer.synthesize = AsyncMock(side_effect=RuntimeError("tts failed"))

        with self.assertRaisesRegex(RuntimeError, "tts failed"):
            asyncio.run(self.service.run_once())

        self.assertEqual(self.player.speech, [])

    def test_response_text_handles_memory_and_failed_reports(self) -> None:
        self.assertEqual(response_text((object(),)), "Remembered 1 item.")
        failed = ExecutionReport(goal=Goal("goal"), passed=False, steps=())
        self.assertEqual(response_text(failed), "task failed (verification failure)")


if __name__ == "__main__":
    unittest.main()
