"""Offline tests for the local voice vertical slice."""

from __future__ import annotations

import asyncio
import io
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from core.voice import VoiceService, response_text
from integrations.audio.microphone import (
    MAX_RECORD_SECONDS,
    MicrophoneError,
    SoundDeviceMicrophone,
)
from integrations.tts.edge_tts import EdgeTTS, WindowsAudioPlayer
from interfaces.audio_input import AudioInput
from interfaces.speech import AudioPlayer, SpeechSynthesizer, SynthesizedSpeech
from interfaces.transcription import Transcriber, Transcription
from interfaces.planning import ExecutionReport, Goal


class _Samples:
    def tobytes(self) -> bytes:
        return b"\x01\x00\x02\x00"


class TestMicrophone(unittest.TestCase):
    def test_valid_duration_is_accepted(self) -> None:
        microphone = SoundDeviceMicrophone(duration=MAX_RECORD_SECONDS)

        self.assertEqual(microphone.duration, MAX_RECORD_SECONDS)

    def test_invalid_duration_is_rejected(self) -> None:
        for duration in (0, -1, MAX_RECORD_SECONDS + 1):
            with self.subTest(duration=duration):
                with self.assertRaises(ValueError):
                    SoundDeviceMicrophone(duration=duration)

    def test_configuration_is_mono_int16_and_returns_wav_bytes(self) -> None:
        microphone = SoundDeviceMicrophone(sample_rate=8000, duration=0.5)
        sounddevice = types.SimpleNamespace(rec=MagicMock(return_value=_Samples()), wait=MagicMock())

        with patch.dict(sys.modules, {"sounddevice": sounddevice}):
            audio = asyncio.run(microphone.record())

        sounddevice.rec.assert_called_once_with(
            4000, samplerate=8000, channels=1, dtype="int16"
        )
        sounddevice.wait.assert_called_once_with()
        self.assertTrue(audio.startswith(b"RIFF"))
        with __import__("wave").open(io.BytesIO(audio), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getframerate(), 8000)
            self.assertEqual(wav.getsampwidth(), 2)

    def test_device_errors_become_microphone_errors(self) -> None:
        sounddevice = types.SimpleNamespace(rec=MagicMock(side_effect=OSError("no device")))

        with patch.dict(sys.modules, {"sounddevice": sounddevice}):
            with self.assertRaisesRegex(MicrophoneError, "audio device unavailable"):
                asyncio.run(SoundDeviceMicrophone().record())

    def test_implements_audio_input(self) -> None:
        self.assertIsInstance(SoundDeviceMicrophone(), AudioInput)


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


class TestEdgeTTS(unittest.TestCase):
    def test_synthesizer_uses_configured_provider_options(self) -> None:
        communicator = MagicMock()
        communicator.save = AsyncMock()
        module = types.SimpleNamespace(Communicate=MagicMock(return_value=communicator))
        with patch.dict(sys.modules, {"edge_tts": module}), patch(
            "pathlib.Path.read_bytes", return_value=b"audio"
        ):
            result = asyncio.run(
                EdgeTTS(voice="voice", rate="+10%", volume="-5%").synthesize("hello")
            )

        module.Communicate.assert_called_once_with(
            "hello", "voice", rate="+10%", volume="-5%"
        )
        communicator.save.assert_awaited_once()
        self.assertEqual(result, SynthesizedSpeech(b"audio", "audio/mpeg", "speech.mp3"))

    def test_player_writes_audio_and_calls_playback(self) -> None:
        playsound = MagicMock()
        module = types.SimpleNamespace(playsound=playsound)
        speech = SynthesizedSpeech(b"audio", "audio/mpeg", "speech.mp3")
        with patch.dict(sys.modules, {"playsound3": module}):
            asyncio.run(WindowsAudioPlayer().play(speech))

        playsound.assert_called_once()
        self.assertTrue(playsound.call_args.kwargs["block"])

    def test_interfaces_are_explicit(self) -> None:
        self.assertIsInstance(EdgeTTS(), SpeechSynthesizer)
        self.assertIsInstance(WindowsAudioPlayer(), AudioPlayer)


if __name__ == "__main__":
    unittest.main()