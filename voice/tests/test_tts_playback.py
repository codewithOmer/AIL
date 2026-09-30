"""Offline tests for the Edge TTS synthesis and Windows audio playback adapters."""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from voice.tts.edge_tts import EdgeTTS, WindowsAudioPlayer
from interfaces.speech import AudioPlayer, SpeechSynthesizer, SynthesizedSpeech


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
