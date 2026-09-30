"""Offline tests for the local microphone capture adapter."""

from __future__ import annotations

import asyncio
import io
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from voice.audio.sounddevice_microphone import (
    MAX_RECORD_SECONDS,
    MicrophoneError,
    SoundDeviceMicrophone,
)
from interfaces.audio_input import AudioInput


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


if __name__ == "__main__":
    unittest.main()
