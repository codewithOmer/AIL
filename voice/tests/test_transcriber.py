"""Unit tests for AIL's local Whisper transcriber."""

import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

from voice.stt import local_whisper
from interfaces.transcription import Transcriber, Transcription


class _Segment:
    def __init__(self, text: str) -> None:
        self.text = text


class _Info:
    language = "en"


class _FakeWhisperModel:
    instances: list[tuple[str, str, str, int]] = []
    calls: list[dict[str, object]] = []
    audio: list[bytes] = []
    segments: list[_Segment] = [_Segment(" hello "), _Segment(" world ")]
    error: Exception | None = None

    def __init__(
        self,
        model: str,
        *,
        device: str,
        compute_type: str,
        cpu_threads: int,
    ) -> None:
        self.instances.append((model, device, compute_type, cpu_threads))

    def transcribe(self, audio, **kwargs):
        self.audio.append(audio.read())
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.segments, _Info()


class TestLocalWhisperTranscriber(unittest.TestCase):
    def setUp(self) -> None:
        local_whisper._MODELS.clear()
        _FakeWhisperModel.instances.clear()
        _FakeWhisperModel.calls.clear()
        _FakeWhisperModel.audio.clear()
        _FakeWhisperModel.segments = [_Segment(" hello "), _Segment(" world ")]
        _FakeWhisperModel.error = None
        self.fake_module = types.ModuleType("faster_whisper")
        self.fake_module.WhisperModel = _FakeWhisperModel

    def _transcribe(self, transcriber: Transcriber, audio: bytes = b"RIFF-wav") -> Transcription:
        return asyncio.run(
            transcriber.transcribe(
                audio,
                mime_type="audio/wav",
                filename="sample.wav",
            )
        )

    def test_initializes_model_lazily(self) -> None:
        transcriber = local_whisper.LocalWhisperTranscriber()
        self.assertEqual(_FakeWhisperModel.instances, [])

        with patch.dict(sys.modules, {"faster_whisper": self.fake_module}):
            self._transcribe(transcriber)

        self.assertEqual(len(_FakeWhisperModel.instances), 1)

    def test_transcribes_wav_bytes_and_returns_text(self) -> None:
        transcriber = local_whisper.LocalWhisperTranscriber()

        with patch.dict(sys.modules, {"faster_whisper": self.fake_module}):
            result = self._transcribe(transcriber, b"RIFF-real-wav")

        self.assertEqual(result, Transcription(text="hello  world", language="en"))
        self.assertEqual(_FakeWhisperModel.audio, [b"RIFF-real-wav"])

    def test_passes_model_device_and_compute_type(self) -> None:
        transcriber = local_whisper.LocalWhisperTranscriber(
            model="base",
            device="cpu",
            compute_type="float32",
        )

        with patch.dict(sys.modules, {"faster_whisper": self.fake_module}):
            self._transcribe(transcriber)

        model, device, compute_type, cpu_threads = _FakeWhisperModel.instances[0]
        self.assertEqual((model, device, compute_type), ("base", "cpu", "float32"))
        self.assertGreaterEqual(cpu_threads, 1)

    def test_reuses_cached_model_and_enables_vad(self) -> None:
        transcriber = local_whisper.LocalWhisperTranscriber()

        with patch.dict(sys.modules, {"faster_whisper": self.fake_module}):
            self._transcribe(transcriber, b"one")
            self._transcribe(transcriber, b"two")

        self.assertEqual(len(_FakeWhisperModel.instances), 1)
        self.assertEqual(len(_FakeWhisperModel.calls), 2)
        self.assertTrue(all(call["vad_filter"] is True for call in _FakeWhisperModel.calls))

    def test_inference_runs_through_asyncio_to_thread(self) -> None:
        transcriber = local_whisper.LocalWhisperTranscriber()
        to_thread = AsyncMock(return_value=Transcription(text="threaded"))

        with patch.object(local_whisper.asyncio, "to_thread", to_thread):
            result = self._transcribe(transcriber)

        self.assertEqual(result, Transcription(text="threaded"))
        to_thread.assert_awaited_once_with(transcriber._transcribe_sync, b"RIFF-wav")

    def test_empty_transcription_returns_empty_text(self) -> None:
        _FakeWhisperModel.segments = []
        transcriber = local_whisper.LocalWhisperTranscriber()

        with patch.dict(sys.modules, {"faster_whisper": self.fake_module}):
            result = self._transcribe(transcriber)

        self.assertEqual(result, Transcription(text="", language="en"))

    def test_invalid_audio_error_is_propagated(self) -> None:
        _FakeWhisperModel.error = ValueError("invalid audio")
        transcriber = local_whisper.LocalWhisperTranscriber()

        with patch.dict(sys.modules, {"faster_whisper": self.fake_module}):
            with self.assertRaisesRegex(ValueError, "invalid audio"):
                self._transcribe(transcriber, b"not-wav")

    def test_implements_ail_transcriber(self) -> None:
        self.assertIsInstance(local_whisper.LocalWhisperTranscriber(), Transcriber)


if __name__ == "__main__":
    unittest.main()
