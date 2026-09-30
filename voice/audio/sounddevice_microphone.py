"""Small local microphone adapter backed by sounddevice."""

from __future__ import annotations

import asyncio
import io
import wave

from interfaces.audio_input import AudioInput

MAX_RECORD_SECONDS = 60.0


class MicrophoneError(RuntimeError):
    """The local microphone could not be opened or read."""


class SoundDeviceMicrophone(AudioInput):
    """Capture one mono int16 recording and return an in-memory WAV."""

    def __init__(self, *, sample_rate: int = 16000, duration: float = 5.0) -> None:
        if sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if duration <= 0:
            raise ValueError("duration must be positive")
        if duration > MAX_RECORD_SECONDS:
            raise ValueError(
                f"duration must be no more than {MAX_RECORD_SECONDS:g} seconds"
            )
        self.sample_rate = sample_rate
        self.channels = 1
        self.duration = duration

    def _record_sync(self) -> bytes:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise MicrophoneError(
                "sounddevice is required for microphone input"
            ) from exc

        frames = round(self.sample_rate * self.duration)
        try:
            samples = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
            )
            sd.wait()
        except Exception as exc:
            raise MicrophoneError(f"microphone/audio device unavailable: {exc}") from exc

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(samples.tobytes())
        return buffer.getvalue()

    async def record(self) -> bytes:
        return await asyncio.to_thread(self._record_sync)