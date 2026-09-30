"""AIL-local Whisper transcription via faster-whisper."""

from __future__ import annotations

import asyncio
import io
import os
from typing import Any

from interfaces.transcription import Transcriber, Transcription

_MODELS: dict[tuple[str, str, str, int], Any] = {}


def _load_model(
    model: str,
    device: str,
    compute_type: str,
    cpu_threads: int,
) -> Any:
    key = (model, device, compute_type, cpu_threads)
    cached = _MODELS.get(key)
    if cached is not None:
        return cached

    from faster_whisper import WhisperModel

    instance = WhisperModel(
        model,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )
    _MODELS[key] = instance
    return instance


class LocalWhisperTranscriber(Transcriber):
    """Transcribe audio in-process with a cached faster-whisper model."""

    name = "whisper-local"

    def __init__(
        self,
        *,
        model: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        cpu_threads: int = 0,
        language: str | None = None,
    ) -> None:
        self._model = model or os.getenv("AIL_STT_MODEL", "tiny")
        self._device = device or os.getenv("AIL_STT_DEVICE", "cpu")
        self._compute_type = compute_type or os.getenv(
            "AIL_STT_COMPUTE_TYPE", "int8"
        )
        self._cpu_threads = (
            cpu_threads if cpu_threads > 0 else min(os.cpu_count() or 4, 8)
        )
        self._language = language if language and language.lower() != "auto" else None

    def _transcribe_sync(self, audio: bytes) -> Transcription:
        model = _load_model(
            self._model,
            self._device,
            self._compute_type,
            self._cpu_threads,
        )
        segments, info = model.transcribe(
            io.BytesIO(audio),
            language=self._language,
            vad_filter=True,
        )
        text = "".join(segment.text for segment in segments).strip()
        return Transcription(text=text, language=getattr(info, "language", None))

    async def transcribe(
        self,
        audio: bytes,
        *,
        mime_type: str,
        filename: str,
    ) -> Transcription:
        del mime_type, filename
        return await asyncio.to_thread(self._transcribe_sync, audio)
