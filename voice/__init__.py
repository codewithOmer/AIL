"""AIL Voice: microphone capture, speech-to-text, synthesis, and playback.

This package owns the voice feature workspace.  The public surface is the
voice service and its result types; concrete adapters live in the ``audio``,
``stt`` and ``tts`` subpackages and are wired by the composition root.
"""

from voice.service import VoiceResult, VoiceService, response_text

__all__ = ["VoiceResult", "VoiceService", "response_text"]
