"""AIL main entry point.

Demonstrates the AIL → Open Interpreter App Server vertical slice with
long-term memory: recall relevant memories before each turn, and store a
small set of explicit user-provided facts after each turn.
"""

import asyncio
import sys

from core.agent import Agent, MockLLM
from core.config import setup_logging
from core.planner import UnsupportedTaskError
from core.reporting import report_text


def run_mock_mode() -> None:
    """Original mock LLM mode."""
    agent = Agent(MockLLM())
    user_input = input("You: ")
    response = agent.respond(user_input)
    print(f"AIL: {response}")


def run_oi_mode() -> None:
    """Run supported verified tasks against a real OI App Server."""
    from core.application import AILApplication
    from integrations.open_interpreter.client import (
        OIConnectionError,
        OIError,
    )

    application: AILApplication | None = None
    try:
        application = AILApplication.create()

        while True:
            message = input("You: ").strip()
            if not message or message.lower() in ("exit", "quit"):
                break


            try:
                result = application.handle(message)
            except UnsupportedTaskError as exc:
                print(f"AIL: unsupported task ({exc})")
                continue

            if isinstance(result, tuple):
                continue

            _print_report(result)
    except OIConnectionError as exc:
        print(f"Failed to connect to OI App Server: {exc}", file=sys.stderr)
        sys.exit(1)
    except OIError as exc:
        print(f"OI error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if application is not None:
            application.close()


def run_voice_mode() -> None:
    """Run one fixed-window microphone -> AIL -> speaker turn."""
    from core.application import AILApplication
    from core.config import Config
    from integrations.open_interpreter.client import OIError
    from voice.audio.sounddevice_microphone import MicrophoneError, SoundDeviceMicrophone
    from voice.service import VoiceService
    from voice.stt.local_whisper import LocalWhisperTranscriber
    from voice.tts.edge_tts import EdgeTTS, WindowsAudioPlayer

    application: AILApplication | None = None
    try:
        application = AILApplication.create()
        service = VoiceService(
            audio_input=SoundDeviceMicrophone(
                sample_rate=Config.voice_sample_rate,
                duration=Config.voice_record_seconds,
            ),
            transcriber=LocalWhisperTranscriber(
                model=Config.stt_model,
                device=Config.stt_device,
                compute_type=Config.stt_compute_type,
            ),
            application=application,
            synthesizer=EdgeTTS(
                voice=Config.tts_voice,
                rate=Config.tts_rate,
                volume=Config.tts_volume,
            ),
            player=WindowsAudioPlayer(),
        )
        print("Listening...")
        result = asyncio.run(service.run_once())
        print(f"You said: {result.transcription}")
        print(f"AIL: {result.response}")
    except (MicrophoneError, OIError, RuntimeError, UnsupportedTaskError, ValueError) as exc:
        print(f"Voice error: {exc}", file=sys.stderr)
    finally:
        if application is not None:
            application.close()


def _print_report(report) -> None:
    """Print a concise result using only the verified execution report."""
    print(f"AIL: {report_text(report)}")


def main() -> None:
    setup_logging()
    mode = sys.argv[1] if len(sys.argv) > 1 else "mock"
    if mode == "voice":
        run_voice_mode()
    elif mode == "oi":
        run_oi_mode()
    else:
        run_mock_mode()


if __name__ == "__main__":
    main()
