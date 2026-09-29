import logging
import os
from dotenv import load_dotenv


load_dotenv()

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: str | None = None) -> None:
    """Configure root logging for AIL."""
    log_level = level or os.getenv("AIL_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=LOG_FORMAT,
    )


class Config:
    """AIL configuration."""

    llm_provider = os.getenv("AIL_LLM_PROVIDER", "openrouter")
    llm_model = os.getenv("AIL_LLM_MODEL", "")
    llm_api_key = os.getenv("AIL_LLM_API_KEY", "")
    llm_base_url = os.getenv(
        "AIL_LLM_BASE_URL",
        "https://openrouter.ai/api/v1"
    )
    oi_executable = os.getenv("AIL_OI_EXECUTABLE", "interpreter")
    stt_model = os.getenv("AIL_STT_MODEL", "tiny")
    stt_device = os.getenv("AIL_STT_DEVICE", "cpu")
    stt_compute_type = os.getenv("AIL_STT_COMPUTE_TYPE", "int8")
    voice_sample_rate = int(os.getenv("AIL_VOICE_SAMPLE_RATE", "16000"))
    voice_record_seconds = float(os.getenv("AIL_VOICE_RECORD_SECONDS", "5"))
    tts_voice = os.getenv("AIL_TTS_VOICE", "en-US-AriaNeural")
    tts_rate = os.getenv("AIL_TTS_RATE", "+0%")
    tts_volume = os.getenv("AIL_TTS_VOLUME", "+0%")