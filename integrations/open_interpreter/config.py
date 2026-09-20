import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class OIConfig:
    """Configuration for the Open Interpreter App Server adapter.

    All values come from environment variables so nothing is hardcoded.
    """

    executable: str = field(
        default_factory=lambda: os.getenv(
            "AIL_OI_EXECUTABLE",
            "interpreter",
        )
    )
    listen: str = field(
        default_factory=lambda: os.getenv("AIL_OI_LISTEN", "stdio://")
    )
    model: str = field(
        default_factory=lambda: os.getenv("AIL_OI_MODEL", "")
    )
    model_provider: str = field(
        default_factory=lambda: os.getenv("AIL_OI_MODEL_PROVIDER", "")
    )
    approval_policy: str = field(
        default_factory=lambda: os.getenv("AIL_OI_APPROVAL_POLICY", "never")
    )
    sandbox: str = field(
        default_factory=lambda: os.getenv("AIL_OI_SANDBOX", "read-only")
    )
    cwd: str = field(
        default_factory=lambda: os.getenv("AIL_OI_CWD", ".")
    )
    request_timeout: float = field(
        default_factory=lambda: float(os.getenv("AIL_OI_REQUEST_TIMEOUT", "120"))
    )
    connect_timeout: float = field(
        default_factory=lambda: float(os.getenv("AIL_OI_CONNECT_TIMEOUT", "30"))
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("AIL_OI_LOG_LEVEL", "INFO")
    )
    extra_args: list[str] = field(
        default_factory=lambda: [
            a.strip()
            for a in os.getenv("AIL_OI_EXTRA_ARGS", "").split(",")
            if a.strip()
        ]
    )
