"""AIL-owned recovery orchestration on top of ``execute_and_verify``.

Bounded, conservative loop: execute → verify → (fail → recover → retry).
Open Interpreter stays the executor; ``VerificationResult`` is the sole
authority for success; strategies only shape the next message and never run
commands or perform destructive actions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from core.actions import execute_and_verify
from interfaces.image import LocalImage
from interfaces.recovery import RecoveryResult, RecoveryStrategy
from interfaces.verification import VerificationResult


def _attempt_noun(count: int) -> str:
    return "attempt" if count == 1 else "attempts"


def _display_name(name: str) -> str:
    """Return a safe display label for a failed check.

    ``CheckResult.name`` is the absolute path the verifier examined.  Handing
    that to the executor leaks the internal filesystem layout, so only the
    final path component (the file/directory the check is actually about) is
    shown, stripped of any parent directories above it.
    """
    return name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


class DefaultRetry(RecoveryStrategy):
    """Conservative default strategy.

    Re-issues the original message and tells the executor which checks failed,
    asking it to correct them.  Never invents shell commands and never
    introduces destructive actions.
    """

    def next_message(
        self, original: str, result: VerificationResult, attempt: int
    ) -> str | None:
        if result.passed:
            return None
        failed = [c for c in result.checks if not c.passed]
        details = "; ".join(
            f"{_display_name(c.name)} (expected {c.expected}, got {c.actual})"
            for c in failed
        )
        return (
            f"{original}\n\n"
            "The previous attempt did not satisfy all requirements. "
            "Please correct the failed checks listed below. Do not take "
            "destructive actions and do not invent new commands.\n"
            f"{details}"
        )


def execute_with_recovery(
    client: Any,
    verifier: Any,
    message: str,
    expectations: Any,
    thread_id: str | None = None,
    base_dir: str = ".",
    strategy: RecoveryStrategy | None = None,
    max_attempts: int = 2,
    timeout: float | None = None,
    images: Sequence[LocalImage] | None = None,
) -> RecoveryResult:
    """Execute and verify up to *max_attempts* times, recovering on failure.

    Every attempt runs ``execute_and_verify`` and verifies the filesystem.  The
    loop stops on the first PASS, when *max_attempts* is reached, or when the
    *strategy* returns None.  The Open Interpreter response text is never
    inspected; only the ``VerificationResult`` decides success.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    recovery = strategy or DefaultRetry()
    history: list[VerificationResult] = []
    current_message = message
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        _, result = execute_and_verify(
            client,
            verifier,
            current_message,
            expectations,
            thread_id=thread_id,
            base_dir=base_dir,
            timeout=timeout,
            images=images,
        )
        history.append(result)

        if result.passed:
            return RecoveryResult(
                passed=True,
                attempts=attempt,
                reason=f"PASS on attempt {attempt}",
                verification=result,
                history=tuple(history),
            )

        if attempt >= max_attempts:
            break

        next_message = recovery.next_message(message, result, attempt)
        if not next_message:
            break
        current_message = next_message

    final = history[-1]
    return RecoveryResult(
        passed=False,
        attempts=attempt,
        reason=f"FINAL FAIL after {attempt} {_attempt_noun(attempt)}",
        verification=final,
        history=tuple(history),
    )