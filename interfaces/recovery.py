"""AIL-owned recovery contract.

A ``RecoveryStrategy`` decides how to continue after a ``VerificationResult``
fails.  Strategies only ever produce the next *message* for the executor; they
never run or invent commands and never mutate state themselves.  This contract
is executor-agnostic — no Open Interpreter types are involved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from interfaces.verification import VerificationResult


@dataclass(frozen=True)
class RecoveryResult:
    """Outcome of a bounded execute → verify → recover loop."""

    passed: bool
    attempts: int
    reason: str
    verification: VerificationResult
    history: tuple[VerificationResult, ...] = ()


class RecoveryStrategy(ABC):
    """Contract for deciding the next executor message after a failure."""

    @abstractmethod
    def next_message(
        self, original: str, result: VerificationResult, attempt: int
    ) -> str | None:
        """Return the next message for the executor, or None to give up."""
        raise NotImplementedError