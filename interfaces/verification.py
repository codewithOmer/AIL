"""AIL-owned verification contract.

``Verifier`` implementations independently confirm that an executed action
produced the expected state.  They must inspect the real filesystem directly
and must never delegate to — or trust — the executor that performed the action.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    """Outcome of verifying a single expectation."""

    name: str
    passed: bool
    expected: str
    actual: str
    error: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    """Aggregate outcome of verifying a set of expectations."""

    passed: bool
    reason: str
    checks: tuple[CheckResult, ...] = ()


class Verifier(ABC):
    """Contract every AIL verifier must implement."""

    @abstractmethod
    def verify(self, expectations: Any, base_dir: Any) -> VerificationResult:
        """Verify *expectations* against the real filesystem under *base_dir*."""
        raise NotImplementedError