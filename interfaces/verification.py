"""AIL-owned verification contract.

``Verifier`` implementations independently confirm that an executed action
produced the expected state.  They must inspect the real filesystem directly
and must never delegate to — or trust — the executor that performed the action.

``FileExpectation`` lives here, not in a concrete verifier, because it is the
shared vocabulary between the layers that *declare* expected state (planners)
and the layers that *check* it (verifiers).  It is pure, behaviour-free data so
that a planner can emit it without depending on any verifier implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FileExpectation:
    """Expected filesystem state, relative to the verification base directory."""

    path: str | Path
    exists: bool = True
    contains: str | None = None


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