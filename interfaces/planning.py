"""AIL-owned planning contract.

A ``Plan`` is a static, ordered description of intended work that sits BEFORE
execution.  Planning never executes anything, never verifies anything and is
completely executor-agnostic: the planner treats ``PlanStep.expectations`` as
opaque data and imports no executor, verifier, recovery or action modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from interfaces.recovery import RecoveryResult


@dataclass(frozen=True)
class Goal:
    """The intent a plan is built to satisfy."""

    description: str


@dataclass(frozen=True)
class PlanStep:
    """One unit of planned work, carried verbatim to the executor later."""

    id: str
    action: str
    expectations: tuple[Any, ...] = ()
    depends_on: tuple[str, ...] = ()
    max_attempts: int | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("step id must be non-empty")
        if self.max_attempts is not None and self.max_attempts < 1:
            raise ValueError("max_attempts must be None or >= 1")


@dataclass(frozen=True)
class Plan:
    """A static, ordered plan.  Execution order is tuple order."""

    goal: Goal
    steps: tuple[PlanStep, ...]


class Planner(ABC):
    """Contract for producing a :class:`Plan` from a :class:`Goal`."""

    @abstractmethod
    def plan(self, goal: Goal) -> Plan:
        """Return a deterministic plan for *goal* without executing anything."""
        raise NotImplementedError


class StepStatus(Enum):
    """Outcome of one planned step under the runner."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class StepResult:
    """Runner outcome for a single :class:`PlanStep`.

    ``recovery`` preserves the bounded loop's ``RecoveryResult`` verbatim; it
    is ``None`` for steps that never executed (skipped or failed early).
    """

    step_id: str
    action: str
    status: StepStatus
    recovery: "RecoveryResult | None" = None
    error: str | None = None


@dataclass(frozen=True)
class ExecutionReport:
    """Final outcome of running a :class:`Plan`.

    ``passed`` is True only when every step PASSED.
    """

    goal: Goal
    passed: bool
    steps: tuple[StepResult, ...]