"""AIL-owned planning contract.

A ``Plan`` is a static, ordered description of intended work that sits BEFORE
execution.  Planning never executes anything and never verifies anything.

The plan *structure* is executor-agnostic: this module defines no expectation
type and ``PlanStep.expectations`` stays opaque ``Any`` data, so the planning
layer imports no executor, verifier, recovery or action implementation.
Concrete planners are free to emit expectation types owned by the interface
layer — for example ``interfaces.verification.FileExpectation`` — and it is the
concrete verifier's responsibility to consume them.
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
        if not isinstance(self.action, str) or not self.action.strip():
            raise ValueError("step action must be non-empty")
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
class PlanAttempt:
    """Outcome of executing one complete plan."""

    plan: Plan
    passed: bool
    steps: tuple[StepResult, ...]


@dataclass(frozen=True)
class ExecutionReport:
    """Final outcome of running a :class:`Plan`.

    ``passed`` is True only when every step PASSED.
    """

    goal: Goal
    passed: bool
    steps: tuple[StepResult, ...]
    attempts: tuple[PlanAttempt, ...] = ()


class Replanner(ABC):
    """Contract for producing one replacement plan after failure."""

    @abstractmethod
    def replan(self, plan: Plan, report: ExecutionReport) -> Plan | None:
        """Return one replacement plan, or None to stop."""
        raise NotImplementedError