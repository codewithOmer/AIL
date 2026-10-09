"""AIL Plan Runner (Milestone 2G).

Composes the existing PLAN → EXECUTE → VERIFY → RECOVER pieces without
replacing any of them: planning stays in the planner, verification in the
verifier, retry logic in ``execute_with_recovery``.  The runner only orders
steps, threads execution context, marks outcomes and aggregates the report.

Open Interpreter is only the executor and is never imported here; the client
arrives through the constructor seam.  The executor's response text is never
inspected — ``RecoveryResult.passed`` (a filesystem verdict) is the sole
authority for step success.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.planner import validate_plan
from core.recovery import execute_with_recovery
from interfaces.image import LocalImage
from interfaces.planning import (
    ExecutionReport,
    Goal,
    Plan,
    PlanAttempt,
    Planner,
    Replanner,
    StepResult,
    StepStatus,
)
from interfaces.recovery import RecoveryStrategy


class PlanRunner:
    """Execute a validated :class:`Plan` step by step, in tuple order.

    A step runs only when all of its declared dependencies PASSED.  A step
    whose dependency FAILED or was SKIPPED is itself SKIPPED without
    executing.  The first permanent step failure is fail-fast: every
    remaining step is marked SKIPPED and the report is returned immediately.
    """

    def __init__(
        self,
        planner: Planner,
        client: Any,
        verifier: Any,
        strategy: RecoveryStrategy | None = None,
        base_dir: str = ".",
        default_max_attempts: int = 2,
        thread_id: str | None = None,
        replanner: Replanner | None = None,
        timeout: float | None = None,
        tools: Mapping[str, Any] | None = None,
    ) -> None:
        self.planner = planner
        self.client = client
        self.verifier = verifier
        self.strategy = strategy
        self.base_dir = base_dir
        self.default_max_attempts = default_max_attempts
        self.thread_id = thread_id
        self.replanner = replanner
        self.timeout = timeout
        self.tools = dict(tools) if tools is not None else {}

    def run(
        self,
        goal: Goal,
        *,
        images: Sequence[LocalImage] | None = None,
    ) -> ExecutionReport:
        """Plan *goal*, execute its steps, and return the final report."""
        plan = self.planner.plan(goal)
        return self._run_with_replanning(plan, images=images)

    def run_plan(
        self,
        plan: Plan,
        *,
        images: Sequence[LocalImage] | None = None,
    ) -> ExecutionReport:
        """Execute an existing :class:`Plan` and return the final report."""
        return self._run_with_replanning(plan, images=images)

    def _run_with_replanning(
        self,
        plan: Plan,
        *,
        images: Sequence[LocalImage] | None = None,
    ) -> ExecutionReport:
        initial = self._execute_plan(plan, images=images)
        if initial.passed or self.replanner is None:
            return initial

        replacement = self.replanner.replan(plan, initial)
        if replacement is None:
            return initial

        try:
            validate_plan(replacement)
            self._validate_step_tools(replacement)
        except ValueError:
            return initial

        final = self._execute_plan(replacement, images=images)
        return ExecutionReport(
            goal=final.goal,
            passed=final.passed,
            steps=final.steps,
            attempts=initial.attempts + final.attempts,
        )

    def _validate_step_tools(self, plan: Plan) -> None:
        """Reject steps whose declared tool is not registered with the runner.

        Runs before any step executes, so an unknown tool can never reach an
        executor or allow an earlier step to run before the plan is rejected.
        ``None`` needs no registration because it selects the runner's default
        ``client``.
        """
        unknown = sorted(
            {
                step.tool
                for step in plan.steps
                if step.tool is not None and step.tool not in self.tools
            }
        )
        if unknown:
            raise ValueError(
                "unknown tool(s) not registered with the runner: "
                + ", ".join(repr(name) for name in unknown)
            )

    def _execute_plan(
        self,
        plan: Plan,
        *,
        images: Sequence[LocalImage] | None = None,
    ) -> ExecutionReport:
        validate_plan(plan)
        self._validate_step_tools(plan)

        results: list[StepResult] = []
        passed_ids: set[str] = set()
        failed = False

        for step in plan.steps:
            if failed:
                results.append(
                    StepResult(
                        step_id=step.id,
                        action=step.action,
                        status=StepStatus.SKIPPED,
                    )
                )
                continue

            blocked = any(
                dep not in passed_ids
                and any(r.step_id == dep for r in results)
                for dep in step.depends_on
            )
            if blocked:
                results.append(
                    StepResult(
                        step_id=step.id,
                        action=step.action,
                        status=StepStatus.SKIPPED,
                    )
                )
                continue

            executor = self.client if step.tool is None else self.tools[step.tool]
            try:
                recovery = execute_with_recovery(
                    executor,
                    self.verifier,
                    step.action,
                    step.expectations,
                    thread_id=self.thread_id,
                    base_dir=self.base_dir,
                    strategy=self.strategy,
                    max_attempts=(
                        step.max_attempts
                        if step.max_attempts is not None
                        else self.default_max_attempts
                    ),
                    timeout=self.timeout,
                    images=images,
                )
                status = StepStatus.PASSED if recovery.passed else StepStatus.FAILED
                results.append(
                    StepResult(
                        step_id=step.id,
                        action=step.action,
                        status=status,
                        recovery=recovery,
                    )
                )
                if recovery.passed:
                    passed_ids.add(step.id)
                else:
                    failed = True
            except Exception as exc:  # executor/verification crash stays contained
                failed = True
                results.append(
                    StepResult(
                        step_id=step.id,
                        action=step.action,
                        status=StepStatus.FAILED,
                        error=str(exc),
                    )
                )

        report = ExecutionReport(
            goal=plan.goal,
            passed=all(r.status is StepStatus.PASSED for r in results),
            steps=tuple(results),
        )
        attempt = PlanAttempt(
            plan=plan,
            passed=report.passed,
            steps=report.steps,
        )
        return ExecutionReport(
            goal=report.goal,
            passed=report.passed,
            steps=report.steps,
            attempts=(attempt,),
        )
