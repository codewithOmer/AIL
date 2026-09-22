"""Deterministic, execution-free planning foundation (Milestone 2F).

``DeterministicPlanner`` produces a reflexive one-step plan per goal and never
executes, verifies or communicates with any executor.  ``validate_plan`` is an
AIL-owned integrity check over the static plan form.
"""

from __future__ import annotations

from interfaces.planning import Goal, Plan, PlanStep, Planner


class DeterministicPlanner(Planner):
    """Reflexive planner: one executor step per goal.

    For Milestone 2F the plan is a single step whose action is the goal's
    description.  Verification/recovery/report remain runner responsibilities
    (see 2D/2E) and are deliberately NOT plan steps.
    """

    def plan(self, goal: Goal) -> Plan:
        return Plan(
            goal=goal,
            steps=(PlanStep(id="step-1", action=goal.description),),
        )


class DeterministicMultiStepPlanner(Planner):
    """Fixed two-step planner for proving structured plan execution."""

    def plan(self, goal: Goal) -> Plan:
        return Plan(
            goal=goal,
            steps=(
                PlanStep(
                    id="prepare-workspace",
                    action="Prepare the workspace for the requested task.",
                    expectations=("workspace-prepared",),
                ),
                PlanStep(
                    id="complete-task",
                    action=goal.description,
                    expectations=("task-completed",),
                    depends_on=("prepare-workspace",),
                ),
            ),
        )


def validate_plan(plan: Plan) -> None:
    """Reject structurally invalid plans.

    Dependencies are metadata only in 2F — execution order remains tuple
    order, so this never schedules anything.  Raises ``ValueError`` on the
    first problem found.
    """
    if not plan.steps:
        raise ValueError("plan must contain at least one step")

    step_ids: set[str] = set()
    for step in plan.steps:
        sid = step.id
        if not sid or not sid.strip():
            raise ValueError("every step must have a non-empty id")
        if sid in step_ids:
            raise ValueError(f"duplicate step id: {sid!r}")
        step_ids.add(sid)
        if step.max_attempts is not None and step.max_attempts < 1:
            raise ValueError(
                f"step {sid!r} max_attempts must be None or >= 1, "
                f"got {step.max_attempts}"
            )
        if len(set(step.depends_on)) != len(step.depends_on):
            raise ValueError(f"step {sid!r} lists a dependency more than once")

    for step in plan.steps:
        for dep in step.depends_on:
            if dep == step.id:
                raise ValueError(f"step {step.id!r} cannot depend on itself")
            if dep not in step_ids:
                raise ValueError(
                    f"step {step.id!r} depends on unknown step {dep!r}"
                )