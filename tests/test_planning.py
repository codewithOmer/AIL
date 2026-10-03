"""Tests for AIL planning foundation (Milestone 2F).

Covers deterministic plan creation and integrity validation.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from core.actions import execute_and_verify  # noqa: E402
from core.planner import (  # noqa: E402
    DeterministicMultiStepPlanner,
    DeterministicPlanner,
    GoalAwarePlanner,
    SupportedFileTaskPlanner,
    validate_plan,
)
from core.verification import FilesystemVerifier  # noqa: E402
from interfaces.planning import Goal, Plan, PlanStep  # noqa: E402
from interfaces.verification import FileExpectation  # noqa: E402


class TestPlanning(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = DeterministicPlanner()

    def test_deterministic_planner_creates_valid_plan(self) -> None:
        goal = Goal(description="Create a file")
        plan = self.planner.plan(goal)
        self.assertIsInstance(plan, Plan)
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].action, "Create a file")
        self.assertEqual(plan.steps[0].id, "step-1")

    def test_immutability(self) -> None:
        goal = Goal(description="x")
        with self.assertRaises(Exception):
            goal.description = "y"  # type: ignore

    def test_empty_plan_rejected(self) -> None:
        plan = Plan(goal=Goal("x"), steps=())
        with self.assertRaisesRegex(ValueError, "plan must contain at least one step"):
            validate_plan(plan)

    def test_duplicate_ids_rejected(self) -> None:
        steps = (
            PlanStep(id="1", action="a"),
            PlanStep(id="1", action="b"),
        )
        plan = Plan(goal=Goal("x"), steps=steps)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_plan(plan)

    def test_unknown_dependency_rejected(self) -> None:
        steps = (
            PlanStep(id="2", action="a", depends_on=("1",)),
        )
        plan = Plan(goal=Goal("x"), steps=steps)
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_plan(plan)

    def test_self_dependency_rejected(self) -> None:
        plan = Plan(
            goal=Goal("x"),
            steps=(PlanStep(id="a", action="A", depends_on=("a",)),),
        )
        with self.assertRaisesRegex(ValueError, "cannot depend on itself"):
            validate_plan(plan)

    def test_duplicate_dependency_rejected(self) -> None:
        plan = Plan(
            goal=Goal("x"),
            steps=(
                PlanStep(id="a", action="A"),
                PlanStep(id="b", action="B", depends_on=("a", "a")),
            ),
        )
        with self.assertRaisesRegex(ValueError, "more than once"):
            validate_plan(plan)

    def test_valid_dependency_chain_is_accepted(self) -> None:
        plan = Plan(
            goal=Goal("x"),
            steps=(
                PlanStep(id="a", action="A"),
                PlanStep(id="b", action="B", depends_on=("a",)),
                PlanStep(id="c", action="C", depends_on=("b",)),
            ),
        )
        validate_plan(plan)

    def test_valid_independent_steps_are_accepted(self) -> None:
        plan = Plan(
            goal=Goal("x"),
            steps=(
                PlanStep(id="a", action="A"),
                PlanStep(id="b", action="B"),
                PlanStep(id="c", action="C"),
            ),
        )
        validate_plan(plan)

    def test_forward_dependency_is_rejected(self) -> None:
        plan = Plan(
            goal=Goal("x"),
            steps=(
                PlanStep(id="a", action="A", depends_on=("b",)),
                PlanStep(id="b", action="B"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "later"):
            validate_plan(plan)

    def test_two_step_dependency_cycle_is_rejected(self) -> None:
        plan = Plan(
            goal=Goal("x"),
            steps=(
                PlanStep(id="a", action="A", depends_on=("b",)),
                PlanStep(id="b", action="B", depends_on=("a",)),
            ),
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_plan(plan)

    def test_longer_dependency_cycle_is_rejected(self) -> None:
        plan = Plan(
            goal=Goal("x"),
            steps=(
                PlanStep(id="a", action="A", depends_on=("b",)),
                PlanStep(id="b", action="B", depends_on=("c",)),
                PlanStep(id="c", action="C", depends_on=("a",)),
            ),
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_plan(plan)

    def test_invalid_max_attempts_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PlanStep(id="1", action="a", max_attempts=0)

    def test_valid_max_attempts_accepted(self) -> None:
        step = PlanStep(id="1", action="a", max_attempts=5)
        self.assertEqual(step.max_attempts, 5)

    def test_arbitrary_expectations_opaque(self) -> None:
        class Opaque:
            pass
        exp = Opaque()
        step = PlanStep(id="1", action="a", expectations=(exp,))
        self.assertIs(step.expectations[0], exp)

    def test_planner_requires_no_context(self) -> None:
        # Should not raise
        self.planner.plan(Goal("x"))

    def test_supported_file_task_has_explicit_executor_action(self) -> None:
        plan = SupportedFileTaskPlanner().plan(
            Goal("Create a file named notes.txt containing project notes")
        )

        self.assertEqual(
            plan.steps[0].action,
            "Create notes.txt in the current workspace containing exactly "
            "the text project notes.",
        )


class TestDeterministicMultiStepPlanner(unittest.TestCase):
    def test_creates_stable_dependency_safe_plan(self) -> None:
        planner = DeterministicMultiStepPlanner()
        goal = Goal("Complete the task")

        plan = planner.plan(goal)

        self.assertEqual(
            plan.steps,
            (
                PlanStep(
                    id="prepare-workspace",
                    action="Prepare the workspace for the requested task.",
                    expectations=("workspace-prepared",),
                ),
                PlanStep(
                    id="complete-task",
                    action="Complete the task",
                    expectations=("task-completed",),
                    depends_on=("prepare-workspace",),
                ),
            ),
        )

    def test_is_deterministic_and_validates(self) -> None:
        planner = DeterministicMultiStepPlanner()
        goal = Goal("Complete the task")

        first = planner.plan(goal)
        second = planner.plan(goal)

        self.assertEqual(first, second)
        self.assertEqual(
            [step.id for step in first.steps],
            ["prepare-workspace", "complete-task"],
        )
        validate_plan(first)


class TestGoalAwarePlanner(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = GoalAwarePlanner()

    def test_workspace_goal_selects_predefined_plan(self) -> None:
        goal = Goal("Create a project workspace")

        plan = self.planner.plan(goal)

        self.assertIs(plan.goal, goal)
        self.assertEqual(
            plan.steps,
            (
                PlanStep(
                    id="create-workspace",
                    action="Create the project workspace.",
                    expectations=("project-workspace-created",),
                ),
                PlanStep(
                    id="verify-workspace",
                    action="Verify the project workspace.",
                    expectations=("project-workspace-verified",),
                    depends_on=("create-workspace",),
                ),
            ),
        )
        validate_plan(plan)

    def test_summary_goal_selects_different_predefined_plan(self) -> None:
        goal = Goal("Create a project summary")

        plan = self.planner.plan(goal)

        self.assertIs(plan.goal, goal)
        self.assertEqual(
            plan.steps,
            (
                PlanStep(
                    id="collect-summary",
                    action="Collect project summary details.",
                    expectations=("summary-details-collected",),
                ),
                PlanStep(
                    id="write-summary",
                    action="Write the project summary.",
                    expectations=("project-summary-written",),
                    depends_on=("collect-summary",),
                ),
            ),
        )
        validate_plan(plan)

    def test_recognized_plans_are_deterministic(self) -> None:
        workspace = Goal("Create a project workspace")
        summary = Goal("Create a project summary")

        self.assertEqual(self.planner.plan(workspace), self.planner.plan(workspace))
        self.assertNotEqual(self.planner.plan(workspace), self.planner.plan(summary))

    def test_unsupported_goal_falls_back_to_single_step_plan(self) -> None:
        goal = Goal("Do something else")

        plan = self.planner.plan(goal)

        self.assertIs(plan.goal, goal)
        self.assertEqual(plan.steps, (PlanStep(id="step-1", action=goal.description),))
        validate_plan(plan)


class TestProductionPlannerExpectationContract(unittest.TestCase):
    """The production planner's expectations must be consumable by the real verifier.

    ``SupportedFileTaskPlanner`` is the planner ``AILApplication.create`` wires by
    default.  Its ``PlanStep.expectations`` are opaque at the planning-contract
    level, so nothing in ``interfaces.planning`` guarantees the concrete verifier
    can read them.  These tests drive the real ``FilesystemVerifier`` over a real
    filesystem so that a future planner returning an expectation shape the
    verifier cannot consume fails here instead of at runtime.

    This deliberately does not cover ``DeterministicMultiStepPlanner`` or
    ``GoalAwarePlanner``: they are shape-only test fixtures whose string
    expectations the filesystem verifier is not built to consume.
    """

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.verifier = FilesystemVerifier()
        self.planner = SupportedFileTaskPlanner()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def expectations_for(self, filename: str, content: str) -> tuple:
        plan = self.planner.plan(
            Goal(f"Create a file named {filename} containing {content}")
        )
        self.assertEqual(len(plan.steps), 1)
        return plan.steps[0].expectations

    def test_production_expectations_are_consumed_by_the_real_verifier(self) -> None:
        expectations = self.expectations_for("contract.txt", "expected text")
        self.assertEqual(len(expectations), 1)
        self.assertIsInstance(expectations[0], FileExpectation)

        (self.temp_dir / "contract.txt").write_text("expected text")

        result = self.verifier.verify(expectations, self.temp_dir)

        self.assertTrue(result.passed, result.reason)
        self.assertEqual(len(result.checks), 1)
        self.assertTrue(result.checks[0].passed)

    def test_production_expectations_fail_when_state_is_absent(self) -> None:
        expectations = self.expectations_for("absent.txt", "expected text")

        result = self.verifier.verify(expectations, self.temp_dir)

        self.assertFalse(result.passed)
        self.assertEqual(len(result.checks), 1)
        self.assertFalse(result.checks[0].passed)

    def test_production_expectations_detect_wrong_content(self) -> None:
        expectations = self.expectations_for("wrong.txt", "expected text")
        (self.temp_dir / "wrong.txt").write_text("something else")

        result = self.verifier.verify(expectations, self.temp_dir)

        self.assertFalse(result.passed)
        self.assertFalse(result.checks[0].passed)

    def test_production_expectations_detect_missing_trailing_content(self) -> None:
        expectations = self.expectations_for("partial.txt", "expected text")
        (self.temp_dir / "partial.txt").write_text("expected")

        result = self.verifier.verify(expectations, self.temp_dir)

        self.assertFalse(result.passed)
        self.assertFalse(result.checks[0].passed)

    def test_full_planned_expectations_round_trip_through_execute_and_verify(
        self,
    ) -> None:
        """The whole production seam: plan -> executor -> verify, same objects."""

        class WritingClient:
            def send_message(self, message, thread_id=None, timeout=None):
                (self.temp_dir / "seam.txt").write_text("seam content")
                return "executor text is never trusted"

        client = WritingClient()
        expectations = self.expectations_for("seam.txt", "seam content")

        _, result = execute_and_verify(
            client,
            self.verifier,
            "ignored action",
            expectations,
            base_dir=str(self.temp_dir),
        )

        self.assertTrue(result.passed, result.reason)


if __name__ == "__main__":
    unittest.main()
