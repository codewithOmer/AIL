"""Tests for AIL planning foundation (Milestone 2F).

Covers deterministic plan creation and integrity validation.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from core.actions import execute_and_verify  # noqa: E402
from core.plan_runner import PlanRunner  # noqa: E402
from core.planner import (  # noqa: E402
    DeterministicMultiStepPlanner,
    DeterministicPlanner,
    FirstMatchPlanner,
    GoalAwarePlanner,
    SupportedFileTaskPlanner,
    UnrecognizedTaskError,
    UnsupportedTaskError,
    WorkspaceSetupPlanner,
    validate_plan,
)
from core.verification import FilesystemVerifier  # noqa: E402
from interfaces.planning import Goal, Plan, PlanStep, StepStatus  # noqa: E402
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
            """Writes the expected file into the test's temp workspace."""

            def __init__(self, temp_dir: Path) -> None:
                self.temp_dir = temp_dir

            def send_message(self, message, thread_id=None, timeout=None):
                (self.temp_dir / "seam.txt").write_text("seam content")
                return "executor text is never trusted"

        client = WritingClient(self.temp_dir)
        expectations = self.expectations_for("seam.txt", "seam content")

        _, result = execute_and_verify(
            client,
            self.verifier,
            "ignored action",
            expectations,
            base_dir=str(self.temp_dir),
        )

        self.assertTrue(result.passed, result.reason)


_GOAL = (
    "set up a project workspace src with a readme containing hello world "
    'and a config file containing {"a": 1}'
)


class _WorkspaceWritingClient:
    """Fake executor that materialises exactly what the planner's actions ask for.

    Outcomes are decided by the independent ``FilesystemVerifier`` reading the
    real filesystem; this client's response text is never consulted.
    """

    _DIR_ACTION = re.compile(
        r"^Create the directory (?P<dir>\S+) in the current workspace\.$"
    )
    _FILE_ACTION = re.compile(
        r"^Create (?P<path>\S+) inside \S+ containing exactly the text "
        r"(?P<content>.*)\.$"
    )

    def __init__(self, base_dir: Path, *, skip: str | None = None) -> None:
        self.base_dir = base_dir
        self.skip = skip
        self.calls: list[str] = []

    def send_message(self, message, thread_id=None, timeout=None, **kwargs):
        self.calls.append(message)
        if self.skip is not None and self.skip in message:
            return "pretended to do the work"
        if match := self._DIR_ACTION.match(message):
            (self.base_dir / match.group("dir")).mkdir(
                parents=True, exist_ok=True
            )
        elif match := self._FILE_ACTION.match(message):
            target = self.base_dir / match.group("path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(match.group("content"))
        return "executor text is never trusted"


class TestWorkspaceSetupPlanner(unittest.TestCase):
    """Production multi-step filesystem planning.

    The planner emits a directory step plus two dependent file steps.  Every
    step carries at least one real ``FileExpectation`` so the runner's
    independent verification always has non-vacuous evidence to check.
    """

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.verifier = FilesystemVerifier()
        self.planner = WorkspaceSetupPlanner()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def plan_for(self, description: str = _GOAL):
        return self.planner.plan(Goal(description))

    # --- plan shape -----------------------------------------------------

    def test_generated_plan_contains_multiple_steps(self) -> None:
        plan = self.plan_for()
        self.assertEqual(len(plan.steps), 3)

    def test_step_ids_are_unique(self) -> None:
        ids = [step.id for step in self.plan_for().steps]
        self.assertEqual(len(ids), len(set(ids)))

    def test_dependencies_reference_only_earlier_steps(self) -> None:
        plan = self.plan_for()
        seen: list[str] = []
        for step in plan.steps:
            for dependency in step.depends_on:
                self.assertIn(dependency, seen)
            seen.append(step.id)

    def test_validate_plan_accepts_generated_plan(self) -> None:
        validate_plan(self.plan_for())

    def test_every_step_has_at_least_one_expectation(self) -> None:
        for step in self.plan_for().steps:
            self.assertGreaterEqual(
                len(step.expectations),
                1,
                msg=f"step {step.id} would verify vacuously",
            )

    def test_plan_is_deterministic(self) -> None:
        self.assertEqual(self.plan_for(), self.plan_for())

    def test_current_goal_wins_over_matching_recalled_context(self) -> None:
        context = (
            "Relevant memories:\n"
            "- The user asked to set up a project workspace old with a readme "
            "containing stale and a config file containing stale\n\n"
        )
        plan = self.planner.plan(Goal(context + _GOAL))
        paths = [
            expectation.path for step in plan.steps for expectation in step.expectations
        ]
        self.assertIn("src", paths)
        self.assertNotIn("old", paths)

    # --- path safety ----------------------------------------------------

    def test_safe_nested_paths_are_accepted(self) -> None:
        plan = self.planner.plan(
            Goal(
                "set up a project workspace src/config "
                "with a readme containing hi and a config file containing bye"
            )
        )
        validate_plan(plan)
        paths = [
            expectation.path for step in plan.steps for expectation in step.expectations
        ]
        self.assertIn("src/config", paths)
        self.assertIn("src/config/README.md", paths)
        self.assertIn("src/config/config.json", paths)

    def test_workspace_escaping_paths_are_rejected(self) -> None:
        for directory in (
            "../outside",
            "..\\outside",
            "C:\\outside.txt",
            "C:/outside.txt",
            "/etc/passwd",
            "\\\\server\\share",
            "src/../../etc",
        ):
            with self.subTest(directory=directory):
                with self.assertRaises(UnsupportedTaskError):
                    self.planner.plan(
                        Goal(
                            f"set up a project workspace {directory} with a "
                            "readme containing hi and a config file containing bye"
                        )
                    )

    def test_unsupported_goal_raises(self) -> None:
        for description in (
            "create a file named a.txt containing x",
            "send an email to someone",
            "",
        ):
            with self.subTest(description=description):
                with self.assertRaises(UnsupportedTaskError):
                    self.planner.plan(Goal(description))

    # --- verifier contract ---------------------------------------------

    def test_every_expectation_is_consumable_by_the_real_verifier(self) -> None:
        plan = self.plan_for()
        client = _WorkspaceWritingClient(self.temp_dir)

        for step in plan.steps:
            for expectation in step.expectations:
                self.assertIsInstance(expectation, FileExpectation)
                # Nothing exists yet, so every expectation must genuinely fail
                # rather than raise or silently pass.
                before = self.verifier.verify(step.expectations, self.temp_dir)
                self.assertFalse(before.passed)
                self.assertEqual(len(before.checks), 1)
            client.send_message(step.action)
            after = self.verifier.verify(step.expectations, self.temp_dir)
            self.assertTrue(after.passed, after.reason)

    # --- end-to-end through PlanRunner ----------------------------------

    def _run(self, *, skip: str | None = None):
        runner = PlanRunner(
            planner=self.planner,
            client=_WorkspaceWritingClient(self.temp_dir, skip=skip),
            verifier=self.verifier,
            base_dir=str(self.temp_dir),
            default_max_attempts=2,
        )
        return runner, runner.run(Goal(_GOAL))

    @staticmethod
    def _diagnose(report) -> str:
        """``ExecutionReport`` carries no ``reason``; build one from steps."""
        return "\n".join(
            f"{step.step_id}: {step.status.value} error={step.error!r}"
            for step in report.steps
        )

    def test_generated_plan_executes_successfully(self) -> None:
        runner, report = self._run()
        self.assertTrue(report.passed, self._diagnose(report))
        self.assertEqual(
            [step.status for step in report.steps],
            [StepStatus.PASSED, StepStatus.PASSED, StepStatus.PASSED],
        )
        self.assertEqual(len(runner.client.calls), 3)
        self.assertTrue((self.temp_dir / "src" / "README.md").is_file())
        self.assertTrue((self.temp_dir / "src" / "config.json").is_file())

    def test_failing_directory_step_skips_both_dependent_steps(self) -> None:
        """Both file steps depend on ``create-workspace``, so failing the
        directory step must skip *both* of them rather than only the ones
        declared downstream of the failure."""
        runner, report = self._run(skip="Create the directory src")
        self.assertFalse(report.passed)
        self.assertEqual(
            [(step.step_id, step.status) for step in report.steps],
            [
                ("create-workspace", StepStatus.FAILED),
                ("create-readme", StepStatus.SKIPPED),
                ("create-config", StepStatus.SKIPPED),
            ],
        )
        # Fail-fast: neither dependent action was ever sent to the executor.
        self.assertEqual(len(runner.client.calls), 2)

    def test_failing_readme_does_not_invalidate_independent_config_step(self) -> None:
        """``create-config`` depends only on ``create-workspace``.

        The planner therefore makes no claim that a failed ``create-readme``
        implies a failed ``create-config``; this test pins the dependency
        edges rather than asserting a runner-level fail-fast policy.
        """
        plan = self.plan_for()
        self.assertEqual(
            dict((step.id, step.depends_on) for step in plan.steps),
            {
                "create-workspace": (),
                "create-readme": ("create-workspace",),
                "create-config": ("create-workspace",),
            },
        )


class TestFirstMatchPlanner(unittest.TestCase):
    """Composition must fall through on a recognition miss only.

    A planner that recognizes the syntax but rejects the goal raises the base
    ``UnsupportedTaskError``, which the composite must not swallow -- otherwise
    an unsafe path could be silently retried against a different planner.
    """

    def setUp(self) -> None:
        self.chain = FirstMatchPlanner(
            WorkspaceSetupPlanner(),
            SupportedFileTaskPlanner(),
        )

    def test_unrecognized_is_a_subclass_so_existing_handlers_keep_working(self) -> None:
        self.assertTrue(issubclass(UnrecognizedTaskError, UnsupportedTaskError))

    def test_recognition_miss_raises_the_unrecognized_subclass(self) -> None:
        for planner in (WorkspaceSetupPlanner(), SupportedFileTaskPlanner()):
            with self.subTest(planner=type(planner).__name__):
                with self.assertRaises(UnrecognizedTaskError):
                    planner.plan(Goal("Calculate 23 times 19"))

    def test_rejected_as_unsafe_is_not_a_recognition_miss(self) -> None:
        for directory in ("../escape", "/etc", r"C:\outside", "\\\\server\\share"):
            with self.subTest(directory=directory):
                with self.assertRaises(UnsupportedTaskError) as caught:
                    WorkspaceSetupPlanner().plan(
                        Goal(
                            f"set up a project workspace {directory} "
                            "with a readme containing hi "
                            "and a config file containing bye"
                        )
                    )
                self.assertNotIsInstance(caught.exception, UnrecognizedTaskError)

    def test_workspace_goal_is_handled_without_reaching_the_second_planner(self) -> None:
        plan = self.chain.plan(
            Goal(
                "set up a project workspace src with a readme containing hi "
                "and a config file containing bye"
            )
        )

        self.assertEqual(
            [step.id for step in plan.steps],
            ["create-workspace", "create-readme", "create-config"],
        )

    def test_create_file_goal_falls_through_to_the_second_planner(self) -> None:
        plan = self.chain.plan(Goal("Create a file named hello.txt containing coffee"))

        self.assertEqual([step.id for step in plan.steps], ["create-file"])
        self.assertEqual(plan.steps[0].expectations[0].path, "hello.txt")

    def test_unsafe_workspace_path_does_not_fall_through(self) -> None:
        with self.assertRaises(UnsupportedTaskError) as caught:
            self.chain.plan(
                Goal(
                    "set up a project workspace ../escape "
                    "with a readme containing hi "
                    "and a config file containing bye"
                )
            )

        self.assertNotIsInstance(caught.exception, UnrecognizedTaskError)

    def test_unmatched_goal_raises_the_plain_unsupported_error(self) -> None:
        with self.assertRaises(UnsupportedTaskError) as caught:
            self.chain.plan(Goal("Calculate 23 times 19"))

        self.assertNotIsInstance(caught.exception, UnrecognizedTaskError)

    def test_planner_order_is_observable_and_deterministic(self) -> None:
        self.assertEqual(
            [type(p).__name__ for p in self.chain.planners],
            ["WorkspaceSetupPlanner", "SupportedFileTaskPlanner"],
        )
        goal = Goal(
            "set up a project workspace src with a readme containing hi "
            "and a config file containing bye"
        )
        self.assertEqual(self.chain.plan(goal), self.chain.plan(goal))


if __name__ == "__main__":
    unittest.main()
