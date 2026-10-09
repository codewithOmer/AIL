"""Tests for the AIL Plan Runner (Milestone 2G).

The runner composes the existing PLAN → EXECUTE → VERIFY → RECOVER pieces.
Fake planners and fake executor clients manipulate the real filesystem so the
independent ``FilesystemVerifier`` decides every outcome; the executor's
response text is never trusted.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from core.plan_runner import PlanRunner  # noqa: E402
from core.planner import DeterministicPlanner  # noqa: E402
from interfaces.planning import (  # noqa: E402
    ExecutionReport,
    Goal,
    Plan,
    PlanStep,
    Replanner,
    StepStatus,
)
from interfaces.recovery import RecoveryStrategy  # noqa: E402
from core.verification import FileExpectation, FilesystemVerifier  # noqa: E402

TEXT_A = "content a"
TEXT_B = "content b"


class ScriptedClient:
    """Writes a real file once its key has been seen N times.

    ``script`` maps a substring of the message to
    ``(succeed_on, target_path, text)``.  The Nth call whose message contains
    the key writes *text* to *target_path*; earlier calls do nothing, so the
    independent verifier decides the outcome.
    """

    def __init__(self, script: dict[str, tuple[int, Path, str]]) -> None:
        self.script = script
        self.calls: list[str] = []
        self.counts: dict[str, int] = {}
        self.timeouts: list[float | None] = []

    def send_message(
        self, message: str, thread_id: str | None = None, timeout: float | None = None
    ) -> str:
        self.calls.append(message)
        self.timeouts.append(timeout)
        for key, (succeed_on, target, text) in self.script.items():
            if key in message:
                self.counts[key] = self.counts.get(key, 0) + 1
                if self.counts[key] >= succeed_on:
                    target.write_text(text)
        return "fake executor response text"


class ExplodingClient:
    """Raises immediately, simulating an executor crash."""

    def __init__(self) -> None:
        self.calls = 0

    def send_message(
        self, message: str, thread_id: str | None = None, timeout: float | None = None
    ) -> str:
        self.calls += 1
        raise RuntimeError("executor exploded")


class FixedReplanner(Replanner):
    def __init__(self, replacement: Plan | None) -> None:
        self.replacement = replacement
        self.calls: list[tuple[Plan, ExecutionReport]] = []

    def replan(self, plan: Plan, report: ExecutionReport) -> Plan | None:
        self.calls.append((plan, report))
        return self.replacement


class PlanRunnerTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.temp_dir = Path(self._tmp)
        self.verifier = FilesystemVerifier()

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp)

    def runner(
        self,
        client: Any,
        strategy: RecoveryStrategy | None = None,
        default_max_attempts: int = 2,
        replanner: Replanner | None = None,
        timeout: float | None = None,
        tools: dict[str, Any] | None = None,
    ) -> PlanRunner:
        return PlanRunner(
            planner=DeterministicPlanner(),
            client=client,
            verifier=self.verifier,
            strategy=strategy,
            base_dir=str(self.temp_dir),
            default_max_attempts=default_max_attempts,
            replanner=replanner,
            timeout=timeout,
            tools=tools,
        )

    def run_steps(self, client: Any, steps: tuple[PlanStep, ...]) -> ExecutionReport:
        plan = Plan(goal=Goal("fake goal"), steps=steps)
        return self.runner(client).run_plan(plan)

    def step(
        self,
        sid: str,
        filename: str,
        text: str,
        **kwargs: Any,
    ) -> PlanStep:
        """A step whose action mentions *filename* and expects it to exist."""
        return PlanStep(
            id=sid,
            action=f"write {filename}",
            expectations=(
                FileExpectation(filename, exists=True, contains=text),
            ),
            **kwargs,
        )


class TestPlanRunnerBasics(PlanRunnerTestBase):
    def test_all_steps_pass(self) -> None:
        target = self.temp_dir / "a.txt"
        client = ScriptedClient({"a.txt": (1, target, TEXT_A)})
        report = self.run_steps(client, (self.step("1", "a.txt", TEXT_A),))
        self.assertTrue(report.passed)
        self.assertEqual(report.goal, Goal("fake goal"))
        self.assertEqual(len(report.steps), 1)
        self.assertEqual(report.steps[0].status, StepStatus.PASSED)
        self.assertTrue(target.exists())

    def test_multiple_steps_execute_in_tuple_order(self) -> None:
        a = self.temp_dir / "a.txt"
        b = self.temp_dir / "b.txt"
        client = ScriptedClient(
            {
                "a.txt": (1, a, TEXT_A),
                "b.txt": (1, b, TEXT_B),
            }
        )
        report = self.run_steps(
            client,
            (
                self.step("1", "a.txt", TEXT_A),
                self.step("2", "b.txt", TEXT_B),
            ),
        )
        self.assertTrue(report.passed)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("a.txt", client.calls[0])
        self.assertIn("b.txt", client.calls[1])
        self.assertEqual(
            [s.status for s in report.steps],
            [StepStatus.PASSED, StepStatus.PASSED],
        )

    def test_failed_step_recovers_and_becomes_passed(self) -> None:
        target = self.temp_dir / "a.txt"
        client = ScriptedClient({"a.txt": (2, target, TEXT_A)})
        report = self.run_steps(client, (self.step("1", "a.txt", TEXT_A),))
        self.assertTrue(report.passed)
        self.assertEqual(report.steps[0].status, StepStatus.PASSED)
        self.assertIsNotNone(report.steps[0].recovery)
        assert report.steps[0].recovery is not None
        self.assertEqual(report.steps[0].recovery.attempts, 2)
        self.assertEqual(len(report.steps[0].recovery.history), 2)
        self.assertEqual(client.counts["a.txt"], 2)

    def test_permanently_failed_step_becomes_failed(self) -> None:
        client = ScriptedClient({})  # never writes anything
        report = self.run_steps(client, (self.step("1", "a.txt", TEXT_A),))
        self.assertFalse(report.passed)
        self.assertEqual(report.steps[0].status, StepStatus.FAILED)
        self.assertIsNotNone(report.steps[0].recovery)
        self.assertIsNone(report.steps[0].error)

    def test_permanent_failure_is_fail_fast(self) -> None:
        client = ScriptedClient({})  # nothing ever verifies
        report = self.run_steps(
            client,
            (
                self.step("1", "a.txt", TEXT_A),
                self.step("2", "b.txt", TEXT_B),
                self.step("3", "c.txt", "content c"),
            ),
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.steps[0].status, StepStatus.FAILED)
        self.assertEqual(
            [s.status for s in report.steps[1:]],
            [StepStatus.SKIPPED, StepStatus.SKIPPED],
        )
        self.assertEqual(len(client.calls), 2)  # only step 1's bounded attempts
        for skipped in report.steps[1:]:
            self.assertIsNone(skipped.recovery)
            self.assertIsNone(skipped.error)

    def test_executor_exception_becomes_failed_with_error(self) -> None:
        client = ExplodingClient()
        report = self.run_steps(client, (self.step("1", "a.txt", TEXT_A),))
        self.assertFalse(report.passed)
        self.assertEqual(report.steps[0].status, StepStatus.FAILED)
        self.assertIsNotNone(report.steps[0].error)
        self.assertIn("executor exploded", report.steps[0].error)
        self.assertIsNone(report.steps[0].recovery)

    def test_executor_exception_is_not_retried(self) -> None:
        """An executor crash ends the step immediately.

        Recovery retries a *verification* failure only.  A crashing executor
        must consume exactly one call even though the default attempt budget is
        two, and must yield a clear failure result instead of a retry.
        """
        client = ExplodingClient()

        report = self.runner(client).run_plan(
            Plan(goal=Goal("initial"), steps=(self.step("1", "a.txt", TEXT_A),))
        )

        self.assertFalse(report.passed)
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(report.attempts), 1)
        self.assertEqual(report.steps[0].status, StepStatus.FAILED)
        self.assertIsNone(report.steps[0].recovery)
        self.assertIn("executor exploded", report.steps[0].error)


class TestPlanRunnerDependencies(PlanRunnerTestBase):
    def test_dependency_on_failed_step_is_skipped(self) -> None:
        b = self.temp_dir / "b.txt"
        client = ScriptedClient({})  # nothing ever verifies
        report = self.run_steps(
            client,
            (
                self.step("1", "a.txt", TEXT_A),
                self.step("2", "b.txt", TEXT_B, depends_on=("1",)),
            ),
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.steps[0].status, StepStatus.FAILED)
        self.assertEqual(report.steps[1].status, StepStatus.SKIPPED)
        self.assertIsNone(report.steps[1].recovery)
        self.assertEqual(len(client.calls), 2)  # fail-fast: no step 2 execution
        self.assertFalse(b.exists())

    def test_dependency_on_skipped_step_is_skipped(self) -> None:
        # A skipped step arises via fail-fast; model it directly: step 1
        # fails, steps 2 and 3 would be skipped — then depend step 3 on 2
        # through an explicit chained plan.
        c = self.temp_dir / "c.txt"
        client = ScriptedClient({})  # nothing ever verifies
        report = self.run_steps(
            client,
            (
                self.step("1", "a.txt", TEXT_A),
                self.step("2", "b.txt", TEXT_B, depends_on=("1",)),
                self.step("3", "c.txt", "content c", depends_on=("2",)),
            ),
        )
        self.assertFalse(report.passed)
        self.assertEqual(
            [s.status for s in report.steps],
            [StepStatus.FAILED, StepStatus.SKIPPED, StepStatus.SKIPPED],
        )
        self.assertIsNone(report.steps[2].recovery)
        self.assertFalse(c.exists())

    def test_independent_steps_after_successful_dependency_execute(self) -> None:
        a = self.temp_dir / "a.txt"
        b = self.temp_dir / "b.txt"
        client = ScriptedClient(
            {
                "a.txt": (1, a, TEXT_A),
                "b.txt": (1, b, TEXT_B),
            }
        )
        report = self.run_steps(
            client,
            (
                self.step("1", "a.txt", TEXT_A),
                self.step("2", "b.txt", TEXT_B, depends_on=("1",)),
            ),
        )
        self.assertTrue(report.passed)
        self.assertEqual(
            [s.status for s in report.steps],
            [StepStatus.PASSED, StepStatus.PASSED],
        )


class TestPlanRunnerPolicy(PlanRunnerTestBase):
    def test_timeout_is_forwarded_on_each_recovery_attempt(self) -> None:
        target = self.temp_dir / "a.txt"
        client = ScriptedClient({"a.txt": (2, target, TEXT_A)})

        runner = self.runner(
            client,
            timeout=4.25,
        )
        report = runner.run_plan(
            Plan(
                goal=Goal("fake goal"),
                steps=(self.step("1", "a.txt", TEXT_A),),
            )
        )

        self.assertTrue(report.passed)
        self.assertEqual(client.timeouts, [4.25, 4.25])

    def test_step_max_attempts_overrides_default(self) -> None:
        target = self.temp_dir / "a.txt"

        # Default 2 attempts: a step needing 3 fails permanently.
        client = ScriptedClient({"a.txt": (3, target, TEXT_A)})
        report = self.run_steps(client, (self.step("1", "a.txt", TEXT_A),))
        self.assertFalse(report.passed)
        self.assertEqual(report.steps[0].status, StepStatus.FAILED)
        self.assertEqual(client.counts["a.txt"], 2)

        # Explicit step.max_attempts=3 recovers on the third attempt.
        client2 = ScriptedClient({"a.txt": (3, target, TEXT_A)})
        runner2 = self.runner(client2, default_max_attempts=2)
        plan2 = Plan(
            goal=Goal("fake goal"),
            steps=(self.step("1", "a.txt", TEXT_A, max_attempts=3),),
        )
        report2 = runner2.run_plan(plan2)
        self.assertTrue(report2.passed)
        self.assertEqual(client2.counts["a.txt"], 3)
        assert report2.steps[0].recovery is not None
        self.assertEqual(report2.steps[0].recovery.attempts, 3)

    def test_report_aggregates_recovery_history(self) -> None:
        a = self.temp_dir / "a.txt"
        b = self.temp_dir / "b.txt"
        client = ScriptedClient(
            {
                "a.txt": (2, a, TEXT_A),  # recovers on attempt 2
                "b.txt": (1, b, TEXT_B),  # passes immediately
            }
        )
        report = self.run_steps(
            client,
            (
                self.step("1", "a.txt", TEXT_A),
                self.step("2", "b.txt", TEXT_B),
            ),
        )
        self.assertTrue(report.passed)
        first, second = report.steps
        assert first.recovery is not None and second.recovery is not None
        self.assertEqual(first.recovery.attempts, 2)
        self.assertEqual(len(first.recovery.history), 2)
        self.assertFalse(first.recovery.history[0].passed)
        self.assertTrue(first.recovery.history[1].passed)
        self.assertIs(first.recovery.verification, first.recovery.history[1])
        self.assertEqual(second.recovery.attempts, 1)
        self.assertEqual(len(second.recovery.history), 1)

    def test_empty_plan_rejected_through_validate_plan(self) -> None:
        runner = self.runner(ScriptedClient({}))
        with self.assertRaisesRegex(ValueError, "at least one step"):
            runner.run_plan(Plan(goal=Goal("fake goal"), steps=()))

    def test_invalid_dependency_rejected_through_validate_plan(self) -> None:
        runner = self.runner(ScriptedClient({}))
        plan = Plan(
            goal=Goal("fake goal"),
            steps=(PlanStep(id="1", action="write a.txt", depends_on=("ghost",)),),
        )
        with self.assertRaisesRegex(ValueError, "unknown step"):
            runner.run_plan(plan)

    def test_invalid_action_cannot_reach_the_executor(self) -> None:
        """``validate_plan`` runs before any step is handed to the executor.

        ``PlanStep`` already rejects a blank action at construction, so this
        mutates a frozen instance to prove the runner's own boundary still
        stops it with zero executor calls.
        """
        client = ScriptedClient({})
        step = self.step("1", "a.txt", TEXT_A)
        object.__setattr__(step, "action", "   ")

        runner = self.runner(client)
        with self.assertRaisesRegex(ValueError, "action must be non-empty"):
            runner.run_plan(Plan(goal=Goal("fake goal"), steps=(step,)))

        self.assertEqual(client.calls, [])

    def test_run_uses_planner_goal_and_thread_id_passthrough(self) -> None:
        captured: dict[str, Any] = {}

        class RecordingPlanner(DeterministicPlanner):
            def plan(self, goal: Goal) -> Plan:
                captured["goal"] = goal
                return Plan(
                    goal=goal,
                    steps=(PlanStep(id="step-1", action=goal.description),),
                )

        class RecordingClient:
            def __init__(self, base: Path) -> None:
                self.base = base

            def send_message(
                self,
                message: str,
                thread_id: str | None = None,
                timeout: float | None = None,
            ) -> str:
                captured["thread_id"] = thread_id
                (self.base / "done.txt").write_text(TEXT_A)
                return "fake response"

        client = RecordingClient(self.temp_dir)
        runner = PlanRunner(
            planner=RecordingPlanner(),
            client=client,
            verifier=self.verifier,
            base_dir=str(self.temp_dir),
            thread_id="thread-42",
        )
        goal = Goal("create done.txt")
        plan = runner.planner.plan(goal)
        plan = Plan(
            goal=goal,
            steps=(
                PlanStep(
                    id="step-1",
                    action=goal.description,
                    expectations=(
                        FileExpectation("done.txt", exists=True, contains=TEXT_A),
                    ),
                ),
            ),
        )
        report = runner.run_plan(plan)
        self.assertTrue(report.passed)
        self.assertEqual(captured["goal"], goal)
        self.assertEqual(captured["thread_id"], "thread-42")


class TestPlanRunnerToolBinding(PlanRunnerTestBase):
    """Per-step tool binding (Milestone 2F.8).

    ``PlanStep.tool is None`` keeps the runner's default ``client``; a declared
    name must be registered with the runner or the whole plan is rejected
    before any step (and therefore any side effect) runs.
    """

    def test_none_tool_uses_the_default_client(self) -> None:
        target = self.temp_dir / "a.txt"
        default = ScriptedClient({"a.txt": (1, target, TEXT_A)})
        unused = ScriptedClient({})

        report = self.runner(default, tools={"writer": unused}).run_plan(
            Plan(goal=Goal("g"), steps=(self.step("1", "a.txt", TEXT_A),))
        )

        self.assertTrue(report.passed)
        self.assertEqual(len(default.calls), 1)
        self.assertEqual(unused.calls, [])

    def test_declared_tool_routes_to_the_registered_executor(self) -> None:
        target = self.temp_dir / "tool.txt"
        default = ScriptedClient({})
        writer = ScriptedClient({"tool.txt": (1, target, TEXT_A)})

        report = self.runner(default, tools={"writer": writer}).run_plan(
            Plan(
                goal=Goal("g"),
                steps=(self.step("1", "tool.txt", TEXT_A, tool="writer"),),
            )
        )

        self.assertTrue(report.passed)
        self.assertTrue(target.exists())
        self.assertEqual(default.calls, [])
        self.assertEqual(len(writer.calls), 1)

    def test_each_step_resolves_its_own_executor(self) -> None:
        a = self.temp_dir / "a.txt"
        b = self.temp_dir / "b.txt"
        default = ScriptedClient({"a.txt": (1, a, TEXT_A)})
        writer = ScriptedClient({"b.txt": (1, b, TEXT_B)})

        report = self.runner(default, tools={"writer": writer}).run_plan(
            Plan(
                goal=Goal("g"),
                steps=(
                    self.step("1", "a.txt", TEXT_A),
                    self.step("2", "b.txt", TEXT_B, tool="writer"),
                ),
            )
        )

        self.assertTrue(report.passed)
        self.assertEqual(len(default.calls), 1)
        self.assertEqual(len(writer.calls), 1)
        self.assertIn("a.txt", default.calls[0])
        self.assertIn("b.txt", writer.calls[0])

    def test_unknown_tool_rejected_before_any_executor_call(self) -> None:
        default = ScriptedClient({})
        with self.assertRaisesRegex(ValueError, "unknown tool"):
            self.runner(default).run_plan(
                Plan(
                    goal=Goal("g"),
                    steps=(self.step("1", "a.txt", TEXT_A, tool="ghost"),),
                )
            )
        self.assertEqual(default.calls, [])

    def test_declared_tool_fails_closed_without_a_registry(self) -> None:
        default = ScriptedClient({})
        with self.assertRaisesRegex(ValueError, "unknown tool"):
            self.runner(default).run_plan(
                Plan(
                    goal=Goal("g"),
                    steps=(self.step("1", "a.txt", TEXT_A, tool="writer"),),
                )
            )
        self.assertEqual(default.calls, [])

    def test_declared_tool_fails_closed_when_not_in_registry(self) -> None:
        default = ScriptedClient({})
        other = ScriptedClient({})
        with self.assertRaisesRegex(ValueError, "unknown tool"):
            self.runner(default, tools={"other": other}).run_plan(
                Plan(
                    goal=Goal("g"),
                    steps=(self.step("1", "a.txt", TEXT_A, tool="writer"),),
                )
            )
        self.assertEqual(default.calls, [])
        self.assertEqual(other.calls, [])

    def test_unknown_tool_in_later_step_prevents_partial_execution(self) -> None:
        first = self.temp_dir / "a.txt"
        default = ScriptedClient({"a.txt": (1, first, TEXT_A)})

        with self.assertRaisesRegex(ValueError, "unknown tool"):
            self.runner(default).run_plan(
                Plan(
                    goal=Goal("g"),
                    steps=(
                        self.step("1", "a.txt", TEXT_A),
                        self.step("2", "b.txt", TEXT_B, tool="ghost"),
                    ),
                )
            )

        self.assertEqual(default.calls, [])
        self.assertFalse(first.exists())


class TestPlanRunnerReplanning(PlanRunnerTestBase):
    def replacement_plan(self, goal: Goal | None = None) -> Plan:
        return Plan(
            goal=goal or Goal("replacement goal"),
            steps=(self.step("replacement", "replacement.txt", TEXT_B),),
        )

    def test_failed_plan_replans_once_and_replacement_passes(self) -> None:
        client = ScriptedClient(
            {
                "a.txt": (99, self.temp_dir / "a.txt", TEXT_A),
                "replacement.txt": (1, self.temp_dir / "replacement.txt", TEXT_B),
            }
        )
        initial = Plan(
            goal=Goal("initial goal"),
            steps=(self.step("initial", "a.txt", TEXT_A),),
        )
        replanner = FixedReplanner(self.replacement_plan())
        report = self.runner(client, replanner=replanner).run_plan(initial)

        self.assertTrue(report.passed)
        self.assertEqual(len(replanner.calls), 1)
        self.assertEqual(len(report.attempts), 2)
        self.assertFalse(report.attempts[0].passed)
        self.assertTrue(report.attempts[1].passed)
        self.assertEqual(report.steps, report.attempts[1].steps)
        self.assertEqual(report.steps[0].step_id, "replacement")

    def test_replanner_receives_exact_failed_plan_and_report(self) -> None:
        client = ScriptedClient({})
        initial = Plan(
            goal=Goal("initial goal"),
            steps=(self.step("initial", "a.txt", TEXT_A),),
        )
        replanner = FixedReplanner(None)
        report = self.runner(client, replanner=replanner).run_plan(initial)

        self.assertFalse(report.passed)
        self.assertEqual(len(replanner.calls), 1)
        received_plan, received_report = replanner.calls[0]
        self.assertIs(received_plan, initial)
        self.assertIs(received_report, report)

    def test_passing_initial_plan_does_not_replan(self) -> None:
        target = self.temp_dir / "a.txt"
        replanner = FixedReplanner(self.replacement_plan())
        report = self.runner(
            ScriptedClient({"a.txt": (1, target, TEXT_A)}),
            replanner=replanner,
        ).run_plan(
            Plan(
                goal=Goal("initial"),
                steps=(self.step("initial", "a.txt", TEXT_A),),
            )
        )

        self.assertTrue(report.passed)
        self.assertEqual(replanner.calls, [])
        self.assertEqual(len(report.attempts), 1)

    def test_no_replanner_preserves_failed_behavior(self) -> None:
        report = self.run_steps(ScriptedClient({}), (self.step("initial", "a.txt", TEXT_A),))

        self.assertFalse(report.passed)
        self.assertEqual(len(report.attempts), 1)

    def test_replanner_returning_none_does_not_execute_replacement(self) -> None:
        client = ScriptedClient({})
        replanner = FixedReplanner(None)
        report = self.runner(client, replanner=replanner).run_plan(
            Plan(goal=Goal("initial"), steps=(self.step("initial", "a.txt", TEXT_A),))
        )

        self.assertFalse(report.passed)
        self.assertEqual(len(replanner.calls), 1)
        self.assertEqual(len(report.attempts), 1)
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(all("replacement.txt" not in call for call in client.calls))

    def test_failed_replacement_does_not_replan_again(self) -> None:
        replanner = FixedReplanner(self.replacement_plan())
        report = self.runner(
            ScriptedClient({}),
            replanner=replanner,
        ).run_plan(Plan(goal=Goal("initial"), steps=(self.step("initial", "a.txt", TEXT_A),)))

        self.assertFalse(report.passed)
        self.assertEqual(len(replanner.calls), 1)
        self.assertEqual(len(report.attempts), 2)
        self.assertFalse(report.attempts[1].passed)

    def test_executor_crash_replan_is_bounded_to_one_replacement(self) -> None:
        """A replacement entered via an executor crash is still bounded.

        ``PlanRunner`` hands any failing report to the replanner once.  A
        replanner that keeps returning a failing replacement must be consulted
        exactly once, and each plan costs exactly one executor call because a
        crash is never retried.
        """
        client = ExplodingClient()
        replanner = FixedReplanner(self.replacement_plan())

        report = self.runner(client, replanner=replanner).run_plan(
            Plan(goal=Goal("initial"), steps=(self.step("initial", "a.txt", TEXT_A),))
        )

        self.assertFalse(report.passed)
        self.assertEqual(len(replanner.calls), 1)
        self.assertEqual(len(report.attempts), 2)
        self.assertFalse(report.attempts[1].passed)
        self.assertEqual(client.calls, 2)  # one per plan, never retried

    def test_step_recovery_finishes_before_replanning(self) -> None:
        client = ScriptedClient(
            {
                "a.txt": (3, self.temp_dir / "a.txt", TEXT_A),
                "replacement.txt": (1, self.temp_dir / "replacement.txt", TEXT_B),
            }
        )
        replanner = FixedReplanner(self.replacement_plan())
        report = self.runner(client, replanner=replanner).run_plan(
            Plan(goal=Goal("initial"), steps=(self.step("initial", "a.txt", TEXT_A),))
        )

        self.assertTrue(report.passed)
        self.assertEqual(len(replanner.calls), 1)
        initial_report = replanner.calls[0][1]
        self.assertEqual(initial_report.attempts[0].steps[0].recovery.attempts, 2)
        self.assertEqual(client.counts["a.txt"], 2)

    def test_invalid_replacement_is_not_executed(self) -> None:
        invalid = Plan(goal=Goal("invalid"), steps=())
        replanner = FixedReplanner(invalid)
        client = ScriptedClient({})
        report = self.runner(
            client,
            replanner=replanner,
        ).run_plan(Plan(goal=Goal("initial"), steps=(self.step("initial", "a.txt", TEXT_A),)))

        self.assertFalse(report.passed)
        self.assertEqual(len(replanner.calls), 1)
        self.assertEqual(len(report.attempts), 1)
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(all("replacement.txt" not in call for call in client.calls))

    def test_replacement_with_unregistered_tool_is_discarded(self) -> None:
        """2F.8: a replacement declaring an unregistered tool is discarded.

        Tool validation is part of the existing replacement guard, so an
        unknown tool name returns the initial failed report instead of
        propagating ``ValueError``, and no executor runs the replacement.
        """
        invalid = Plan(
            goal=Goal("invalid"),
            steps=(
                self.step("replacement", "replacement.txt", TEXT_B, tool="ghost"),
            ),
        )
        replanner = FixedReplanner(invalid)
        default = ScriptedClient({})
        other = ScriptedClient({})

        report = self.runner(
            default,
            replanner=replanner,
            tools={"other": other},
        ).run_plan(Plan(goal=Goal("initial"), steps=(self.step("initial", "a.txt", TEXT_A),)))

        self.assertFalse(report.passed)
        self.assertEqual(len(replanner.calls), 1)
        self.assertEqual(len(report.attempts), 1)
        self.assertEqual(len(default.calls), 2)
        self.assertEqual(other.calls, [])
        self.assertTrue(all("replacement.txt" not in call for call in default.calls))


if __name__ == "__main__":
    unittest.main()
