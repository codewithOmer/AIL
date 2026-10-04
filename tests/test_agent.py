"""Focused tests for the thin AIL Agent orchestration layer."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from core.agent import Agent, MockLLM  # noqa: E402
from core.inspection import (  # noqa: E402
    INSPECTION_FOOTER,
    INSPECTION_HEADER,
    WorkspaceInspector,
)
from interfaces.planning import ExecutionReport, Goal  # noqa: E402
from interfaces.memory import Memory, MemoryStore  # noqa: E402


class FakeMemoryStore(MemoryStore):
    def __init__(self, recalled: list[Memory]) -> None:
        self.recalled = recalled
        self.calls: list[tuple[str, int]] = []

    def store(self, text: str, *, kind: str = "semantic", confidence: float = 0.8) -> Memory:
        raise AssertionError("Agent must not store memories")

    def recall(self, query: str, *, top_k: int = 5) -> list[Memory]:
        self.calls.append((query, top_k))
        return self.recalled

    def list_all(self) -> list[Memory]:
        raise AssertionError("Agent must not list memories")

    def get(self, memory_id: str) -> Memory | None:
        raise AssertionError("Agent must not get memories")

    def delete(self, memory_id: str) -> None:
        raise AssertionError("Agent must not delete memories")


class FakePlanRunner:
    def __init__(self, report: ExecutionReport) -> None:
        self.report = report
        self.goals: list[Goal] = []

    def run(self, goal: Goal) -> ExecutionReport:
        self.goals.append(goal)
        return self.report


class TestAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.report = ExecutionReport(
            goal=Goal("planned goal"),
            passed=True,
            steps=(),
        )

    def test_run_accepts_string_and_returns_runner_report(self) -> None:
        memory = FakeMemoryStore([])
        runner = FakePlanRunner(self.report)

        result = Agent(memory, runner).run("Create a file", top_k=4)

        self.assertIs(result, self.report)
        self.assertEqual(memory.calls, [("Create a file", 4)])
        self.assertEqual(runner.goals, [Goal("Create a file")])

    def test_run_accepts_goal_and_recalls_original_text(self) -> None:
        original = Goal("Create hello.txt")
        memory = FakeMemoryStore(
            [Memory(id="1", text="The user prefers UTF-8 text files.")]
        )
        runner = FakePlanRunner(self.report)

        Agent(memory, runner).run(original, top_k=2)

        self.assertEqual(memory.calls, [("Create hello.txt", 2)])
        self.assertEqual(
            runner.goals[0].description,
            "[Memory context from previous conversations:]\n"
            "- The user prefers UTF-8 text files.\n\n"
            "Create hello.txt",
        )

    def test_empty_memory_keeps_goal_unchanged(self) -> None:
        memory = FakeMemoryStore([])
        runner = FakePlanRunner(self.report)

        Agent(memory, runner).run(Goal("Create hello.txt"))

        self.assertEqual(runner.goals, [Goal("Create hello.txt")])

    def test_legacy_respond_behavior_remains_compatible(self) -> None:
        agent = Agent(MockLLM())

        self.assertEqual(agent.respond("hello"), "AIL received: hello")


class TestAgentInspectionContext(unittest.TestCase):
    """Inspection is offered to the planner as labelled data."""

    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp())
        self.report = ExecutionReport(goal=Goal("planned"), passed=True, steps=())

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)

    def make_agent(
        self, recalled: list[Memory] | None = None
    ) -> tuple[Agent, FakePlanRunner]:
        runner = FakePlanRunner(self.report)
        agent = Agent(
            FakeMemoryStore(recalled or []),
            runner,
            inspector=WorkspaceInspector(self.workspace),
        )
        return agent, runner

    def test_inspection_context_reaches_the_planner(self) -> None:
        (self.workspace / "notes.txt").write_text("data")
        (self.workspace / "src").mkdir()
        agent, runner = self.make_agent()

        agent.run("Create hello.txt")

        description = runner.goals[0].description
        self.assertIn("- notes.txt", description)
        self.assertIn("- src", description)

    def test_inspection_context_is_labelled_and_delimited_as_data(self) -> None:
        (self.workspace / "notes.txt").write_text("data")
        agent, runner = self.make_agent()

        agent.run("Create hello.txt")

        description = runner.goals[0].description
        self.assertTrue(description.startswith(INSPECTION_HEADER))
        self.assertIn(INSPECTION_FOOTER, description)
        self.assertIn("not verification evidence", description)

    def test_memory_and_inspection_context_coexist_with_goal_last(self) -> None:
        (self.workspace / "notes.txt").write_text("data")
        recalled = [Memory(id="1", text="The user prefers UTF-8.")]
        agent, runner = self.make_agent(recalled)

        agent.run("Create hello.txt")

        description = runner.goals[0].description
        self.assertIn(INSPECTION_HEADER, description)
        self.assertIn("- notes.txt", description)
        self.assertIn("[Memory context from previous conversations:]", description)
        self.assertIn("- The user prefers UTF-8.", description)
        # The user goal stays last so planner goal-matching is unchanged.
        self.assertTrue(description.endswith("Create hello.txt"))

    def test_empty_workspace_adds_no_inspection_context(self) -> None:
        agent, runner = self.make_agent()

        agent.run("Create hello.txt")

        description = runner.goals[0].description
        self.assertEqual(description, "Create hello.txt")
        self.assertNotIn(INSPECTION_HEADER, description)

    def test_inspector_none_leaves_goal_unchanged(self) -> None:
        (self.workspace / "notes.txt").write_text("data")
        runner = FakePlanRunner(self.report)
        agent = Agent(FakeMemoryStore([]), runner, inspector=None)

        agent.run("Create hello.txt")

        self.assertEqual(runner.goals, [Goal("Create hello.txt")])


if __name__ == "__main__":
    unittest.main()