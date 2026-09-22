"""Focused tests for the thin AIL Agent orchestration layer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

AIL_ROOT = str(Path(__file__).resolve().parents[1])
if AIL_ROOT not in sys.path:
    sys.path.insert(0, AIL_ROOT)

from core.agent import Agent, MockLLM  # noqa: E402
from interfaces.planning import ExecutionReport, Goal  # noqa: E402
from memory.interface import Memory, MemoryStore  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()