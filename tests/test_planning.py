"""Tests for AIL planning foundation (Milestone 2F).

Covers deterministic plan creation and integrity validation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

# Ensure AIL root is importable.
AIL_ROOT = str(Path(__file__).resolve().parents[1])
if AIL_ROOT not in sys.path:
    sys.path.insert(0, AIL_ROOT)

from core.planner import DeterministicPlanner, validate_plan  # noqa: E402
from interfaces.planning import Goal, Plan, PlanStep  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
