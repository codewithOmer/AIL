from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from core.inspection import WorkspaceInspector
from core.planner import UnsupportedTaskError
from interfaces.image import LocalImage
from interfaces.llm import LLM
from interfaces.memory import MemoryStore
from interfaces.planning import ExecutionReport, Goal

if TYPE_CHECKING:
    from core.plan_runner import PlanRunner


class MockLLM(LLM):
    """Temporary model used to test AIL architecture."""

    def generate(self, prompt: str) -> str:
        return f"AIL received: {prompt}"


class Agent:
    """Thin orchestration layer over plan execution.

    Instruction/data separation (Milestone 2F.5): the ``Goal`` handed to the
    planner carries the actual user instruction and nothing else.  Recalled
    memory and workspace inspection are untrusted data, so they are never
    fetched into, nor concatenated onto, the planner's input.  The
    deterministic planner parses narrow task syntax only and needs neither to
    build a plan.
    """

    def __init__(
        self,
        memory_store: MemoryStore | LLM,
        plan_runner: PlanRunner | None = None,
        inspector: WorkspaceInspector | None = None,
    ) -> None:
        # Keep the original constructor shape working for mock mode.
        self.llm = memory_store if plan_runner is None else None
        self.memory_store = memory_store if plan_runner is not None else None
        self.plan_runner = plan_runner
        self.inspector = inspector

    def respond(self, user_input: str) -> str:
        if self.llm is None:
            raise RuntimeError("respond() is only available in legacy LLM mode")
        return self.llm.generate(user_input)

    def run(
        self,
        goal: str | Goal,
        *,
        top_k: int = 3,
        images: Sequence[LocalImage] | None = None,
    ) -> ExecutionReport:
        """Plan and execute the user instruction, and return its report.

        ``top_k`` is accepted for interface compatibility and is deliberately
        unused: no memory or inspection data is fetched for planning, because
        neither may become planner instruction.  A blank instruction is
        rejected here, before any planner runs, so surrounding context can
        never turn an empty request into a task.
        """
        if self.memory_store is None or self.plan_runner is None:
            raise RuntimeError("run() requires a memory store and plan runner")

        clean_goal = Goal(goal) if isinstance(goal, str) else goal
        if not clean_goal.description.strip():
            raise UnsupportedTaskError(
                "goal must be a non-empty user instruction"
            )

        if images:
            return self.plan_runner.run(clean_goal, images=images)
        return self.plan_runner.run(clean_goal)
