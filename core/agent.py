from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from core.inspection import WorkspaceInspector, apply_inspection_context
from interfaces.image import LocalImage
from interfaces.llm import LLM
from interfaces.memory import MemoryStore
from interfaces.planning import ExecutionReport, Goal
from memory.flow import apply_context

if TYPE_CHECKING:
    from core.plan_runner import PlanRunner


class MockLLM(LLM):
    """Temporary model used to test AIL architecture."""

    def generate(self, prompt: str) -> str:
        return f"AIL received: {prompt}"


class Agent:
    """Thin orchestration layer over memory recall and plan execution."""

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
        """Recall context, execute the contextual goal, and return its report."""
        if self.memory_store is None or self.plan_runner is None:
            raise RuntimeError("run() requires a memory store and plan runner")

        original_goal = Goal(goal) if isinstance(goal, str) else goal
        recalled = self.memory_store.recall(
            original_goal.description,
            top_k=top_k,
        )
        # Inspection is offered to the planner as labelled, untrusted data.
        # It is never verification evidence: the verifier alone decides whether
        # a step passed, by reading the filesystem after execution.
        description = apply_inspection_context(
            self.inspector,
            apply_context(original_goal.description, recalled),
        )
        contextual_goal = Goal(description)
        if images:
            return self.plan_runner.run(contextual_goal, images=images)
        return self.plan_runner.run(contextual_goal)