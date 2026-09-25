"""Production composition for AIL's bounded verified-task path."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from core.agent import Agent
from core.intent_router import DefaultIntentRouter
from core.plan_runner import PlanRunner
from core.planner import DeterministicFileReplanner, SupportedFileTaskPlanner
from integrations.open_interpreter.client import OpenInterpreterClient
from integrations.open_interpreter.config import OIConfig
from integrations.personalai.memory import PersonalAIMemoryStore
from interfaces.intent import IntentKind, IntentRouter
from interfaces.planning import ExecutionReport, Planner, Replanner
from memory.flow import extract_explicit_facts, store_if_new
from memory.interface import Memory, MemoryStore
from tools.fs_verifier import FilesystemVerifier


@dataclass
class AILApplication:
    """Own the dependencies and lifecycle for one AIL task session."""

    memory_store: MemoryStore
    client: Any
    agent: Agent
    intent_router: IntentRouter
    thread_id: str | None = None
    owns_client: bool = True

    @classmethod
    def create(
        cls,
        *,
        base_dir: str | Path | None = None,
        memory_store: MemoryStore | None = None,
        client: Any | None = None,
        planner: Planner | None = None,
        verifier: Any | None = None,
        replanner: Replanner | None = None,
        oi_config: OIConfig | None = None,
        start_client: bool = True,
        timeout: float | None = None,
        intent_router: IntentRouter | None = None,
    ) -> "AILApplication":
        config = oi_config or replace(OIConfig(), sandbox="workspace-write")
        workspace = Path(base_dir or config.cwd).resolve()
        config = replace(config, cwd=str(workspace))
        store = memory_store or PersonalAIMemoryStore()
        runtime_client = client or OpenInterpreterClient(config)

        thread_id: str | None = None
        if start_client:
            runtime_client.start()
            thread_id = runtime_client.create_thread(cwd=str(workspace))

        runner = PlanRunner(
            planner=planner or SupportedFileTaskPlanner(),
            client=runtime_client,
            verifier=verifier or FilesystemVerifier(),
            base_dir=str(workspace),
            thread_id=thread_id,
            replanner=replanner or DeterministicFileReplanner(),
            timeout=timeout,
        )
        return cls(
            memory_store=store,
            client=runtime_client,
            agent=Agent(store, runner),
            intent_router=intent_router or DefaultIntentRouter(),
            thread_id=thread_id,
            owns_client=client is None,
        )

    def run(self, message: str, *, top_k: int = 3) -> ExecutionReport:
        """Run a task through the existing Agent pipeline."""
        return self.agent.run(message, top_k=top_k)

    def handle(
        self, message: str, *, top_k: int = 3
    ) -> ExecutionReport | tuple[Memory, ...]:
        """Route one message to either memory writing or task execution."""
        intent = self.intent_router.route(message)
        if intent.kind is IntentKind.MEMORY:
            return self.remember(message)
        return self.agent.run(message, top_k=top_k)

    def remember(self, message: str) -> tuple[Memory, ...]:
        stored: list[Memory] = []
        for fact in extract_explicit_facts(message):
            memory = store_if_new(self.memory_store, fact)
            if memory is not None:
                stored.append(memory)
        return tuple(stored)

    def close(self) -> None:
        if self.owns_client:
            self.client.shutdown()