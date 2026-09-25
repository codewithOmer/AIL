"""Focused tests for the 2M production application path."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

AIL_ROOT = str(Path(__file__).resolve().parents[1])
if AIL_ROOT not in sys.path:
    sys.path.insert(0, AIL_ROOT)

from core.application import AILApplication  # noqa: E402
from core.planner import (  # noqa: E402
    DeterministicFileReplanner,
    SupportedFileTaskPlanner,
    UnsupportedTaskError,
)
from interfaces.planning import (  # noqa: E402
    ExecutionReport,
    Goal,
    Plan,
    PlanStep,
    StepResult,
    StepStatus,
)
from integrations.open_interpreter.config import OIConfig  # noqa: E402
from memory.interface import Memory, MemoryStore  # noqa: E402
from tools.fs_verifier import FileExpectation  # noqa: E402


class RecordingMemoryStore(MemoryStore):
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.memories: list[Memory] = []

    def store(
        self,
        text: str,
        *,
        kind: str = "semantic",
        confidence: float = 0.8,
    ) -> Memory:
        self.events.append("store")
        memory = Memory(
            id=str(len(self.memories)),
            text=text,
            kind=kind,
            confidence=confidence,
        )
        self.memories.append(memory)
        return memory

    def recall(self, query: str, *, top_k: int = 5) -> list[Memory]:
        self.events.append("recall")
        return []

    def list_all(self) -> list[Memory]:
        self.events.append("list_all")
        return list(self.memories)

    def get(self, memory_id: str) -> Memory | None:
        return next((memory for memory in self.memories if memory.id == memory_id), None)

    def delete(self, memory_id: str) -> None:
        self.memories = [memory for memory in self.memories if memory.id != memory_id]


class WritingClient:
    def __init__(
        self,
        base_dir: Path,
        events: list[str],
        write: bool = True,
        succeed_on: int = 1,
    ) -> None:
        self.base_dir = base_dir
        self.events = events
        self.write = write
        self.succeed_on = succeed_on
        self.calls: list[str] = []

    def send_message(
        self,
        message: str,
        thread_id: str | None = None,
        timeout: float | None = None,
    ) -> str:
        self.events.append("execute")
        self.calls.append(message)
        if self.write and len(self.calls) >= self.succeed_on:
            (self.base_dir / "hello.txt").write_text(
                "remember that I like coffee",
                encoding="utf-8",
            )
        return "executor response is not used for verification"


class TestApplication(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def make_application(
        self, write: bool = True
    ) -> tuple[AILApplication, list[str], WritingClient]:
        events: list[str] = []
        memory = RecordingMemoryStore(events)
        client = WritingClient(self.temp_dir, events, write=write)
        application = AILApplication.create(
            base_dir=self.temp_dir,
            memory_store=memory,
            client=client,
            start_client=False,
        )
        return application, events, client

    def test_supported_task_has_explicit_expectation_and_executes(self) -> None:
        application, events, client = self.make_application()

        report = application.run(
            "Create a file named hello.txt containing remember that I like coffee"
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.steps[0].step_id, "create-file")
        expectation = report.attempts[0].plan.steps[0].expectations[0]
        self.assertEqual(expectation.path, "hello.txt")
        self.assertEqual(expectation.contains, "remember that I like coffee")
        self.assertEqual(events, ["recall", "execute", "list_all", "store"])
        self.assertEqual(len(client.calls), 1)

    def test_existing_short_file_task_syntax_is_supported(self) -> None:
        application, _, _ = self.make_application()

        report = application.run("Create hello.txt containing remember that I like coffee")

        self.assertTrue(report.passed)
        self.assertEqual(
            report.attempts[0].plan.steps[0].expectations[0].path,
            "hello.txt",
        )

    def test_supported_planner_accepts_normal_relative_filenames(self) -> None:
        planner = SupportedFileTaskPlanner()

        for filename in ("2m_real_test.txt", "notes.txt", "project_summary.md"):
            with self.subTest(filename=filename):
                plan = planner.plan(
                    Goal(f"Create a file named {filename} containing content")
                )
                self.assertEqual(plan.steps[0].expectations[0].path, filename)

    def test_supported_planner_rejects_workspace_escaping_paths(self) -> None:
        planner = SupportedFileTaskPlanner()

        for filename in ("../outside.txt", r"..\outside.txt", r"C:\outside.txt"):
            with self.subTest(filename=filename):
                with self.assertRaises(UnsupportedTaskError):
                    planner.plan(
                        Goal(f"Create a file named {filename} containing content")
                    )

    def test_current_task_wins_over_matching_recalled_context(self) -> None:
        application, _, _ = self.make_application()
        application.agent.memory_store.recall = lambda query, top_k=5: [
            Memory(id="old", text="Create a file named old.txt containing old")
        ]

        report = application.run("Create a file named hello.txt containing current")

        self.assertEqual(
            report.attempts[0].plan.steps[0].expectations[0].path,
            "hello.txt",
        )

    def test_unsupported_task_does_not_call_executor(self) -> None:
        application, events, client = self.make_application()

        with self.assertRaises(UnsupportedTaskError):
            application.run("Calculate 23 times 19")

        self.assertEqual(events, ["recall"])
        self.assertEqual(client.calls, [])

    def test_failed_verification_keeps_bounded_recovery(self) -> None:
        application, _, client = self.make_application(write=False)

        report = application.run("Create a file named hello.txt containing expected")

        self.assertFalse(report.passed)
        self.assertEqual(report.steps[0].status, StepStatus.FAILED)
        self.assertIsNotNone(report.steps[0].recovery)
        assert report.steps[0].recovery is not None
        self.assertEqual(report.steps[0].recovery.attempts, 2)
        self.assertEqual(len(report.attempts), 2)
        self.assertEqual(len(client.calls), 4)

    def test_application_composes_without_starting_oi(self) -> None:
        application, _, _ = self.make_application()

        self.assertEqual(type(application.agent).__name__, "Agent")
        self.assertEqual(type(application.agent.plan_runner).__name__, "PlanRunner")
        self.assertEqual(
            type(application.agent.plan_runner.verifier).__name__,
            "FilesystemVerifier",
        )
        self.assertEqual(
            type(application.agent.plan_runner.planner).__name__,
            "SupportedFileTaskPlanner",
        )
        self.assertIsInstance(
            application.agent.plan_runner.replanner,
            DeterministicFileReplanner,
        )

    def test_default_oi_configuration_allows_workspace_writes(self) -> None:
        with patch("core.application.OpenInterpreterClient") as client_factory:
            AILApplication.create(
                base_dir=self.temp_dir,
                memory_store=RecordingMemoryStore([]),
                start_client=False,
            )

        config = client_factory.call_args.args[0]
        self.assertEqual(config.sandbox, "workspace-write")
        self.assertEqual(config.cwd, str(self.temp_dir.resolve()))

    def test_explicit_oi_sandbox_is_preserved(self) -> None:
        explicit = OIConfig(sandbox="read-only")
        with patch("core.application.OpenInterpreterClient") as client_factory:
            AILApplication.create(
                base_dir=self.temp_dir,
                memory_store=RecordingMemoryStore([]),
                oi_config=explicit,
                start_client=False,
            )

        config = client_factory.call_args.args[0]
        self.assertEqual(config.sandbox, "read-only")
        self.assertEqual(config.cwd, str(self.temp_dir.resolve()))

    def test_failed_initial_plan_can_pass_on_one_replacement(self) -> None:
        events: list[str] = []
        memory = RecordingMemoryStore(events)
        client = WritingClient(self.temp_dir, events, succeed_on=3)
        application = AILApplication.create(
            base_dir=self.temp_dir,
            memory_store=memory,
            client=client,
            start_client=False,
        )

        report = application.run("Create hello.txt containing remember that I like coffee")

        self.assertTrue(report.passed)
        self.assertEqual(len(report.attempts), 2)
        self.assertEqual(len(client.calls), 3)
        replacement = report.attempts[1].plan
        self.assertIsInstance(
            replacement.steps[0].expectations[0],
            FileExpectation,
        )

    def test_failed_replacement_does_not_replan_again(self) -> None:
        application, _, client = self.make_application(write=False)

        report = application.run("Create hello.txt containing expected")

        self.assertFalse(report.passed)
        self.assertEqual(len(report.attempts), 2)
        self.assertEqual(len(client.calls), 4)

    def test_replanner_declines_unsupported_or_unrecoverable_failures(self) -> None:
        replanner = DeterministicFileReplanner()
        expectation = FileExpectation("hello.txt", contains="expected")
        unsupported = Plan(
            goal=Goal("Calculate 23 times 19"),
            steps=(PlanStep(id="create-file", action="calculate", expectations=(expectation,)),),
        )
        failed_report = ExecutionReport(
            goal=unsupported.goal,
            passed=False,
            steps=(
                StepResult(
                    step_id="create-file",
                    action="calculate",
                    status=StepStatus.FAILED,
                ),
            ),
            attempts=(),
        )
        self.assertIsNone(replanner.replan(unsupported, failed_report))

        supported = Plan(
            goal=Goal("Create hello.txt containing expected"),
            steps=(PlanStep(id="create-file", action="create", expectations=(expectation,)),),
        )
        unrecoverable_report = ExecutionReport(
            goal=supported.goal,
            passed=False,
            steps=(
                StepResult(
                    step_id="create-file",
                    action="create",
                    status=StepStatus.FAILED,
                    error="executor exploded",
                ),
            ),
            attempts=(),
        )
        self.assertIsNone(replanner.replan(supported, unrecoverable_report))


if __name__ == "__main__":
    unittest.main()