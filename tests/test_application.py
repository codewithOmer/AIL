"""Focused tests for the 2M production application path."""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from core.application import AILApplication  # noqa: E402
from core.inspection import (  # noqa: E402
    INSPECTION_HEADER,
    WorkspaceInspector,
)
from core.planner import (  # noqa: E402
    DeterministicFileReplanner,
    FirstMatchPlanner,
    SupportedFileTaskPlanner,
    UnrecognizedTaskError,
    UnsupportedTaskError,
    WorkspaceSetupPlanner,
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
from interfaces.memory import Memory, MemoryStore  # noqa: E402
from core.verification import FileExpectation  # noqa: E402


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
        self.timeouts: list[float | None] = []

    def send_message(
        self,
        message: str,
        thread_id: str | None = None,
        timeout: float | None = None,
    ) -> str:
        self.events.append("execute")
        self.calls.append(message)
        self.timeouts.append(timeout)
        if self.write and len(self.calls) >= self.succeed_on:
            (self.base_dir / "hello.txt").write_text(
                "remember that I like coffee",
                encoding="utf-8",
            )
        return "executor response is not used for verification"


class WorkspaceWritingClient:
    """Materialises the multi-step workspace actions the production planner emits.

    Same contract as :class:`WritingClient`: the response text is never trusted,
    outcomes are decided by the real ``FilesystemVerifier`` reading the disk.
    """

    _DIRECTORY_ACTION = re.compile(
        r"^Create the directory (?P<directory>\S+) in the current workspace\.$"
    )
    _FILE_ACTION = re.compile(
        r"^Create (?P<path>\S+) inside \S+ containing exactly the text "
        r"(?P<content>.*)\.$"
    )

    def __init__(self, base_dir: Path, events: list[str]) -> None:
        self.base_dir = base_dir
        self.events = events
        self.calls: list[str] = []
        self.timeouts: list[float | None] = []

    def send_message(
        self,
        message: str,
        thread_id: str | None = None,
        timeout: float | None = None,
    ) -> str:
        self.events.append("execute")
        self.calls.append(message)
        self.timeouts.append(timeout)
        if match := self._DIRECTORY_ACTION.match(message):
            (self.base_dir / match.group("directory")).mkdir(
                parents=True, exist_ok=True
            )
        elif match := self._FILE_ACTION.match(message):
            target = self.base_dir / match.group("path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(match.group("content"), encoding="utf-8")
        return "executor response is not used for verification"


class TestApplication(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def make_application(
        self,
        write: bool = True,
        timeout: float | None = None,
    ) -> tuple[AILApplication, list[str], WritingClient]:
        events: list[str] = []
        memory = RecordingMemoryStore(events)
        client = WritingClient(self.temp_dir, events, write=write)
        application = AILApplication.create(
            base_dir=self.temp_dir,
            memory_store=memory,
            client=client,
            start_client=False,
            timeout=timeout,
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
        self.assertEqual(events, ["execute"])
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

        self.assertEqual(events, [])
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
            "FirstMatchPlanner",
        )
        self.assertEqual(
            [
                type(p).__name__
                for p in application.agent.plan_runner.planner.planners
            ],
            ["WorkspaceSetupPlanner", "SupportedFileTaskPlanner"],
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

    def test_application_timeout_reaches_executor(self) -> None:
        application, _, client = self.make_application(timeout=6.25)

        report = application.run(
            "Create a file named hello.txt containing remember that I like coffee"
        )

        self.assertTrue(report.passed)
        self.assertEqual(client.timeouts, [6.25])

    def test_application_default_timeout_preserves_none(self) -> None:
        application, _, client = self.make_application()

        report = application.run(
            "Create a file named hello.txt containing remember that I like coffee"
        )

        self.assertTrue(report.passed)
        self.assertEqual(client.timeouts, [None])

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


class TestProductionWorkspacePlanner(unittest.TestCase):
    """The composed production planner, exercised through the real
    application -> planner -> runner -> verifier path with a fake executor.
    """

    _WORKSPACE_GOAL = (
        "Set up a project workspace src with a readme containing hello world "
        "and a config file containing {}"
    )

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def make_application(self, client=None, planner=None):
        events: list[str] = []
        return AILApplication.create(
            base_dir=self.temp_dir,
            memory_store=RecordingMemoryStore(events),
            client=client or WorkspaceWritingClient(self.temp_dir, events),
            planner=planner,
            start_client=False,
        ), events

    # --- 1. workspace goal is reachable and executes ---------------------

    def test_default_planner_executes_valid_workspace_goal(self) -> None:
        application, _ = self.make_application()

        report = application.run(self._WORKSPACE_GOAL)

        self.assertTrue(report.passed)
        self.assertEqual(report.goal.description, self._WORKSPACE_GOAL)

    def test_valid_workspace_goal_executes_exactly_three_ordered_steps(self) -> None:
        application, _ = self.make_application()

        report = application.run(self._WORKSPACE_GOAL)

        self.assertEqual(
            [step.step_id for step in report.steps],
            ["create-workspace", "create-readme", "create-config"],
        )
        self.assertEqual(
            [step.status for step in report.steps],
            [StepStatus.PASSED, StepStatus.PASSED, StepStatus.PASSED],
        )
        self.assertTrue((self.temp_dir / "src" / "README.md").is_file())
        self.assertTrue((self.temp_dir / "src" / "config.json").is_file())

    # --- 2. existing create-file behaviour is preserved ------------------

    def test_default_planner_still_executes_create_file_syntax(self) -> None:
        events: list[str] = []
        client = WritingClient(self.temp_dir, events)
        application, _ = self.make_application(client=client)

        report = application.run("Create a file named hello.txt containing coffee")

        self.assertTrue(report.passed)
        self.assertEqual([step.step_id for step in report.steps], ["create-file"])
        expectation = report.attempts[0].plan.steps[0].expectations[0]
        self.assertEqual(expectation.path, "hello.txt")
        self.assertEqual(expectation.contains, "coffee")
        self.assertEqual(len(client.calls), 1)

    # --- 3. explicit planner injection is preserved ----------------------

    def test_explicit_planner_injection_is_preserved(self) -> None:
        custom = SupportedFileTaskPlanner()
        application, _ = self.make_application(planner=custom)

        self.assertIs(application.agent.plan_runner.planner, custom)

    def test_explicit_planner_injection_bypasses_the_composite(self) -> None:
        application, _ = self.make_application(planner=SupportedFileTaskPlanner())

        with self.assertRaises(UnsupportedTaskError):
            application.run(self._WORKSPACE_GOAL)

    # --- 4. unsafe workspace paths never fall through ---------------------

    def test_unsafe_workspace_path_is_rejected_without_falling_through(self) -> None:
        application, events = self.make_application()

        for directory in (
            "../escape",
            r"..\escape",
            "/etc",
            r"C:\outside",
            "\\\\server\\share",
        ):
            with self.subTest(directory=directory):
                with self.assertRaises(UnsupportedTaskError) as caught:
                    application.run(
                        f"set up a project workspace {directory} "
                        "with a readme containing hi "
                        "and a config file containing bye"
                    )
                # An unsafe path must be an explicit rejection, never a
                # recognition miss that would fall through to another planner.
                self.assertNotIsInstance(
                    caught.exception, UnrecognizedTaskError
                )
                self.assertIsInstance(caught.exception, UnsupportedTaskError)

        # Nothing was executed, and nothing escaped the workspace.
        self.assertEqual(events, [])
        self.assertFalse((self.temp_dir.parent / "escape").exists())
        self.assertFalse((Path("/etc") / "README.md").exists())

    # --- 5. unrelated goals still raise the normal unsupported error ----

    def test_completely_unsupported_goal_still_raises(self) -> None:
        application, events = self.make_application()

        with self.assertRaises(UnsupportedTaskError):
            application.run("Calculate 23 times 19")

        self.assertEqual(events, [])
        self.assertEqual(
            application.agent.plan_runner.replanner.__class__.__name__,
            "DeterministicFileReplanner",
        )


class TestProductionInspectionWiring(unittest.TestCase):
    """The production composition wires a real workspace inspector."""

    _WORKSPACE_GOAL = (
        "Set up a project workspace src with a readme containing hello world "
        "and a config file containing {}"
    )
    _FILE_GOAL = (
        "Create a file named hello.txt containing remember that I like coffee"
    )

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def make_application(self, client=None, **kwargs):
        events: list[str] = []
        application = AILApplication.create(
            base_dir=self.temp_dir,
            memory_store=RecordingMemoryStore(events),
            client=client or WritingClient(self.temp_dir, events),
            start_client=False,
            **kwargs,
        )
        return application, events

    def test_production_application_wires_a_workspace_inspector(self) -> None:
        application, _ = self.make_application()

        inspector = application.agent.inspector

        self.assertIsInstance(inspector, WorkspaceInspector)

    def test_configured_base_dir_reaches_the_inspector(self) -> None:
        application, _ = self.make_application()

        inspector = application.agent.inspector

        self.assertEqual(inspector.base_dir, self.temp_dir.resolve())
        self.assertEqual(
            inspector.base_dir,
            Path(application.agent.plan_runner.base_dir).resolve(),
        )

    def test_injected_inspector_is_used_instead_of_the_default(self) -> None:
        other = Path(tempfile.mkdtemp())
        try:
            (other / "custom.txt").write_text("x")
            injected = WorkspaceInspector(other)

            application, _ = self.make_application(inspector=injected)

            self.assertIs(application.agent.inspector, injected)
        finally:
            shutil.rmtree(other, ignore_errors=True)

    def test_existing_file_task_planning_still_works(self) -> None:
        application, _ = self.make_application()

        report = application.run(self._FILE_GOAL)

        self.assertTrue(report.passed)
        self.assertEqual(report.goal.description, self._FILE_GOAL)
        self.assertTrue((self.temp_dir / "hello.txt").is_file())

    def test_workspace_setup_still_produces_the_same_three_step_plan(self) -> None:
        events: list[str] = []
        application, _ = self.make_application(
            client=WorkspaceWritingClient(self.temp_dir, events)
        )

        report = application.run(self._WORKSPACE_GOAL)

        self.assertTrue(report.passed)
        self.assertEqual(
            [step.step_id for step in report.steps],
            ["create-workspace", "create-readme", "create-config"],
        )
        self.assertTrue((self.temp_dir / "src" / "README.md").is_file())
        self.assertTrue((self.temp_dir / "src" / "config.json").is_file())

    def test_inspection_context_does_not_change_the_reported_goal(self) -> None:
        (self.temp_dir / "existing.txt").write_text("already here")
        application, _ = self.make_application()

        report = application.run(self._FILE_GOAL)

        # The inspection block is prepended and the user goal still ends the
        # planner-visible description unchanged.
        description = report.goal.description
        self.assertEqual(description, self._FILE_GOAL)
        self.assertEqual(
            report.attempts[0].plan.goal.description, description
        )
        self.assertTrue(report.passed)


if __name__ == "__main__":
    unittest.main()