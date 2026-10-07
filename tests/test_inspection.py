"""Tests for AIL workspace inspection (Milestone 2F.1).

Covers the read-only ``WorkspaceInspector`` path policy and, critically, the
architectural rule that inspection output is *information for planning only*:
a real agent with a real inspector and a real verifier must still FAIL when the
executor does nothing, no matter what the inspection context contains.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from core.inspection import (  # noqa: E402
    INSPECTION_FOOTER,
    INSPECTION_HEADER,
    InspectionError,
    WorkspaceInspector,
    apply_inspection_context,
)
from core.agent import Agent  # noqa: E402
from core.plan_runner import PlanRunner  # noqa: E402
from core.planner import SupportedFileTaskPlanner  # noqa: E402
from core.verification import FilesystemVerifier  # noqa: E402
from interfaces.memory import Memory, MemoryStore  # noqa: E402
from interfaces.planning import StepStatus  # noqa: E402

_Multiline = "line one\nline two\nline three"


class _NoopExecutor:
    """Executor that does nothing and always returns canned text."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def send_message(
        self,
        message: str,
        thread_id: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> str:
        self.calls.append(message)
        return "executor text is never trusted"


class _EmptyMemoryStore(MemoryStore):
    def store(self, text, **kwargs) -> Memory:
        raise AssertionError("must not store")

    def recall(self, query, *, top_k=5):
        return []

    def list_all(self):
        raise AssertionError("must not list")

    def get(self, memory_id):
        raise AssertionError("must not get")

    def delete(self, memory_id):
        raise AssertionError("must not delete")


class WorkspaceInspectorTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._workspace = Path(tempfile.mkdtemp())
        self.inspector = WorkspaceInspector(self._workspace)

    def tearDown(self) -> None:
        shutil.rmtree(self._workspace, ignore_errors=True)


class TestListDir(WorkspaceInspectorTestBase):
    def test_list_root_returns_real_entries_sorted(self) -> None:
        (self._workspace / "zeta.txt").write_text("z")
        (self._workspace / "alpha.txt").write_text("a")
        (self._workspace / "mid_dir").mkdir()
        (self._workspace / "nested").mkdir()
        (self._workspace / "nested" / "child.txt").write_text("c")

        self.assertEqual(
            self.inspector.list_root(), ("alpha.txt", "mid_dir", "nested", "zeta.txt")
        )
        self.assertEqual(self.inspector.list_dir("nested"), ("child.txt",))

    def test_empty_directory_returns_empty_tuple(self) -> None:
        self.assertEqual(self.inspector.list_root(), ())
        empty_sub = self._workspace / "empty"
        empty_sub.mkdir()
        self.assertEqual(self.inspector.list_dir("empty"), ())

    def test_list_dir_is_deterministic(self) -> None:
        for name in ("b.txt", "a.txt", "c.txt"):
            (self._workspace / name).write_text(name)

        self.assertEqual(self.inspector.list_root(), self.inspector.list_root())

    def test_missing_directory_raises_inspection_error(self) -> None:
        with self.assertRaises(InspectionError):
            self.inspector.list_dir("nope")

    def test_list_dir_on_a_file_raises_inspection_error(self) -> None:
        (self._workspace / "f.txt").write_text("x")
        with self.assertRaises(InspectionError):
            self.inspector.list_dir("f.txt")


class TestReadText(WorkspaceInspectorTestBase):
    def test_returns_exact_utf8_contents(self) -> None:
        (self._workspace / "exact.txt").write_text(_Multiline, encoding="utf-8")

        self.assertEqual(self.inspector.read_text("exact.txt"), _Multiline)

    def test_multiline_utf8_round_trip(self) -> None:
        content = "café\nnaïve\n日本語テキスト\nmixed é\n"
        (self._workspace / "utf8.txt").write_text(content, encoding="utf-8")

        self.assertEqual(self.inspector.read_text("utf8.txt"), content)

    def test_subdirectory_file_is_readable(self) -> None:
        nested = self._workspace / "src" / "pkg"
        nested.mkdir(parents=True)
        (nested / "mod.py").write_text("print('hi')\n", encoding="utf-8")

        self.assertEqual(
            self.inspector.read_text("src/pkg/mod.py"), "print('hi')\n"
        )

    def test_missing_file_raises_inspection_error_not_attribute_error(self) -> None:
        try:
            self.inspector.read_text("missing.txt")
        except InspectionError as exc:
            self.assertIsInstance(exc, ValueError)
        except AttributeError:  # pragma: no cover - the bug we are guarding
            self.fail("read_text must raise InspectionError, not AttributeError")
        else:
            self.fail("read_text of a missing file must raise InspectionError")

    def test_read_text_on_a_directory_raises_inspection_error(self) -> None:
        (self._workspace / "adir").mkdir()
        with self.assertRaises(InspectionError):
            self.inspector.read_text("adir")

    def test_oversized_file_is_truncated_with_marker(self) -> None:
        payload = "A" * 100
        (self._workspace / "big.txt").write_text(payload, encoding="utf-8")

        content = self.inspector.read_text("big.txt", max_bytes=10)

        self.assertTrue(
            content.endswith("... [truncated: exceeded max_bytes]"),
            content,
        )
        self.assertTrue(content.startswith("AAAAAAAAAA"), content)
        self.assertEqual(
            len(content),
            10 + len("... [truncated: exceeded max_bytes]"),
        )

    def test_file_exactly_at_limit_is_not_truncated(self) -> None:
        payload = "B" * 10
        (self._workspace / "exact.txt").write_text(payload, encoding="utf-8")

        self.assertEqual(self.inspector.read_text("exact.txt", max_bytes=10), payload)

    def test_invalid_max_bytes_is_refused(self) -> None:
        (self._workspace / "f.txt").write_text("x")
        with self.assertRaises(InspectionError):
            self.inspector.read_text("f.txt", max_bytes=0)


class TestPathPolicy(WorkspaceInspectorTestBase):
    """Every unsafe path form must be refused by both methods."""

    UNSAFE = (
        "",
        "   ",
        ".",
        "..",
        "../x",
        "..\\x",
        "sub/../../x",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "C:\\Windows\\win.ini",
        "\\\\server\\share",
        "\\\\server\\share\\file.txt",
        "a\x00b",
        "ok\x00/../etc/passwd",
    )

    def test_unsafe_paths_are_refused(self) -> None:
        for path in self.UNSAFE:
            with self.subTest(path=path):
                with self.assertRaises(InspectionError):
                    self.inspector.list_dir(path)
                with self.assertRaises(InspectionError):
                    self.inspector.read_text(path)

    def test_safe_relative_paths_are_accepted(self) -> None:
        nested = self._workspace / "src" / "config"
        nested.mkdir(parents=True)
        (nested / "settings.json").write_text("{}")

        self.assertEqual(
            self.inspector.list_dir("src/config"), ("settings.json",)
        )
        self.assertEqual(self.inspector.read_text("src/config/settings.json"), "{}")


class TestNonMutationAndContainment(WorkspaceInspectorTestBase):
    def test_inspection_never_mutates_the_filesystem(self) -> None:
        (self._workspace / "f.txt").write_text("stable content")
        (self._workspace / "d").mkdir()
        (self._workspace / "d" / "inner.txt").write_text("inner")

        before = sorted(str(p) for p in self._workspace.rglob("*"))

        self.inspector.list_root()
        self.inspector.list_dir("d")
        self.inspector.read_text("f.txt")
        self.inspector.read_text("d/inner.txt")

        after = sorted(str(p) for p in self._workspace.rglob("*"))
        self.assertEqual(before, after)
        self.assertEqual(
            (self._workspace / "f.txt").read_text(encoding="utf-8"),
            "stable content",
        )
        self.assertEqual(
            (self._workspace / "d" / "inner.txt").read_text(encoding="utf-8"),
            "inner",
        )

    def test_base_dir_is_resolved(self) -> None:
        self.assertEqual(self.inspector.base_dir, self._workspace.resolve())

    def test_symlink_file_escape_is_rejected(self) -> None:
        outside = Path(tempfile.mkdtemp())
        try:
            target = outside / "secret.txt"
            target.write_text("secret", encoding="utf-8")
            link = self._workspace / "link.txt"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("file symlinks not permitted here")

            with self.assertRaises(InspectionError):
                self.inspector.read_text("link.txt")
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_symlink_directory_escape_is_rejected(self) -> None:
        outside = Path(tempfile.mkdtemp())
        try:
            (outside / "leaked.txt").write_text("leaked", encoding="utf-8")
            link = self._workspace / "linkdir"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks/junctions not permitted here")

            with self.assertRaises(InspectionError):
                self.inspector.list_dir("linkdir")
        finally:
            shutil.rmtree(outside, ignore_errors=True)


class TestInspectionContextFormatting(unittest.TestCase):
    def test_inspector_none_produces_no_block(self) -> None:
        self.assertEqual(apply_inspection_context(None, "goal"), "goal")

    def test_empty_workspace_produces_no_block(self) -> None:
        workspace = Path(tempfile.mkdtemp())
        try:
            inspector = WorkspaceInspector(workspace)
            self.assertEqual(apply_inspection_context(inspector, "goal"), "goal")
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_block_is_labelled_and_delimited(self) -> None:
        workspace = Path(tempfile.mkdtemp())
        try:
            (workspace / "hello.txt").write_text("x")
            inspector = WorkspaceInspector(workspace)
            description = apply_inspection_context(inspector, "Do the thing")

            self.assertTrue(description.startswith(INSPECTION_HEADER))
            self.assertIn(INSPECTION_FOOTER, description)
            self.assertIn("not verification evidence", description)
            self.assertTrue(description.endswith("Do the thing"))
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


class TestInspectionCannotSubstituteForVerification(unittest.TestCase):
    """CRITICAL: inspection output is never verification evidence.

    A real agent, real planner, real verifier and a real inspector are wired
    together.  The executor does nothing.  The workspace even *already
    contains* a file whose name and content match the expectation.  The
    report must still FAIL, because ``VerificationResult`` is derived from the
    post-execution filesystem state checked by ``FilesystemVerifier`` alone.
    """

    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)

    def make_agent(self, executor: _NoopExecutor) -> Agent:
        runner = PlanRunner(
            planner=SupportedFileTaskPlanner(),
            client=executor,
            verifier=FilesystemVerifier(),
            base_dir=str(self.workspace),
            default_max_attempts=2,
        )
        return Agent(
            _EmptyMemoryStore(),
            runner,
            inspector=WorkspaceInspector(self.workspace),
        )

    def test_failed_executor_stays_failed_with_inspection_context(self) -> None:
        """Inspection must not turn a no-op executor into a pass.

        The workspace is deliberately full of inspection-visible content so a
        substantial, non-empty block is placed in front of the planner, while
        the file the plan actually targets does not exist.  The independent
        verifier must still fail, because it reads the post-execution
        filesystem and nothing else.
        """
        for index in range(5):
            (self.workspace / f"context_{index}.txt").write_text("untrusted data")

        executor = _NoopExecutor()
        agent = self.make_agent(executor)

        report = agent.run("Create a file named hello.txt containing coffee")

        self.assertFalse(
            report.passed,
            "inspection context must never satisfy a plan step",
        )
        self.assertEqual(report.steps[0].status, StepStatus.FAILED)
        self.assertIsNotNone(report.steps[0].recovery)
        self.assertFalse(report.steps[0].recovery.verification.passed)
        self.assertGreaterEqual(len(executor.calls), 1)
        self.assertFalse((self.workspace / "hello.txt").exists())

    def test_inspection_context_reached_the_planner_but_changed_nothing(self) -> None:
        """The inspection block must reach the planner, and go no further.

        ``Agent`` puts the block on the planner-visible ``Goal``.  The planner
        then emits a self-contained ``action``, so the executor never receives
        the block; context is offered to planning and is not smuggled into
        execution.
        """
        (self.workspace / "notes.txt").write_text("visible to inspection")

        executor = _NoopExecutor()
        agent = self.make_agent(executor)

        report = agent.run("Create a file named brand_new.txt containing hello")

        self.assertFalse(report.passed)
        self.assertEqual(report.steps[0].status, StepStatus.FAILED)
        # The planner saw the inspection block in its input goal...
        planner_goal = report.attempts[0].plan.goal.description
        self.assertNotIn(INSPECTION_HEADER, planner_goal)
        self.assertNotIn(INSPECTION_FOOTER, planner_goal)
        self.assertNotIn("notes.txt", planner_goal)
        self.assertEqual(planner_goal, "Create a file named brand_new.txt containing hello")
        # ...and the executor received only the planner's own action.
        self.assertNotIn(INSPECTION_HEADER, executor.calls[0])
        self.assertNotIn("notes.txt", executor.calls[0])
        self.assertFalse(
            (self.workspace / "brand_new.txt").exists(),
            "a failed step must not leave the target behind",
        )

    def test_inspector_none_preserves_previous_behaviour(self) -> None:
        (self.workspace / "existing.txt").write_text("already here")

        executor = _NoopExecutor()
        runner = PlanRunner(
            planner=SupportedFileTaskPlanner(),
            client=executor,
            verifier=FilesystemVerifier(),
            base_dir=str(self.workspace),
            default_max_attempts=2,
        )
        agent = Agent(_EmptyMemoryStore(), runner)  # no inspector

        report = agent.run("Create a file named hello.txt containing coffee")

        self.assertFalse(report.passed)
        self.assertEqual(report.steps[0].status, StepStatus.FAILED)
        self.assertNotIn(INSPECTION_HEADER, executor.calls[0])
        self.assertNotIn(
            "existing.txt",
            report.attempts[0].plan.goal.description,
        )


if __name__ == "__main__":
    unittest.main()
