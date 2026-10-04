"""Tests for AIL filesystem verification (Milestone 2D).

Covers the pure ``FilesystemVerifier`` and the thin ``execute_and_verify``
composition.  Verification must be executor-independent: a fake client's
response text is ignored for verification purposes.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from core.actions import execute_and_verify  # noqa: E402
from interfaces.verification import VerificationResult  # noqa: E402
from core.verification import FileExpectation, FilesystemVerifier  # noqa: E402


class FakeClient:
    """Records calls and returns a canned response regardless of message."""

    def __init__(self, response: object = "fake response text") -> None:
        self.response = response
        self.calls: list[tuple[str, str | None]] = []

    def send_message(
        self, message: str, thread_id: str | None = None, timeout: float | None = None
    ) -> object:
        self.calls.append((message, thread_id))
        return self.response


class FilesystemVerifierTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.temp_dir = Path(self._tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp)

    def make_verifier(self) -> FilesystemVerifier:
        return FilesystemVerifier()


class TestFilesystemVerifier(FilesystemVerifierTestBase):
    def test_existing_file_passes(self) -> None:
        (self.temp_dir / "a.txt").write_text("data")
        result = self.make_verifier().verify(
            [FileExpectation("a.txt")], self.temp_dir
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "All checks passed")
        self.assertEqual(len(result.checks), 1)
        self.assertTrue(result.checks[0].passed)

    def test_missing_expected_file_fails(self) -> None:
        result = self.make_verifier().verify(
            [FileExpectation("missing.txt")], self.temp_dir
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.checks[0].expected, "exists")
        self.assertEqual(result.checks[0].actual, "missing")

    def test_expected_nonexistent_passes(self) -> None:
        result = self.make_verifier().verify(
            [FileExpectation("missing.txt", exists=False)], self.temp_dir
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.checks[0].expected, "not exists")
        self.assertEqual(result.checks[0].actual, "not exists")

    def test_contains_matching_text_passes(self) -> None:
        (self.temp_dir / "note.txt").write_text("The user likes coffee.")
        result = self.make_verifier().verify(
            [FileExpectation("note.txt", exists=True, contains="coffee")],
            self.temp_dir,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.checks[0].expected, "contains 'coffee'")
        self.assertEqual(result.checks[0].actual, "contains")

    def test_contains_wrong_text_fails(self) -> None:
        (self.temp_dir / "note.txt").write_text("The user likes coffee.")
        result = self.make_verifier().verify(
            [FileExpectation("note.txt", exists=True, contains="tea")],
            self.temp_dir,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.checks[0].actual, "does not contain 'tea'")

    def test_multiple_expectations_aggregate(self) -> None:
        (self.temp_dir / "a.txt").write_text("aaa")
        result = self.make_verifier().verify(
            [
                FileExpectation("a.txt", exists=True),
                FileExpectation("b.txt", exists=True),
            ],
            self.temp_dir,
        )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.checks), 2)
        self.assertTrue(result.checks[0].passed)
        self.assertFalse(result.checks[1].passed)
        self.assertIn("1 of 2 checks failed", result.reason)
        self.assertIn(str(self.temp_dir / "b.txt"), result.reason)

    def test_does_not_mutate_filesystem(self) -> None:
        path = self.temp_dir / "note.txt"
        path.write_text("content")
        before = sorted(str(p) for p in self.temp_dir.rglob("*"))
        self.make_verifier().verify(
            [FileExpectation("note.txt", exists=True, contains="content")],
            self.temp_dir,
        )
        after = sorted(str(p) for p in self.temp_dir.rglob("*"))
        self.assertEqual(before, after)
        self.assertEqual(path.read_text(encoding="utf-8"), "content")

class TestSecurityContainment(FilesystemVerifierTestBase):
    """The verifier itself enforces the workspace boundary.

    Every case here calls ``FilesystemVerifier.verify`` directly; none of them
    depend on planner validation, so a planner or executor that supplies an
    unsafe expectation cannot make the verifier read or assert anything outside
    the configured workspace.
    """

    def _workspace(self) -> Path:
        workspace = self.temp_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _try_symlink(self, link: Path, target: Path, *, target_is_directory: bool) -> bool:
        """Attempt to create a symlink; return False if the platform forbids it."""
        try:
            link.symlink_to(target, target_is_directory=target_is_directory)
            return True
        except (OSError, NotImplementedError):
            return False

    def test_traversal_rejected(self) -> None:
        workspace = self._workspace()
        (self.temp_dir / "outside.txt").write_text("secret")
        exp = FileExpectation(path="../outside.txt", exists=True)
        result = self.make_verifier().verify([exp], workspace)
        self.assertFalse(result.passed)
        self.assertIn("Security violation", result.checks[0].error)

    def test_deep_traversal_rejected(self) -> None:
        workspace = self._workspace()
        (self.temp_dir / "outside.txt").write_text("secret")
        exp = FileExpectation(path="../../outside.txt", exists=True)
        result = self.make_verifier().verify([exp], workspace)
        self.assertFalse(result.passed)
        self.assertIn("Security violation", result.checks[0].error)

    def test_nested_safe_path_accepted(self) -> None:
        workspace = self._workspace()
        (workspace / "sub" / "deeper").mkdir(parents=True)
        (workspace / "sub" / "deeper" / "a.txt").write_text("data")
        exp = FileExpectation(path="sub/deeper/a.txt", exists=True, contains="data")
        result = self.make_verifier().verify([exp], workspace)
        self.assertTrue(result.passed)
        self.assertIsNone(result.checks[0].error)

    def test_dot_dot_path_rejected(self) -> None:
        workspace = self._workspace()
        exp = FileExpectation(path="..", exists=True)
        result = self.make_verifier().verify([exp], workspace)
        self.assertFalse(result.passed)
        self.assertIn("Security violation", result.checks[0].error)

    def test_dot_path_is_contained_not_an_escape(self) -> None:
        """``.`` resolves to the base itself, so containment allows it.

        This is not an escape: the expectation is checked against the workspace
        directory, which is inside the workspace.  The test pins that ``.`` is
        *not* reported as a security violation, while still not passing
        silently when the base is missing.
        """
        workspace = self._workspace()
        exp = FileExpectation(path=".", exists=True)
        result = self.make_verifier().verify([exp], workspace)
        self.assertTrue(result.passed)
        self.assertIsNone(result.checks[0].error)
        self.assertEqual(result.checks[0].actual, "exists")

        missing_base = self.temp_dir / "nope"
        result_missing = self.make_verifier().verify(
            [FileExpectation(path=".", exists=True)], missing_base
        )
        self.assertFalse(result_missing.passed)
        self.assertIsNone(result_missing.checks[0].error)
        self.assertEqual(result_missing.checks[0].actual, "missing")

    @unittest.skipUnless(os.name == "nt", "Windows drive/UNC semantics")
    def test_windows_drive_path_rejected(self) -> None:
        workspace = self._workspace()
        for drive_path in ("C:/Windows/win.ini", "D:/data"):
            with self.subTest(path=drive_path):
                exp = FileExpectation(path=drive_path, exists=True)
                result = self.make_verifier().verify([exp], workspace)
                self.assertFalse(result.passed)
                self.assertIn("Security violation", result.checks[0].error)

    @unittest.skipUnless(os.name == "nt", "Windows UNC semantics")
    def test_windows_unc_path_rejected(self) -> None:
        workspace = self._workspace()
        exp = FileExpectation(path="//server/share/file.txt", exists=True)
        result = self.make_verifier().verify([exp], workspace)
        self.assertFalse(result.passed)
        self.assertIn("Security violation", result.checks[0].error)

    @unittest.skipUnless(os.name == "nt", "POSIX absolute semantics")
    def test_posix_absolute_path_rejected(self) -> None:
        workspace = self._workspace()
        exp = FileExpectation(path="/etc/passwd", exists=True)
        result = self.make_verifier().verify([exp], workspace)
        self.assertFalse(result.passed)
        self.assertIn("Security violation", result.checks[0].error)

    def test_sibling_prefix_escape_rejected(self) -> None:
        """A sibling directory sharing the base's name prefix must not match.

        A string-prefix containment bug would treat ``workspace-evil`` as
        inside ``workspace``; the component-aware check must reject it, whether
        the expectation reaches it by traversal or by absolute path.
        """
        workspace = self._workspace()
        sibling = self.temp_dir / "workspace-evil"
        sibling.mkdir()
        (sibling / "file.txt").write_text("secret")

        for path in ("../workspace-evil/file.txt", sibling / "file.txt"):
            with self.subTest(path=str(path)):
                exp = FileExpectation(path=path, exists=True)
                result = self.make_verifier().verify([exp], workspace)
                self.assertFalse(result.passed)
                self.assertIn("Security violation", result.checks[0].error)

    def test_symlink_file_escape_rejected(self) -> None:
        workspace = self._workspace()
        outside = self.temp_dir / "outside.txt"
        outside.write_text("secret")
        link = workspace / "link.txt"
        if not self._try_symlink(link, outside, target_is_directory=False):
            self.skipTest("platform forbids file symlinks")

        exp = FileExpectation(path="link.txt", exists=True)
        result = self.make_verifier().verify([exp], workspace)
        self.assertFalse(result.passed)
        self.assertIn("Security violation", result.checks[0].error)

    def test_symlink_directory_escape_rejected(self) -> None:
        workspace = self._workspace()
        outside_dir = self.temp_dir / "outside_dir"
        outside_dir.mkdir()
        (outside_dir / "secret.txt").write_text("secret")
        link = workspace / "linkdir"
        if not self._try_symlink(link, outside_dir, target_is_directory=True):
            self.skipTest("platform forbids directory symlinks/junctions")

        exp = FileExpectation(path="linkdir/secret.txt", exists=True)
        result = self.make_verifier().verify([exp], workspace)
        self.assertFalse(result.passed)
        self.assertIn("Security violation", result.checks[0].error)

    def test_symlink_inside_workspace_remains_valid(self) -> None:
        workspace = self._workspace()
        real = workspace / "real.txt"
        real.write_text("inside content")
        link = workspace / "alias.txt"
        if not self._try_symlink(link, real, target_is_directory=False):
            self.skipTest("platform forbids file symlinks")

        exp = FileExpectation(path="alias.txt", exists=True, contains="inside content")
        result = self.make_verifier().verify([exp], workspace)
        self.assertTrue(result.passed, result.reason)
        self.assertIsNone(result.checks[0].error)

    def test_symlink_directory_inside_workspace_remains_valid(self) -> None:
        workspace = self._workspace()
        real_dir = workspace / "real_dir"
        real_dir.mkdir()
        (real_dir / "data.txt").write_text("nested")
        link = workspace / "alias_dir"
        if not self._try_symlink(link, real_dir, target_is_directory=True):
            self.skipTest("platform forbids directory symlinks/junctions")

        exp = FileExpectation(path="alias_dir/data.txt", exists=True, contains="nested")
        result = self.make_verifier().verify([exp], workspace)
        self.assertTrue(result.passed, result.reason)

    def test_unsafe_expectation_never_reads_outside_file(self) -> None:
        """The boundary holds even when the outside file exists and is readable."""
        workspace = self._workspace()
        outside = self.temp_dir / "outside.txt"
        outside.write_text("should never be read")
        exp = FileExpectation(path="../outside.txt", exists=True, contains="should never be read")
        result = self.make_verifier().verify([exp], workspace)
        self.assertFalse(result.passed)
        self.assertIn("Security violation", result.checks[0].error)

    def test_unsafe_expectation_with_exists_false_is_still_rejected(self) -> None:
        """Containment is checked regardless of exists/absent expectation."""
        workspace = self._workspace()
        exp = FileExpectation(path="../outside.txt", exists=False)
        result = self.make_verifier().verify([exp], workspace)
        self.assertFalse(result.passed)
        self.assertIn("Security violation", result.checks[0].error)

    def test_base_is_resolved_before_containment(self) -> None:
        """A base passed as a relative or non-canonical path still confines."""
        workspace = self._workspace()
        (workspace / "a.txt").write_text("data")
        exp = FileExpectation(path="a.txt", exists=True)

        # Pass the base through a "." component to force resolution.
        result = self.make_verifier().verify([exp], workspace / ".")
        self.assertTrue(result.passed)


class TestExecuteAndVerify(FilesystemVerifierTestBase):
    def test_uses_fake_client(self) -> None:
        client = FakeClient()
        execute_and_verify(
            client,
            self.make_verifier(),
            "create the file",
            [FileExpectation("out.txt", exists=False)],
            base_dir=self.temp_dir,
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][0], "create the file")
        self.assertIsNone(client.calls[0][1])

    def test_response_text_does_not_affect_verification(self) -> None:
        (self.temp_dir / "out.txt").write_text("created")
        client = FakeClient(response="COMPLETELY UNRELATED TEXT")
        result = execute_and_verify(
            client,
            self.make_verifier(),
            "create the file",
            [FileExpectation("out.txt", exists=True, contains="created")],
            base_dir=self.temp_dir,
        )[1]
        self.assertTrue(result.passed)

    def test_returns_both_response_and_result(self) -> None:
        (self.temp_dir / "out.txt").write_text("created")
        response = "fake response object"
        client = FakeClient(response=response)
        returned_response, result = execute_and_verify(
            client,
            self.make_verifier(),
            "create the file",
            [FileExpectation("out.txt", exists=True, contains="created")],
            base_dir=self.temp_dir,
        )
        self.assertIs(returned_response, response)
        self.assertIsInstance(result, VerificationResult)
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()