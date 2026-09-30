"""Tests for AIL filesystem verification (Milestone 2D).

Covers the pure ``FilesystemVerifier`` and the thin ``execute_and_verify``
composition.  Verification must be executor-independent: a fake client's
response text is ignored for verification purposes.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure AIL root is importable.
AIL_ROOT = str(Path(__file__).resolve().parents[1])
if AIL_ROOT not in sys.path:
    sys.path.insert(0, AIL_ROOT)

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

    def test_read_error_becomes_failed_check(self) -> None:
        (self.temp_dir / "adir").mkdir()
        result = self.make_verifier().verify(
            [FileExpectation("adir", exists=True, contains="x")],
            self.temp_dir,
        )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.checks), 1)
        check = result.checks[0]
        self.assertIsNotNone(check.error)
        self.assertEqual(check.actual, "read error")


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