"""Tests for AIL recovery orchestration (Milestone 2E).

Exercises the bounded execute → verify → recover loop on top of the existing
``execute_and_verify``.  The fake executor manipulates the real filesystem on
a chosen attempt so that the independent ``FilesystemVerifier`` decides the
outcome — the executor's response text is never trusted.
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
from core.recovery import DefaultRetry, execute_with_recovery  # noqa: E402
from interfaces.recovery import RecoveryResult, RecoveryStrategy  # noqa: E402
from interfaces.verification import CheckResult, VerificationResult  # noqa: E402
from tools.fs_verifier import FileExpectation, FilesystemVerifier  # noqa: E402

EXPECTED_TEXT = "recovered content"
TARGET = "out.txt"


class FakeExecutor:
    """Simulates the executor writing the real file from the Nth call on."""

    def __init__(self, target: Path, succeed_on: int = 1) -> None:
        self.target = Path(target)
        self.calls = 0
        self.succeed_on = succeed_on
        self.timeouts: list[float | None] = []

    def send_message(
        self, message: str, thread_id: str | None = None, timeout: float | None = None
    ) -> str:
        self.calls += 1
        self.timeouts.append(timeout)
        if self.calls >= self.succeed_on:
            self.target.write_text(EXPECTED_TEXT)
        return "fake executor response text"


class RecordingStrategy(RecoveryStrategy):
    """Records what the loop hands it and always suggests a corrective retry."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, VerificationResult, int]] = []

    def next_message(
        self, original: str, result: VerificationResult, attempt: int
    ) -> str:
        self.calls.append((original, result, attempt))
        return "Please correct the failed checks."


class StopStrategy(RecoveryStrategy):
    """Gives up immediately after the first failure."""

    def next_message(
        self, original: str, result: VerificationResult, attempt: int
    ) -> None:
        return None


class RecoveryTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.temp_dir = Path(self._tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp)

    def make_verifier(self) -> FilesystemVerifier:
        return FilesystemVerifier()

    def expectations(self) -> list[FileExpectation]:
        return [FileExpectation(TARGET, exists=True, contains=EXPECTED_TEXT)]


class TestExecuteWithRecovery(RecoveryTestBase):
    def test_execute_and_verify_forwards_timeout(self) -> None:
        executor = FakeExecutor(self.temp_dir / TARGET)

        _, result = execute_and_verify(
            executor,
            self.make_verifier(),
            "create the file",
            self.expectations(),
            base_dir=self.temp_dir,
            timeout=7.5,
        )

        self.assertTrue(result.passed)
        self.assertEqual(executor.timeouts, [7.5])

    def test_recovery_forwards_timeout_on_every_attempt(self) -> None:
        executor = FakeExecutor(self.temp_dir / TARGET, succeed_on=2)

        result = execute_with_recovery(
            executor,
            self.make_verifier(),
            "create the file",
            self.expectations(),
            base_dir=self.temp_dir,
            timeout=7.5,
        )

        self.assertTrue(result.passed)
        self.assertEqual(executor.timeouts, [7.5, 7.5])

    def test_pass_on_first_attempt_is_single_attempt(self) -> None:
        executor = FakeExecutor(self.temp_dir / TARGET, succeed_on=1)
        result = execute_with_recovery(
            executor,
            self.make_verifier(),
            "create the file",
            self.expectations(),
            base_dir=self.temp_dir,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(executor.calls, 1)
        self.assertEqual(len(result.history), 1)
        self.assertTrue(result.history[0].passed)
        self.assertEqual(result.reason, "PASS on attempt 1")

    def test_fail_then_pass_takes_two_attempts(self) -> None:
        executor = FakeExecutor(self.temp_dir / TARGET, succeed_on=2)
        result = execute_with_recovery(
            executor,
            self.make_verifier(),
            "create the file",
            self.expectations(),
            base_dir=self.temp_dir,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(executor.calls, 2)
        self.assertFalse(result.history[0].passed)
        self.assertTrue(result.history[1].passed)

    def test_always_fail_is_bounded_at_two(self) -> None:
        executor = FakeExecutor(self.temp_dir / TARGET, succeed_on=99)
        result = execute_with_recovery(
            executor,
            self.make_verifier(),
            "create the file",
            self.expectations(),
            base_dir=self.temp_dir,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(executor.calls, 2)
        self.assertIn("FINAL FAIL after 2 attempts", result.reason)
        self.assertTrue(all(not v.passed for v in result.history))

    def test_strategy_receives_failed_verification_result(self) -> None:
        executor = FakeExecutor(self.temp_dir / TARGET, succeed_on=2)
        strategy = RecordingStrategy()
        result = execute_with_recovery(
            executor,
            self.make_verifier(),
            "create the file",
            self.expectations(),
            base_dir=self.temp_dir,
            strategy=strategy,
        )
        self.assertTrue(result.passed)
        self.assertEqual(len(strategy.calls), 1)
        original, failed_result, attempt = strategy.calls[0]
        self.assertEqual(original, "create the file")
        self.assertIsInstance(failed_result, VerificationResult)
        self.assertFalse(failed_result.passed)
        self.assertEqual(attempt, 1)
        failed_names = [c.name for c in failed_result.checks if not c.passed]
        self.assertTrue(any(TARGET in str(n) for n in failed_names))

    def test_strategy_returning_none_stops_recovery_early(self) -> None:
        executor = FakeExecutor(self.temp_dir / TARGET, succeed_on=99)
        result = execute_with_recovery(
            executor,
            self.make_verifier(),
            "create the file",
            self.expectations(),
            base_dir=self.temp_dir,
            strategy=StopStrategy(),
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(executor.calls, 1)
        self.assertIn("FINAL FAIL after 1 attempt", result.reason)

    def test_max_attempts_one_performs_single_attempt(self) -> None:
        executor = FakeExecutor(self.temp_dir / TARGET, succeed_on=2)
        result = execute_with_recovery(
            executor,
            self.make_verifier(),
            "create the file",
            self.expectations(),
            base_dir=self.temp_dir,
            max_attempts=1,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(executor.calls, 1)
        self.assertIn("FINAL FAIL after 1 attempt", result.reason)

    def test_history_contains_each_verification_result(self) -> None:
        executor = FakeExecutor(self.temp_dir / TARGET, succeed_on=2)
        result = execute_with_recovery(
            executor,
            self.make_verifier(),
            "create the file",
            self.expectations(),
            base_dir=self.temp_dir,
        )
        self.assertEqual(len(result.history), 2)
        self.assertIs(result.verification, result.history[1])
        self.assertEqual(result.history[0].passed, False)
        self.assertEqual(result.history[1].passed, True)


class TestDefaultRetry(RecoveryTestBase):
    def make_failed_result(self) -> VerificationResult:
        check = CheckResult(
            name=str(self.temp_dir / TARGET),
            passed=False,
            expected="exists",
            actual="missing",
        )
        return VerificationResult(
            passed=False,
            reason="1 of 1 checks failed",
            checks=(check,),
        )

    def test_returns_none_when_result_passed(self) -> None:
        passed = VerificationResult(
            passed=True, reason="All checks passed"
        )
        self.assertIsNone(DefaultRetry().next_message("create the file", passed, 1))

    def test_does_not_introduce_destructive_commands(self) -> None:
        message = DefaultRetry().next_message(
            "create the file", self.make_failed_result(), 1
        )
        self.assertIsNotNone(message)
        self.assertIn("create the file", message)
        self.assertIn(str(self.temp_dir / TARGET), message)
        self.assertIn("expected exists", message)
        self.assertIn("got missing", message)
        for token in ("delete", "rm ", "rmdir", "del ", "Remove-Item", "format "):
            self.assertNotIn(token, message)


if __name__ == "__main__":
    unittest.main()