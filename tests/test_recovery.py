"""Tests for AIL recovery orchestration (Milestone 2E).

Exercises the bounded execute → verify → recover loop on top of the existing
``execute_and_verify``.  The fake executor manipulates the real filesystem on
a chosen attempt so that the independent ``FilesystemVerifier`` decides the
outcome — the executor's response text is never trusted.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from core.actions import execute_and_verify  # noqa: E402
from core.recovery import DefaultRetry, execute_with_recovery  # noqa: E402
from interfaces.recovery import RecoveryResult, RecoveryStrategy  # noqa: E402
from interfaces.verification import CheckResult, VerificationResult  # noqa: E402
from core.verification import FileExpectation, FilesystemVerifier  # noqa: E402

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


class MessageRecordingExecutor:
    """Always fails and records every message it is handed."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def send_message(
        self, message: str, thread_id: str | None = None, timeout: float | None = None
    ) -> str:
        self.messages.append(message)
        return "fake executor response text"


class ExplodingExecutor:
    """Raises immediately on every call; recovery must not retry it."""

    def __init__(self) -> None:
        self.calls = 0

    def send_message(
        self, message: str, thread_id: str | None = None, timeout: float | None = None
    ) -> str:
        self.calls += 1
        raise RuntimeError("executor exploded")


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

    def test_repeated_recovery_messages_do_not_compound(self) -> None:
        executor = MessageRecordingExecutor()

        result = execute_with_recovery(
            executor,
            self.make_verifier(),
            "create the file",
            self.expectations(),
            base_dir=self.temp_dir,
            max_attempts=4,
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.attempts, 4)
        original, *corrected = executor.messages
        self.assertEqual(original, "create the file")
        # Every recovery message is built from the *original* action, so the
        # prompt cannot grow by accumulating prior corrective text.
        self.assertEqual(len(set(corrected)), 1)
        for message in corrected:
            self.assertIn(original, message)
            self.assertEqual(
                message.count("The previous attempt did not satisfy"), 1
            )
            self.assertNotIn(str(self.temp_dir), message)

    def test_larger_max_attempts_is_honored(self) -> None:
        executor = FakeExecutor(self.temp_dir / TARGET, succeed_on=5)

        result = execute_with_recovery(
            executor,
            self.make_verifier(),
            "create the file",
            self.expectations(),
            base_dir=self.temp_dir,
            max_attempts=5,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.attempts, 5)
        self.assertEqual(executor.calls, 5)
        self.assertEqual(len(result.history), 5)
        self.assertEqual(result.reason, "PASS on attempt 5")

    def test_never_passing_executor_stops_at_configured_limit(self) -> None:
        executor = FakeExecutor(self.temp_dir / TARGET, succeed_on=99)

        result = execute_with_recovery(
            executor,
            self.make_verifier(),
            "create the file",
            self.expectations(),
            base_dir=self.temp_dir,
            max_attempts=7,
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.attempts, 7)
        self.assertEqual(executor.calls, 7)
        self.assertEqual(len(result.history), 7)
        self.assertIn("FINAL FAIL after 7 attempts", result.reason)

    def test_executor_exception_terminates_recovery_without_retry(self) -> None:
        """Executor crashes are not retried; they end the recovery path."""
        executor = ExplodingExecutor()

        with self.assertRaisesRegex(RuntimeError, "executor exploded"):
            execute_with_recovery(
                executor,
                self.make_verifier(),
                "create the file",
                self.expectations(),
                base_dir=self.temp_dir,
                max_attempts=3,
            )

        self.assertEqual(executor.calls, 1)

    def test_recovery_message_reaches_executor_on_retry(self) -> None:
        executor = MessageRecordingExecutor()

        execute_with_recovery(
            executor,
            self.make_verifier(),
            "create the file",
            self.expectations(),
            base_dir=self.temp_dir,
            max_attempts=2,
        )

        self.assertEqual(len(executor.messages), 2)
        retry_message = executor.messages[1]
        self.assertIn("create the file", retry_message)
        self.assertIn(TARGET, retry_message)
        self.assertNotIn(str(self.temp_dir), retry_message)


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
        self.assertIn(TARGET, message)
        self.assertIn("expected exists", message)
        self.assertIn("got missing", message)
        for token in ("delete", "rm ", "rmdir", "del ", "Remove-Item", "format "):
            self.assertNotIn(token, message)


class TestDefaultRetryHidesInternalPaths(RecoveryTestBase):
    """A recovery message must never disclose the internal filesystem layout.

    ``CheckResult.name`` is the absolute path the verifier stat'd.  That path
    carries AIL's own workspace location and any surrounding temp/profile
    directories, none of which the executor needs in order to correct a failed
    check, so only the final path component is rendered.
    """

    def message_for(self, *names: str) -> str:
        checks = tuple(
            CheckResult(
                name=name,
                passed=False,
                expected="exists",
                actual="missing",
            )
            for name in names
        )
        result = VerificationResult(
            passed=False,
            reason=f"{len(checks)} of {len(checks)} checks failed",
            checks=checks,
        )
        message = DefaultRetry().next_message("create the file", result, 1)
        self.assertIsNotNone(message)
        return message

    def test_absolute_base_dir_is_not_leaked(self) -> None:
        message = self.message_for(str(self.temp_dir / TARGET))

        self.assertNotIn(str(self.temp_dir), message)
        self.assertNotIn(str(self.temp_dir / TARGET), message)
        self.assertNotIn(str(self.temp_dir).rsplit("\\", 1)[-1], message)

    def test_relative_filename_is_still_present(self) -> None:
        message = self.message_for(str(self.temp_dir / TARGET))

        self.assertIn(TARGET, message)
        self.assertIn("out.txt (expected exists, got missing)", message)

    def test_nested_path_renders_only_its_final_component(self) -> None:
        nested = self.temp_dir / "src" / "config" / "settings.json"

        message = self.message_for(str(nested))

        self.assertIn("settings.json", message)
        self.assertNotIn(str(nested), message)
        self.assertNotIn(str(self.temp_dir), message)

    def test_multiple_failed_checks_remain_understandable(self) -> None:
        message = self.message_for(
            str(self.temp_dir / "alpha.txt"),
            str(self.temp_dir / "sub" / "beta.txt"),
        )

        self.assertIn("alpha.txt (expected exists, got missing)", message)
        self.assertIn("beta.txt (expected exists, got missing)", message)
        self.assertIn(";", message)  # both checks still listed, not collapsed
        self.assertNotIn(str(self.temp_dir), message)

    def test_passing_check_is_not_reported_as_failed(self) -> None:
        failed = CheckResult(
            name=str(self.temp_dir / "alpha.txt"),
            passed=False,
            expected="exists",
            actual="missing",
        )
        passed = CheckResult(
            name=str(self.temp_dir / "beta.txt"),
            passed=True,
            expected="exists",
            actual="exists",
        )
        result = VerificationResult(
            passed=False,
            reason="1 of 2 checks failed",
            checks=(failed, passed),
        )

        message = DefaultRetry().next_message("create the files", result, 1)

        self.assertIsNotNone(message)
        self.assertIn("alpha.txt", message)
        self.assertNotIn("beta.txt", message)


if __name__ == "__main__":
    unittest.main()