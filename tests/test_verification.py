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
from core.verification import (  # noqa: E402
    MAX_VERIFY_READ_BYTES,
    FileExpectation,
    FilesystemVerifier,
)


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


class TestNonVacuousVerification(FilesystemVerifierTestBase):
    """``passed=True`` requires at least one check that actually ran (2F.3).

    The verifier refuses to certify success it has no evidence for.  These
    cases call ``FilesystemVerifier.verify`` directly, with no planner, runner
    or agent in the loop: a plan that declares nothing to check, or an
    expectation the verifier cannot consume, must fail at this boundary rather
    than passing vacuously or crashing out of ``verify``.

    Assertions about unsupported expectations are deliberately type-oriented.
    They check the *type name* the verifier reports, never the rendered value,
    so they stay valid for any object whose ``str`` is arbitrary.
    """

    def test_empty_list_fails(self) -> None:
        result = self.make_verifier().verify([], self.temp_dir)
        self.assertFalse(result.passed)
        self.assertEqual(result.checks, ())

    def test_empty_tuple_fails(self) -> None:
        result = self.make_verifier().verify((), self.temp_dir)
        self.assertFalse(result.passed)
        self.assertEqual(result.checks, ())

    def test_empty_generator_fails(self) -> None:
        result = self.make_verifier().verify(iter(()), self.temp_dir)
        self.assertFalse(result.passed)
        self.assertEqual(result.checks, ())

    def test_empty_expectations_never_report_all_checks_passed(self) -> None:
        """``all([])`` is True; the empty set must not inherit that verdict."""
        for empty in ([], (), iter(()), set()):
            with self.subTest(kind=type(empty).__name__):
                result = self.make_verifier().verify(empty, self.temp_dir)
                self.assertFalse(result.passed)
                self.assertNotEqual(result.reason, "All checks passed")
                self.assertIn("no checks were performed", result.reason.lower())

    def test_empty_expectations_do_not_raise(self) -> None:
        # An empty set is a controlled failure, not an error condition.
        self.make_verifier().verify([], self.temp_dir)
        self.make_verifier().verify((), self.temp_dir)
        self.make_verifier().verify(iter(()), self.temp_dir)

    def test_empty_expectations_still_touch_no_file(self) -> None:
        before = sorted(str(p) for p in self.temp_dir.rglob("*"))
        self.make_verifier().verify([], self.temp_dir)
        after = sorted(str(p) for p in self.temp_dir.rglob("*"))
        self.assertEqual(before, after)

    def test_unsupported_string_expectation_fails_without_raising(self) -> None:
        """A ``str`` has no ``.path``; it must not raise ``AttributeError``."""
        result = self.make_verifier().verify(["workspace-prepared"], self.temp_dir)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.checks), 1)
        check = result.checks[0]
        self.assertFalse(check.passed)
        self.assertIn("str", check.actual)
        self.assertIn("str", check.name)

    def test_unsupported_object_expectation_fails_without_raising(self) -> None:
        class SomeExpectation:
            pass

        result = self.make_verifier().verify([SomeExpectation()], self.temp_dir)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.checks), 1)
        self.assertFalse(result.checks[0].passed)
        self.assertIn("SomeExpectation", result.checks[0].actual)

    def test_bare_object_expectation_fails_without_raising(self) -> None:
        result = self.make_verifier().verify([object()], self.temp_dir)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.checks), 1)
        self.assertFalse(result.checks[0].passed)
        self.assertIn("object", result.checks[0].actual)

    def test_none_expectation_fails_without_raising(self) -> None:
        result = self.make_verifier().verify([None], self.temp_dir)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.checks), 1)
        self.assertFalse(result.checks[0].passed)
        self.assertIn("NoneType", result.checks[0].actual)

    def test_unsupported_expectation_reports_type_not_value(self) -> None:
        """The failure must not render the untrusted value itself.

        ``core.recovery`` feeds failed-check detail to the executor, so an
        arbitrary object's ``str`` must never be copied into the check.
        """
        secret = object()
        result = self.make_verifier().verify([secret], self.temp_dir)
        check = result.checks[0]
        rendered = " ".join(
            part for part in (check.name, check.expected, check.actual, check.error or "")
        )
        self.assertNotIn(str(secret), rendered)
        self.assertIn("object", rendered)

    def test_mixed_valid_and_invalid_expectations_produce_two_checks(self) -> None:
        (self.temp_dir / "valid.txt").write_text("content")

        result = self.make_verifier().verify(
            [FileExpectation("valid.txt"), "junk"],
            self.temp_dir,
        )

        self.assertFalse(result.passed)
        self.assertEqual(len(result.checks), 2)
        self.assertTrue(result.checks[0].passed, "the valid expectation is checked")
        self.assertFalse(result.checks[1].passed, "the invalid one fails")
        self.assertIn("1 of 2 checks failed", result.reason)

    def test_invalid_expectation_does_not_abort_later_valid_expectations(self) -> None:
        """An unsupported item must not stop the checks that follow it."""
        (self.temp_dir / "first.txt").write_text("content")
        # "second.txt" is never created, so its own check must genuinely fail.
        result = self.make_verifier().verify(
            [
                FileExpectation("first.txt"),
                "junk",
                FileExpectation("second.txt"),
            ],
            self.temp_dir,
        )

        self.assertFalse(result.passed)
        self.assertEqual(len(result.checks), 3)
        self.assertTrue(result.checks[0].passed)
        self.assertFalse(result.checks[1].passed)
        self.assertIn("str", result.checks[1].actual)
        self.assertFalse(result.checks[2].passed)
        self.assertEqual(result.checks[2].actual, "missing")
        self.assertIn("2 of 3 checks failed", result.reason)

    def test_leading_invalid_expectation_does_not_abort_the_rest(self) -> None:
        (self.temp_dir / "only.txt").write_text("content")
        result = self.make_verifier().verify(
            [None, FileExpectation("only.txt")],
            self.temp_dir,
        )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.checks), 2)
        self.assertFalse(result.checks[0].passed)
        self.assertTrue(result.checks[1].passed)

    def test_non_iterable_expectations_fail_without_type_error(self) -> None:
        """``verify(None, base)`` must not let ``TypeError`` escape."""
        result = self.make_verifier().verify(None, self.temp_dir)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.checks), 1)
        check = result.checks[0]
        self.assertFalse(check.passed)
        self.assertIn("NoneType", check.actual)
        self.assertIsNotNone(check.error)

    def test_non_iterable_expectations_never_report_all_checks_passed(self) -> None:
        for container in (None, 7, object()):
            with self.subTest(kind=type(container).__name__):
                result = self.make_verifier().verify(container, self.temp_dir)
                self.assertFalse(result.passed)
                self.assertNotEqual(result.reason, "All checks passed")

    def test_containment_still_rejects_before_a_valid_check_runs(self) -> None:
        """Non-vacuity must not weaken 2F.2 containment."""
        workspace = self.temp_dir / "workspace"
        workspace.mkdir()
        outside = self.temp_dir / "outside.txt"
        outside.write_text("should never be read")

        result = self.make_verifier().verify(
            [
                FileExpectation("../outside.txt", contains="should never be read"),
                FileExpectation("anything.txt"),
            ],
            workspace,
        )

        self.assertFalse(result.passed)
        self.assertEqual(len(result.checks), 2)
        self.assertIn("Security violation", result.checks[0].error)
        self.assertFalse(result.checks[1].passed)
        self.assertEqual(result.checks[1].actual, "missing")

    def test_unsafe_expectation_is_rejected_even_alongside_an_empty_verdict(self) -> None:
        """A security violation is never laundered into a non-vacuity failure."""
        workspace = self.temp_dir / "workspace"
        workspace.mkdir()
        result = self.make_verifier().verify(
            [FileExpectation("../../etc/passwd")],
            workspace,
        )
        self.assertFalse(result.passed)
        self.assertIn("Security violation", result.checks[0].error)
        self.assertNotIn("no checks were performed", result.reason.lower())

    def test_malformed_expectations_do_not_mutate_the_filesystem(self) -> None:
        (self.temp_dir / "stable.txt").write_text("unchanged")
        (self.temp_dir / "adir").mkdir()
        before = sorted(str(p) for p in self.temp_dir.rglob("*"))

        verifier = self.make_verifier()
        verifier.verify([], self.temp_dir)
        verifier.verify(None, self.temp_dir)
        verifier.verify(["junk"], self.temp_dir)
        verifier.verify([None, 7, object()], self.temp_dir)
        verifier.verify(
            [FileExpectation("stable.txt", contains="unchanged")], self.temp_dir
        )

        after = sorted(str(p) for p in self.temp_dir.rglob("*"))
        self.assertEqual(before, after)
        self.assertEqual(
            (self.temp_dir / "stable.txt").read_text(encoding="utf-8"), "unchanged"
        )

    def test_valid_expectations_still_pass_after_the_hardening(self) -> None:
        (self.temp_dir / "a.txt").write_text("data")
        result = self.make_verifier().verify(
            [FileExpectation("a.txt", exists=True, contains="data")],
            self.temp_dir,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "All checks passed")
        self.assertEqual(len(result.checks), 1)
        self.assertTrue(result.checks[0].passed)

    def test_valid_aggregation_is_unchanged(self) -> None:
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


class TestBoundedVerificationReads(FilesystemVerifierTestBase):
    """A ``contains`` check reads at most ``MAX_VERIFY_READ_BYTES`` (2F.4).

    Workspace content is untrusted and the executor may write files of any
    size, so the verifier must never pull an unbounded blob into memory to
    decide one check.  A target larger than the bound is refused *without
    being read* and fails closed: the verifier must not PASS a file it
    declined to inspect, because it holds no evidence about that file.

    Fixtures stay small and deliberately above the bound.  The oversized cases
    always embed the expected needle, so a failure can only be attributed to
    the bound and never to the string simply being absent.
    """

    def _write(self, name: str, payload: bytes) -> Path:
        path = self.temp_dir / name
        path.write_bytes(payload)
        return path

    def _payload_with(self, size: int, needle: str) -> bytes:
        """Exactly *size* bytes that do contain *needle*."""
        if size < len(needle):
            raise ValueError("size must fit the needle")
        return needle.encode("utf-8") + b"a" * (size - len(needle))

    # --- A. existing valid behaviour is preserved ------------------------

    def test_file_below_bound_with_matching_contains_passes(self) -> None:
        self._write("small.txt", b"hello world and a little more")
        result = self.make_verifier().verify(
            [FileExpectation("small.txt", contains="hello world")],
            self.temp_dir,
        )
        self.assertTrue(result.passed, result.reason)
        self.assertEqual(result.reason, "All checks passed")
        self.assertEqual(len(result.checks), 1)
        self.assertTrue(result.checks[0].passed)
        self.assertEqual(result.checks[0].actual, "contains")

    def test_file_exactly_at_bound_with_matching_contains_passes(self) -> None:
        """The refusal is ``size > bound``; exactly at the bound still reads."""
        self._write(
            "exact.bin",
            self._payload_with(MAX_VERIFY_READ_BYTES, "NEEDLE"),
        )
        result = self.make_verifier().verify(
            [FileExpectation("exact.bin", contains="NEEDLE")],
            self.temp_dir,
        )
        self.assertTrue(result.passed, result.reason)
        self.assertEqual(result.reason, "All checks passed")

    def test_all_checks_passed_reason_is_unchanged(self) -> None:
        self._write("a.txt", b"data")
        result = self.make_verifier().verify(
            [FileExpectation("a.txt", exists=True, contains="data")],
            self.temp_dir,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "All checks passed")

    def test_bound_matches_the_inspection_precedent(self) -> None:
        """The verifier bound mirrors ``WorkspaceInspector.read_text`` default."""
        self.assertEqual(
            MAX_VERIFY_READ_BYTES,
            inspect_max_bytes_default(),
        )
        # Guard against the two defaults silently drifting apart.
        self.assertEqual(MAX_VERIFY_READ_BYTES, 65536)

    # --- B. oversized targets fail closed --------------------------------

    def test_oversized_file_containing_needle_fails_closed(self) -> None:
        """The needle is present, yet the file is still refused.

        This is the case that distinguishes a bounded refusal from a plain
        content miss: the expected text really is in the file, so the only
        reason this cannot pass is the bound.
        """
        size = MAX_VERIFY_READ_BYTES + 1
        self._write("big.bin", self._payload_with(size, "NEEDLE"))
        result = self.make_verifier().verify(
            [FileExpectation("big.bin", contains="NEEDLE")],
            self.temp_dir,
        )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.checks), 1)
        check = result.checks[0]
        self.assertFalse(check.passed)
        self.assertIn("too large", check.actual)
        self.assertIn(str(size), check.actual)
        self.assertNotIn("does not contain", check.actual)
        self.assertIn("Bounded verification", check.error)

    def test_oversized_file_without_needle_fails_closed(self) -> None:
        size = MAX_VERIFY_READ_BYTES + 1
        self._write("big.bin", b"a" * size)
        result = self.make_verifier().verify(
            [FileExpectation("big.bin", contains="NEEDLE")],
            self.temp_dir,
        )
        self.assertFalse(result.passed)
        self.assertIn("too large", result.checks[0].actual)

    def test_comfortably_oversized_fixture_fails_closed(self) -> None:
        """A clearly oversized file is refused, still without being read."""
        size = 2 * MAX_VERIFY_READ_BYTES
        self._write("big.bin", self._payload_with(size, "NEEDLE"))
        result = self.make_verifier().verify(
            [FileExpectation("big.bin", contains="NEEDLE")],
            self.temp_dir,
        )
        self.assertFalse(result.passed)
        self.assertIn("too large", result.checks[0].actual)

    def test_oversized_failure_is_a_controlled_check_result(self) -> None:
        """No exception escapes; the outcome is ordinary check data."""
        self._write("big.bin", b"a" * (MAX_VERIFY_READ_BYTES + 1))
        result = self.make_verifier().verify(
            [FileExpectation("big.bin", contains="NEEDLE")],
            self.temp_dir,
        )
        self.assertFalse(result.passed)
        self.assertTrue(result.checks, "a CheckResult must be reported")
        check = result.checks[0]
        self.assertFalse(check.passed)
        self.assertIsInstance(check.name, str)
        self.assertIsInstance(check.expected, str)
        self.assertIsInstance(check.actual, str)
        self.assertIsInstance(check.error, str)

    def test_oversized_verification_never_reports_pass(self) -> None:
        """The verifier must never claim success for a file it would not read."""
        for content in (self._payload_with(MAX_VERIFY_READ_BYTES + 1, "NEEDLE"), b"a" * (MAX_VERIFY_READ_BYTES + 1)):
            with self.subTest(size=len(content)):
                (self.temp_dir / "big.bin").write_bytes(content)
                result = self.make_verifier().verify(
                    [FileExpectation("big.bin", contains="NEEDLE")],
                    self.temp_dir,
                )
                self.assertFalse(result.passed)
                self.assertNotEqual(result.reason, "All checks passed")

    def test_exists_only_expectation_is_not_implicated_by_the_bound(self) -> None:
        """The bound applies to ``contains`` reads only; existence is unaffected."""
        self._write("big.bin", b"a" * (MAX_VERIFY_READ_BYTES + 1))
        result = self.make_verifier().verify(
            [FileExpectation("big.bin", exists=True)],
            self.temp_dir,
        )
        self.assertTrue(result.passed, result.reason)

    # --- C/D. regression: 2F.2 containment and 2F.3 non-vacuity ----------

    def test_containment_still_rejects_an_oversized_escape(self) -> None:
        """A bound check must not weaken 2F.2 containment."""
        workspace = self.temp_dir / "workspace"
        workspace.mkdir()
        outside = self.temp_dir / "outside.txt"
        outside.write_bytes(b"a" * (MAX_VERIFY_READ_BYTES + 1))
        result = self.make_verifier().verify(
            [FileExpectation("../outside.txt", contains="a")],
            workspace,
        )
        self.assertFalse(result.passed)
        self.assertIn("Security violation", result.checks[0].error)

    def test_non_vacuity_still_enforced_with_oversized_expectations(self) -> None:
        """2F.3 rules hold unchanged when the bound is in play."""
        self._write("big.bin", b"a" * (MAX_VERIFY_READ_BYTES + 1))
        self._write("ok.txt", b"fine")
        result = self.make_verifier().verify(
            [FileExpectation("ok.txt", contains="fine")],
            self.temp_dir,
        )
        self.assertTrue(result.passed, result.reason)

        empty = self.make_verifier().verify([], self.temp_dir)
        self.assertFalse(empty.passed)
        self.assertEqual(empty.checks, ())

        none = self.make_verifier().verify(None, self.temp_dir)
        self.assertFalse(none.passed)
        self.assertEqual(len(none.checks), 1)

    def test_mixed_sizes_aggregate_with_oversized_items(self) -> None:
        """An oversized item fails its own check; later items still run."""
        self._write("big.bin", self._payload_with(MAX_VERIFY_READ_BYTES + 1, "NEEDLE"))
        self._write("ok.txt", b"fine content")
        result = self.make_verifier().verify(
            [
                FileExpectation("ok.txt", contains="fine content"),
                FileExpectation("big.bin", contains="NEEDLE"),
            ],
            self.temp_dir,
        )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.checks), 2)
        self.assertTrue(result.checks[0].passed)
        self.assertFalse(result.checks[1].passed)
        self.assertIn("too large", result.checks[1].actual)
        self.assertIn("1 of 2 checks failed", result.reason)

    # --- E. newline normalisation is preserved ----------------------------

    def test_crlf_file_within_bound_still_matches_lf_needle(self) -> None:
        """``Path.read_text`` normalises CRLF/CR to LF; that must not change."""
        self._write("crlf.txt", b"line one\r\nline two\rline three\r\n")
        result = self.make_verifier().verify(
            [FileExpectation("crlf.txt", contains="line one\nline two\nline three")],
            self.temp_dir,
        )
        self.assertTrue(result.passed, result.reason)

    def test_cr_only_file_within_bound_still_matches_lf_needle(self) -> None:
        self._write("cr.txt", b"a\rb")
        result = self.make_verifier().verify(
            [FileExpectation("cr.txt", contains="a\nb")],
            self.temp_dir,
        )
        self.assertTrue(result.passed, result.reason)

    def test_newline_normalisation_matches_inspection_reader(self) -> None:
        """The verifier and ``WorkspaceInspector`` must agree on line endings."""
        from core.inspection import WorkspaceInspector  # noqa: PLC0415

        payload = b"x\r\ny\rz\n"
        self._write("crlf.txt", payload)
        result = self.make_verifier().verify(
            [FileExpectation("crlf.txt", contains="x\ny\nz\n")],
            self.temp_dir,
        )
        self.assertTrue(result.passed, result.reason)
        inspected = WorkspaceInspector(self.temp_dir).read_text("crlf.txt")
        self.assertEqual(inspected, "x\ny\nz\n")

    # --- F. safety: no mutation -------------------------------------------

    def test_oversized_verification_does_not_modify_the_target(self) -> None:
        before = sorted(str(p) for p in self.temp_dir.rglob("*"))
        payload = self._payload_with(MAX_VERIFY_READ_BYTES + 1, "NEEDLE")
        self._write("big.bin", payload)
        after_write = sorted(str(p) for p in self.temp_dir.rglob("*"))

        result = self.make_verifier().verify(
            [FileExpectation("big.bin", contains="NEEDLE")],
            self.temp_dir,
        )
        self.assertFalse(result.passed)

        self.assertEqual(before + [str(self.temp_dir / "big.bin")], after_write)
        self.assertEqual(
            sorted(str(p) for p in self.temp_dir.rglob("*")), after_write
        )
        self.assertEqual((self.temp_dir / "big.bin").read_bytes(), payload)

    def test_malformed_and_oversized_verification_create_no_files(self) -> None:
        self._write("big.bin", b"a" * (MAX_VERIFY_READ_BYTES + 1))
        self._write("ok.txt", b"fine")
        baseline = sorted(str(p) for p in self.temp_dir.rglob("*"))

        verifier = self.make_verifier()
        verifier.verify([], self.temp_dir)
        verifier.verify(None, self.temp_dir)
        verifier.verify(["junk"], self.temp_dir)
        verifier.verify(
            [FileExpectation("big.bin", contains="NEEDLE")], self.temp_dir
        )
        verifier.verify([FileExpectation("ok.txt", contains="fine")], self.temp_dir)

        # Verification must neither create nor modify anything, and the
        # oversized target must survive byte-for-byte.
        self.assertEqual(
            sorted(str(p) for p in self.temp_dir.rglob("*")), baseline
        )
        self.assertEqual(
            (self.temp_dir / "big.bin").read_bytes(),
            b"a" * (MAX_VERIFY_READ_BYTES + 1),
        )


def inspect_max_bytes_default() -> int:
    """Return ``WorkspaceInspector.read_text``'s ``max_bytes`` default."""
    from core.inspection import WorkspaceInspector  # noqa: PLC0415
    import inspect as _inspect  # noqa: PLC0415

    signature = _inspect.signature(WorkspaceInspector.read_text)
    return int(signature.parameters["max_bytes"].default)


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