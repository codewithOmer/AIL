"""Filesystem verification tool.

AIL-owned, executor-independent check that inspects the filesystem directly
with pathlib.  It never shells out, never calls Open Interpreter, never parses
an executor response, and never modifies, creates or deletes files.

Containment (Milestone 2F.2): every expectation path is resolved and must
remain inside the resolved base directory.  The verifier enforces this
independently, so a planner or executor that supplies an unsafe expectation
cannot make it read, stat, or assert anything outside the configured
workspace.  Containment is decided with component-aware path comparison, not
string prefixes.

Non-vacuity (Milestone 2F.3): ``passed=True`` is only ever returned after at
least one check was actually performed and every performed check passed.  An
empty expectation set therefore FAILS rather than passing through
``all([])``, and an expectation this verifier cannot consume becomes a failed
``CheckResult`` instead of an ``AttributeError`` or ``TypeError`` escaping
``verify``.  The verifier refuses to certify success it has no evidence for,
independently of whatever planner or executor supplied the expectations.  This
is the same "enforce it where it is used, not where it is called" boundary as
2F.2 containment: only the real filesystem check is trusted, and only a check
that actually ran may contribute to the verdict.

Bounded reads (Milestone 2F.4): a ``contains`` expectation is decided by
reading the target, and the workspace is untrusted, so those reads are bounded
by :data:`MAX_VERIFY_READ_BYTES`.  A target larger than the bound is refused
*without being read* and fails closed: an oversized file can never produce a
PASS, because the verifier will not claim a check it did not perform.  This is
an availability/DoS guard against read amplification (an unbounded
``read_text`` on a multi-gigabyte file would exhaust memory inside the very
component every other guarantee depends on); it mirrors the bounded-read rule
``core.inspection`` already applies to the same content.  The bound mirrors
``WorkspaceInspector.read_text``'s default ``max_bytes`` so both AIL paths
that read workspace content agree.  Workspace containment above remains the
security boundary: this limit is about resource safety, not about deciding
whether a path is allowed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Upper bound on bytes read from one expectation target to decide a
# ``contains`` check.  Matches ``WorkspaceInspector.read_text``'s default
# ``max_bytes``; see the module docstring for why reads are bounded.
MAX_VERIFY_READ_BYTES = 65536

# ``FileExpectation`` is re-exported here for compatibility: it is contract
# data owned by ``interfaces.verification``, but it has always been importable
# from this module and callers rely on that.
from interfaces.verification import (
    CheckResult,
    FileExpectation,
    VerificationResult,
    Verifier,
)


class FilesystemVerifier(Verifier):
    """Verify file existence and text-content expectations."""

    def verify(self, expectations: Any, base_dir: Any) -> VerificationResult:
        # The base is kept unresolved for reporting compatibility; containment
        # itself always compares resolved Path objects.
        base = Path(base_dir)
        results = self._check_all(base, expectations)
        passed = bool(results) and all(r.passed for r in results)
        if not results:
            # `bool(results)` is what makes the verdict non-vacuous: a plan that
            # declares nothing to check is never a pass.  `all([])` is True and
            # would otherwise certify success with zero evidence performed.
            reason = "No expectations to verify: no checks were performed"
        elif passed:
            reason = "All checks passed"
        else:
            failed = [r.name for r in results if not r.passed]
            reason = (
                f"{len(failed)} of {len(results)} checks failed: "
                + ", ".join(failed)
            )
        return VerificationResult(
            passed=passed,
            reason=reason,
            checks=tuple(results),
        )

    def _check_all(self, base: Path, expectations: Any) -> list[CheckResult]:
        """Run one check per expectation, or report why none could be run.

        Malformed *structure* is handled here at the boundary: a
        non-iterable container, an empty set, or an expectation of a type this
        verifier does not support.  The real filesystem verification in
        ``_check`` is left unguarded, so genuine programming or filesystem
        errors still surface instead of being laundered into a check result.
        """
        try:
            iterator = iter(expectations)
        except TypeError:
            return [
                CheckResult(
                    name="<expectations>",
                    passed=False,
                    expected="an iterable of expectations",
                    actual=(
                        f"unsupported expectations container of type "
                        f"{type(expectations).__name__}"
                    ),
                    error="Unsupported expectations: not an iterable container",
                )
            ]

        results: list[CheckResult] = []
        while True:
            try:
                expectation = next(iterator)
            except StopIteration:
                break
            except TypeError as exc:
                # A container that iterates for a while and then turns
                # non-iterable: record it and keep the checks already made.
                results.append(
                    CheckResult(
                        name="<expectations>",
                        passed=False,
                        expected="an iterable of expectations",
                        actual="expectations iteration error",
                        error=f"Unsupported expectations: {exc.__class__.__name__}",
                    )
                )
                break
            if isinstance(expectation, FileExpectation):
                results.append(self._check(base, expectation))
            else:
                # Type-oriented information only.  The value itself is never
                # rendered, because it is untrusted and may carry paths or
                # content that must not reach an executor-facing message.
                results.append(
                    CheckResult(
                        name=f"<expectation:{type(expectation).__name__}>",
                        passed=False,
                        expected="a supported expectation type",
                        actual=(
                            f"unsupported expectation of type "
                            f"{type(expectation).__name__}"
                        ),
                        error=(
                            "Unsupported expectation: FilesystemVerifier only "
                            "verifies FileExpectation"
                        ),
                    )
                )
        return results

    def _check(self, base: Path, exp: FileExpectation) -> CheckResult:
        candidate = base / exp.path
        try:
            resolved_base = base.resolve()
            resolved_target = candidate.resolve()
        except (OSError, ValueError) as exc:
            # Unresolvable base or expectation path is a failed check, never a
            # silent pass and never a crash that escapes verification.
            return CheckResult(
                name=str(candidate),
                passed=False,
                expected=f"exists={exp.exists}",
                actual="path resolution error",
                error=str(exc),
            )

        # Security boundary: the resolved expectation target must stay inside
        # the resolved base.  Component-aware comparison, so sibling-prefix
        # paths such as base-evil are correctly rejected.  Checked before any
        # stat or read so an unsafe expectation never touches the filesystem
        # outside the workspace.
        if not resolved_target.is_relative_to(resolved_base):
            return CheckResult(
                name=str(candidate),
                passed=False,
                expected="Path inside workspace",
                actual="Path escape attempt",
                error="Security violation: path outside workspace",
            )

        target = resolved_target
        try:
            target.stat()
            actual_exists = True
        except FileNotFoundError:
            actual_exists = False
        except OSError as exc:
            return CheckResult(
                name=str(target),
                passed=False,
                expected=f"exists={exp.exists}",
                actual="stat error",
                error=str(exc),
            )
        if exp.exists:
            return self._check_exists(target, exp, actual_exists)
        return self._check_absent(target, actual_exists)

    def _check_exists(
        self, target: Path, exp: FileExpectation, actual_exists: bool
    ) -> CheckResult:
        if not actual_exists:
            return CheckResult(
                name=str(target),
                passed=False,
                expected="exists",
                actual="missing",
            )
        if exp.contains is not None:
            # Resource boundary (2F.4): decide ``contains`` from a bounded read.
            # The target is re-stat'd for its size here, before any read, so an
            # oversized target is never read into memory at all.  Failing closed
            # is deliberate — the verifier must not PASS a file it declined to
            # read, because it has no evidence about that file's contents.
            try:
                size = target.stat().st_size
            except OSError as exc:
                return CheckResult(
                    name=str(target),
                    passed=False,
                    expected=f"contains {exp.contains!r}",
                    actual="stat error",
                    error=str(exc),
                )
            if size > MAX_VERIFY_READ_BYTES:
                return CheckResult(
                    name=str(target),
                    passed=False,
                    expected=f"contains {exp.contains!r}",
                    actual=f"file too large to verify ({size} bytes)",
                    error=(
                        "Bounded verification: refusing to read "
                        f"{size} bytes, limit is {MAX_VERIFY_READ_BYTES} bytes"
                    ),
                )
            try:
                text = target.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                return CheckResult(
                    name=str(target),
                    passed=False,
                    expected=f"contains {exp.contains!r}",
                    actual="read error",
                    error=str(exc),
                )
            if exp.contains not in text:
                return CheckResult(
                    name=str(target),
                    passed=False,
                    expected=f"contains {exp.contains!r}",
                    actual=f"does not contain {exp.contains!r}",
                )
            return CheckResult(
                name=str(target),
                passed=True,
                expected=f"contains {exp.contains!r}",
                actual="contains",
            )
        return CheckResult(
            name=str(target),
            passed=True,
            expected="exists",
            actual="exists",
        )

    def _check_absent(self, target: Path, actual_exists: bool) -> CheckResult:
        if actual_exists:
            return CheckResult(
                name=str(target),
                passed=False,
                expected="not exists",
                actual="exists",
            )
        return CheckResult(
            name=str(target),
            passed=True,
            expected="not exists",
            actual="not exists",
        )
