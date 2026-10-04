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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
        results = [self._check(base, exp) for exp in expectations]
        passed = all(r.passed for r in results)
        failed = [r.name for r in results if not r.passed]
        if passed:
            reason = "All checks passed"
        else:
            reason = (
                f"{len(failed)} of {len(results)} checks failed: "
                + ", ".join(failed)
            )
        return VerificationResult(
            passed=passed,
            reason=reason,
            checks=tuple(results),
        )

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
