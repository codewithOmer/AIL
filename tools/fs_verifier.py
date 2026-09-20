"""Filesystem verification tool.

AIL-owned, executor-independent check that inspects the filesystem directly
with pathlib.  It never shells out, never calls Open Interpreter, never parses
an executor response, and never modifies, creates or deletes files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interfaces.verification import CheckResult, VerificationResult, Verifier


@dataclass(frozen=True)
class FileExpectation:
    """Expected filesystem state, relative to the verification base directory."""

    path: str | Path
    exists: bool = True
    contains: str | None = None


class FilesystemVerifier(Verifier):
    """Verify file existence and text-content expectations."""

    def verify(self, expectations: Any, base_dir: Any) -> VerificationResult:
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
        target = base / exp.path
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