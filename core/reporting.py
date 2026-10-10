"""Small shared formatter for truthful execution results."""

from __future__ import annotations

from interfaces.planning import ExecutionReport, StepFailureKind, StepStatus


def report_text(report: ExecutionReport) -> str:
    if report.passed:
        return "verified task completed"

    failed = [
        step
        for step in report.steps
        if step.status is StepStatus.FAILED
    ]
    if not failed:
        message = "verification failure"
    else:
        details: list[str] = []
        for step in failed:
            kind = step.failure_kind
            label = (
                "verification failure"
                if kind in (StepFailureKind.NONE, StepFailureKind.VERIFICATION)
                else f"{kind.value} failure"
            )
            detail = f": {step.error}" if step.error else ""
            details.append(f"{step.step_id} ({label}{detail})")
        message = ", ".join(details)

    result = f"task failed ({message})"
    recovery_attempts = sum(
        step.recovery.attempts
        for step in report.steps
        if step.recovery is not None and step.recovery.attempts > 1
    )
    if recovery_attempts:
        result += f"; recovery attempts: {recovery_attempts}"
    if len(report.attempts) > 1:
        result += "; replanned once"
    return result
