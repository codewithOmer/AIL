"""AIL main entry point.

Demonstrates the AIL → Open Interpreter App Server vertical slice with
long-term memory: recall relevant memories before each turn, and store a
small set of explicit user-provided facts after each turn.
"""

import sys

from core.agent import Agent, MockLLM
from core.config import setup_logging
from core.planner import UnsupportedTaskError


def run_mock_mode() -> None:
    """Original mock LLM mode."""
    agent = Agent(MockLLM())
    user_input = input("You: ")
    response = agent.respond(user_input)
    print(f"AIL: {response}")


def run_oi_mode() -> None:
    """Run supported verified tasks against a real OI App Server."""
    from core.application import AILApplication
    from integrations.open_interpreter.client import (
        OIConnectionError,
        OIError,
    )

    application: AILApplication | None = None
    try:
        application = AILApplication.create()

        while True:
            message = input("You: ").strip()
            if not message or message.lower() in ("exit", "quit"):
                break

            try:
                report = application.run(message)
            except UnsupportedTaskError as exc:
                print(f"AIL: unsupported task ({exc})")
                continue
            _print_report(report)
    except OIConnectionError as exc:
        print(f"Failed to connect to OI App Server: {exc}", file=sys.stderr)
        sys.exit(1)
    except OIError as exc:
        print(f"OI error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if application is not None:
            application.close()


def _print_report(report) -> None:
    """Print a concise result using only the verified execution report."""
    if report.passed:
        message = "verified task completed"
    else:
        failed = [
            step.step_id
            for step in report.steps
            if step.error or step.status.value == "failed"
        ]
        message = f"task failed ({', '.join(failed) or 'verification failure'})"

    recovery_attempts = sum(
        step.recovery.attempts
        for step in report.steps
        if step.recovery is not None and step.recovery.attempts > 1
    )
    if recovery_attempts:
        message += f"; recovery attempts: {recovery_attempts}"
    if len(report.attempts) > 1:
        message += "; replanned once"
    print(f"AIL: {message}")


def main() -> None:
    setup_logging()
    mode = sys.argv[1] if len(sys.argv) > 1 else "mock"
    if mode == "oi":
        run_oi_mode()
    else:
        run_mock_mode()


if __name__ == "__main__":
    main()