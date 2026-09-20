"""AIL main entry point.

Demonstrates the AIL → Open Interpreter App Server vertical slice.
"""

import sys

from core.agent import Agent, MockLLM
from core.config import setup_logging


def run_mock_mode() -> None:
    """Original mock LLM mode."""
    agent = Agent(MockLLM())
    user_input = input("You: ")
    response = agent.respond(user_input)
    print(f"AIL: {response}")


def run_oi_mode() -> None:
    """Run against a real OI App Server."""
    from integrations.open_interpreter.client import (
        OIConnectionError,
        OIError,
        OpenInterpreterClient,
    )

    try:
        client = OpenInterpreterClient()
        client.start()
        thread_id = client.create_thread()
        message = input("You: ")
        response = client.send_message(message, thread_id=thread_id)
        print(f"AIL: {response.text}")
        client.shutdown()
    except OIConnectionError as exc:
        print(f"Failed to connect to OI App Server: {exc}", file=sys.stderr)
        sys.exit(1)
    except OIError as exc:
        print(f"OI error: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    setup_logging()
    mode = sys.argv[1] if len(sys.argv) > 1 else "mock"
    if mode == "oi":
        run_oi_mode()
    else:
        run_mock_mode()


if __name__ == "__main__":
    main()
