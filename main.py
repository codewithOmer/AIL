"""AIL main entry point.

Demonstrates the AIL → Open Interpreter App Server vertical slice with
long-term memory: recall relevant memories before each turn, and store a
small set of explicit user-provided facts after each turn.
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
    """Run against a real OI App Server with long-term memory."""
    from integrations.open_interpreter.client import (
        OIConnectionError,
        OIError,
        OpenInterpreterClient,
    )
    from integrations.personalai.memory import PersonalAIMemoryStore
    from memory.flow import apply_context, extract_explicit_facts, store_if_new

    try:
        # One in-memory store for the whole process (memory grows as turns flow).
        store = PersonalAIMemoryStore()
        client = OpenInterpreterClient()
        client.start()
        thread_id = client.create_thread()

        while True:
            message = input("You: ").strip()
            if not message or message.lower() in ("exit", "quit"):
                break

            # 1. Recall relevant memories for this turn.
            recalled = store.recall(message, top_k=3)
            if recalled:
                print(f"[AIL recalled {len(recalled)} memory/memories]")

            # 2. Send the user message (with memory context when available).
            response = client.send_message(
                apply_context(message, recalled),
                thread_id=thread_id,
            )
            print(f"AIL: {response.text}")

            # 3. Store only explicit durable facts the user just stated.
            for fact in extract_explicit_facts(message):
                stored = store_if_new(store, fact)
                if stored is not None:
                    print(f"[AIL remembered: {stored.text}]")

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