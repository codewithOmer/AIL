"""Focused tests for the application memory/task boundary."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.application import AILApplication  # noqa: E402
from core.planner import UnsupportedTaskError  # noqa: E402
from integrations.open_interpreter.client import (  # noqa: E402
    OIError,
    OITransportFailure,
)
from interfaces.conversation import ConversationResponse  # noqa: E402
from interfaces.planning import ExecutionReport  # noqa: E402
from interfaces.memory import Memory, MemoryStore  # noqa: E402


class RecordingMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self.memories: list[Memory] = []
        self.store_calls = 0

    def store(
        self,
        text: str,
        *,
        kind: str = "semantic",
        confidence: float = 0.8,
    ) -> Memory:
        self.store_calls += 1
        memory = Memory(id=str(self.store_calls), text=text, kind=kind, confidence=confidence)
        self.memories.append(memory)
        return memory

    def recall(self, query: str, *, top_k: int = 5) -> list[Memory]:
        return self.memories[:top_k]

    def list_all(self) -> list[Memory]:
        return list(self.memories)

    def get(self, memory_id: str) -> Memory | None:
        return next((memory for memory in self.memories if memory.id == memory_id), None)

    def delete(self, memory_id: str) -> None:
        self.memories = [memory for memory in self.memories if memory.id != memory_id]


class RecordingClient:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.calls: list[str] = []

    def send_message(
        self,
        message: str,
        thread_id: str | None = None,
        timeout: float | None = None,
    ) -> str:
        self.calls.append(message)
        content = (
            "remember that I like coffee"
            if "remember that" in message.lower()
            else "hello"
        )
        (self.workspace / "notes.txt").write_text(content, encoding="utf-8")
        return "executor response"


class TestApplicationIntentRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def make_application(self) -> tuple[AILApplication, RecordingMemoryStore, RecordingClient]:
        store = RecordingMemoryStore()
        client = RecordingClient(self.temp_dir)
        application = AILApplication.create(
            base_dir=self.temp_dir,
            memory_store=store,
            client=client,
            start_client=False,
        )
        return application, store, client

    def test_task_uses_agent_path_without_memory_write(self) -> None:
        application, store, client = self.make_application()

        result = application.handle("Create a file named notes.txt containing hello")

        self.assertIsInstance(result, ExecutionReport)
        self.assertTrue(result.passed)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(store.memories, [])

    def test_explicit_memory_writes_without_calling_executor(self) -> None:
        application, store, client = self.make_application()

        result = application.handle("Remember that I like coffee")

        self.assertEqual([memory.text for memory in result], ["I like coffee."])
        self.assertEqual(store.store_calls, 1)
        self.assertEqual(client.calls, [])

    def test_duplicate_explicit_memory_remains_deduplicated(self) -> None:
        application, store, client = self.make_application()

        application.handle("Remember that I like coffee")
        application.handle("Remember that I like coffee!")

        self.assertEqual(len(store.memories), 1)
        self.assertEqual(client.calls, [])

    def test_unsupported_non_memory_message_preserves_behavior(self) -> None:
        application, store, client = self.make_application()

        result = application.handle("Calculate 23 times 19")

        self.assertIsInstance(result, ConversationResponse)
        self.assertEqual(result.text, "executor response")
        self.assertEqual(store.memories, [])
        self.assertEqual(client.calls, ["Calculate 23 times 19"])

    def test_conversation_does_not_claim_filesystem_verification(self) -> None:
        application, _, _ = self.make_application()

        result = application.handle("hello")

        self.assertIsInstance(result, ConversationResponse)
        self.assertNotIsInstance(result, ExecutionReport)

    def test_conversation_uses_text_from_client_response_object(self) -> None:
        application, _, client = self.make_application()
        client.send_message = lambda *args, **kwargs: SimpleNamespace(
            text="model says hello"
        )

        result = application.handle("hello")

        self.assertEqual(result, ConversationResponse("model says hello"))

    def test_empty_and_whitespace_messages_are_rejected(self) -> None:
        application, _, client = self.make_application()

        for message in ("", "   "):
            with self.subTest(message=repr(message)):
                with self.assertRaises(UnsupportedTaskError):
                    application.handle(message)

        self.assertEqual(client.calls, [])

    def test_unsafe_file_path_is_rejected_without_conversation_fallback(self) -> None:
        application, _, client = self.make_application()

        with self.assertRaises(UnsupportedTaskError):
            application.handle(
                r"Create a file named ..\outside.txt containing secret"
            )

        self.assertEqual(client.calls, [])

    def test_conversation_model_and_transport_failures_are_propagated(self) -> None:
        for failure in (OIError("model failed"), OITransportFailure("transport failed")):
            with self.subTest(failure=type(failure).__name__):
                application, _, client = self.make_application()

                def fail(*args: object, **kwargs: object) -> str:
                    raise failure

                client.send_message = fail  # type: ignore[method-assign]
                with self.assertRaises(type(failure)) as caught:
                    application.handle("hello")

                self.assertEqual(str(caught.exception), str(failure))

    def test_task_payload_memory_words_are_not_persisted(self) -> None:
        application, store, client = self.make_application()

        result = application.handle(
            "Create a file named notes.txt containing remember that I like coffee"
        )

        self.assertIsInstance(result, ExecutionReport)
        self.assertTrue(result.passed)
        self.assertEqual(
            (self.temp_dir / "notes.txt").read_text(encoding="utf-8"),
            "remember that I like coffee",
        )
        self.assertEqual(store.memories, [])
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
