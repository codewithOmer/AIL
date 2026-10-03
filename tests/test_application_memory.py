"""Focused tests for the explicit application memory boundary."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from core.agent import Agent  # noqa: E402
from core.application import AILApplication  # noqa: E402
from interfaces.planning import ExecutionReport, Goal  # noqa: E402
from memory.storage.personalai import PersonalAIMemoryStore  # noqa: E402
from interfaces.memory import Memory, MemoryStore  # noqa: E402


class RecordingRunner:
    def __init__(self) -> None:
        self.goals: list[Goal] = []

    def run(self, goal: Goal) -> ExecutionReport:
        self.goals.append(goal)
        return ExecutionReport(goal=goal, passed=True, steps=())


class TaskClient:
    def __init__(self, workspace: Path, content: str = "remember that I like coffee") -> None:
        self.workspace = workspace
        self.content = content

    def send_message(
        self,
        message: str,
        thread_id: str | None = None,
        timeout: float | None = None,
    ) -> str:
        (self.workspace / "hello.txt").write_text(
            self.content,
            encoding="utf-8",
        )
        return "executor response"


class TestApplicationMemory(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.memory_file = self.temp_dir / "memory.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def store(self) -> PersonalAIMemoryStore:
        return PersonalAIMemoryStore(memory_file=self.memory_file)

    def application(
        self,
        store: MemoryStore,
        content: str = "remember that I like coffee",
    ) -> AILApplication:
        return AILApplication.create(
            base_dir=self.temp_dir,
            memory_store=store,
            client=TaskClient(self.temp_dir, content=content),
            start_client=False,
        )

    def test_standalone_name_fact_is_stored(self) -> None:
        store = self.store()
        self.application(store).remember("My name is Omer.")

        self.assertEqual([memory.text for memory in store.list_all()], [
            "The user's name is Omer."
        ])

    def test_standalone_remember_fact_is_stored(self) -> None:
        store = self.store()
        self.application(store).remember("Remember that I like coffee.")

        self.assertEqual([memory.text for memory in store.list_all()], [
            "I like coffee."
        ])

    def test_task_payload_does_not_store_embedded_remember_fact(self) -> None:
        store = self.store()
        application = self.application(store)

        report = application.run(
            "Create a file named hello.txt containing remember that I like coffee"
        )

        self.assertTrue(report.passed)
        self.assertEqual(store.list_all(), [])

    def test_task_payload_without_memory_words_does_not_store_memory(self) -> None:
        store = self.store()

        report = self.application(store, content="anything").run(
            "Create a file named hello.txt containing anything"
        )

        self.assertTrue(report.passed)
        self.assertEqual(store.list_all(), [])

    def test_exact_duplicate_does_not_create_another_memory(self) -> None:
        store = self.store()
        application = self.application(store)

        application.remember("Remember that I like coffee.")
        application.remember("Remember that I like coffee.")

        self.assertEqual(len(store.list_all()), 1)

    def test_persistence_survives_new_store_instance(self) -> None:
        store_a = self.store()
        self.application(store_a).remember("My name is Omer.")

        store_b = self.store()

        self.assertEqual(store_b.list_all()[0].text, "The user's name is Omer.")

    def test_recall_after_reload_uses_token_overlap(self) -> None:
        store_a = self.store()
        self.application(store_a).remember("Remember that I like coffee.")

        store_b = self.store()
        results = store_b.recall("What do I like to drink?")

        self.assertGreater(len(results), 0)
        self.assertIn("coffee", results[0].text)

    def test_agent_context_contains_recalled_memory(self) -> None:
        store = self.store()
        store.store("The user's name is Omer.")
        runner = RecordingRunner()

        Agent(store, runner).run("What is my name?")

        self.assertIn("The user's name is Omer.", runner.goals[0].description)

    def test_deletion_persists_across_new_store_instance(self) -> None:
        store_a = self.store()
        memory = store_a.store("Temporary fact")
        store_a.delete(memory.id)

        store_b = self.store()

        self.assertEqual(store_b.list_all(), [])


if __name__ == "__main__":
    unittest.main()
